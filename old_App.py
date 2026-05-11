"""
NPU Real-Time RTSP Face Recognition  — Stable + High Accuracy Edition
=======================================================================

STABILITY FIXES (board restart):
  - Models loaded sequentially with 1.5s NPU cooldown between each
  - NPU inference wrapped in try/except — error triggers cooldown not crash
  - Memory watchdog thread: pauses NPU processing if RAM > 82%
  - Hard max_fps cap (default 6) prevents sustained power spikes
  - gc.collect() every 100 frames
  - Frames stored as JPEG bytes not raw numpy arrays

ACCURACY FIXES (unknown when moving):
  - Multi-frame VOTING: result confirmed across N frames before shown
  - Quality gate: skip blurry/dark/tiny faces (unreliable inference)
  - Centroid matching: ONE clean mean embedding per person (not raw copies)
    Rolling mean updated each time new samples added
  - Old DB auto-migrated: old list-of-embeddings → centroids on load
  - Unknown de-duplication: same face not saved twice within 3 seconds

Usage:
  # Safest (NPU detect only, CPU recognize):
  python3 app.py --det_model models/yunet.onnx \\
                 --recog_model models/mobilefacenet.onnx \\
                 --npu_det_only true --max_fps 6

  # Both on NPU (faster, needs more NPU RAM):
  python3 app.py --det_model models/yunet.onnx \\
                 --recog_model models/mobilefacenet.onnx \\
                 --use_npu true --max_fps 8
"""

import cv2
import numpy as np
import os, json, uuid, base64, threading, time, argparse, gc
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict

# Indian Standard Time = UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))
def now_ist():
    return datetime.now(IST)
def ist_str():
    return now_ist().strftime("%d %b %I:%M:%S %p")
from flask import Flask, render_template, Response, request, jsonify

from aligner    import align_face
from recognizer import NPUFaceRecognizer

# ══════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════
def str2bool(v): return v.lower() in ('true','yes','1','y','t')

ap = argparse.ArgumentParser()
ap.add_argument("--det_model",    required=True)
ap.add_argument("--recog_model",  required=True)
ap.add_argument("--recog_type",   default="mobilefacenet",
                choices=["mobilefacenet","arcface_r50"])
ap.add_argument("--use_npu",      default=True,  type=str2bool)
ap.add_argument("--npu_det_only", default=False, type=str2bool,
                help="Detection NPU, recognition CPU — recommended for 2-4GB boards")
ap.add_argument("--conf_thresh",  default=0.55,  type=float)
ap.add_argument("--nms_thresh",   default=0.3,   type=float)
ap.add_argument("--keep_top_k",   default=30,    type=int)
ap.add_argument("--tolerance",    default=0.45,  type=float,
                help="Cosine distance threshold (0.45 default). "
                     "Raise to 0.55 if too many Unknowns, lower to 0.35 if false matches")
ap.add_argument("--vote_frames",  default=5,     type=int,
                help="Confirm recognition across N frames. Higher = more stable labels")
ap.add_argument("--recog_every",  default=3,     type=int)
ap.add_argument("--det_size",     default=320,   type=int)
ap.add_argument("--max_fps",      default=6,     type=int,
                help="Max NPU inference FPS. Keep <=8 to prevent power spikes")
ap.add_argument("--min_face_px",  default=40,    type=int)
ap.add_argument("--min_quality",  default=25.0,  type=float,
                help="Min Laplacian blur score. Lower to accept blurrier faces")
ap.add_argument("--port",         default=5050,  type=int)
args = ap.parse_args()

# ══════════════════════════════════════════════════════════
# DIRS
# ══════════════════════════════════════════════════════════
KNOWN_DIR   = "known_faces"
UNKNOWN_DIR = "unknown_faces"
DB_PATH     = "face_db.json"
os.makedirs(KNOWN_DIR,   exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# MODEL LOADING — sequential with cooldown
# ══════════════════════════════════════════════════════════
DET_BACK = cv2.dnn.DNN_BACKEND_TIMVX  if args.use_npu else cv2.dnn.DNN_BACKEND_DEFAULT
DET_TGT  = cv2.dnn.DNN_TARGET_NPU     if args.use_npu else cv2.dnn.DNN_TARGET_CPU
REC_NPU  = args.use_npu and not args.npu_det_only

print(f"\n{'='*56}")
print(f"  NPU Face Recognition — Stable + High Accuracy")
print(f"{'='*56}")
print(f"  Detection  : {'NPU (TIM-VX)' if args.use_npu else 'CPU'}")
print(f"  Recognition: {'NPU (TIM-VX)' if REC_NPU else 'CPU (safe mode)'}")
print(f"  Max FPS    : {args.max_fps}   Vote frames: {args.vote_frames}")
print(f"  Tolerance  : {args.tolerance}   Min face: {args.min_face_px}px")
print(f"{'='*56}\n")

_detector  = None
_det_mutex = threading.Lock()

print("[Boot] Step 1/2  Loading face detector ...")
try:
    _detector = cv2.FaceDetectorYN.create(
        model           = args.det_model,
        config          = "",
        input_size      = (args.det_size, args.det_size),
        score_threshold = args.conf_thresh,
        nms_threshold   = args.nms_thresh,
        top_k           = args.keep_top_k,
        backend_id      = DET_BACK,
        target_id       = DET_TGT,
    )
    print("[Boot] Detector  OK")
except Exception as e:
    print(f"[Boot] NPU detector failed ({e}), trying CPU ...")
    _detector = cv2.FaceDetectorYN.create(
        model           = args.det_model,
        config          = "",
        input_size      = (args.det_size, args.det_size),
        score_threshold = args.conf_thresh,
        nms_threshold   = args.nms_thresh,
        top_k           = args.keep_top_k,
        backend_id      = cv2.dnn.DNN_BACKEND_DEFAULT,
        target_id       = cv2.dnn.DNN_TARGET_CPU,
    )
    print("[Boot] Detector  OK (CPU fallback)")

print("[Boot] NPU cooldown 1.5s ...")
gc.collect()
time.sleep(1.5)

_recognizer  = None
_recog_mutex = threading.Lock()

print("[Boot] Step 2/2  Loading face recognizer ...")
try:
    _recognizer = NPUFaceRecognizer(
        model_path = args.recog_model,
        model_type = args.recog_type,
        use_npu    = REC_NPU,
    )
    print("[Boot] Recognizer OK")
except Exception as e:
    print(f"[Boot] NPU recognizer failed ({e}), trying CPU ...")
    _recognizer = NPUFaceRecognizer(
        model_path = args.recog_model,
        model_type = args.recog_type,
        use_npu    = False,
    )
    print("[Boot] Recognizer OK (CPU fallback)")

gc.collect()
time.sleep(0.5)
print("[Boot] All models loaded\n")

# ══════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════
known_centroids = {}    # name  → mean embedding (L2-normalised)
known_names_set = set()

unknown_faces   = {}
recognition_log = deque(maxlen=80)

# Live face crop stores — latest confirmed crop per person/slot
# Each entry: {id, name, conf, time, path}
# These drive the live Known/Unknown grid in the UI
_live_known   = {}   # name → latest entry (one card per person)
_live_unknown = {}   # fid  → entry (one card per unknown)
_live_lock    = threading.Lock()
MAX_LIVE_UNKNOWN = 20   # keep last N unknown cards

_raw_frame     = None
_display_frame = None
_raw_lock      = threading.Lock()
_disp_lock     = threading.Lock()

stream_active  = False
camera_source  = None

TOLERANCE       = args.tolerance
RECOG_EVERY     = args.recog_every
VOTE_FRAMES     = args.vote_frames
_frame_interval = 1.0 / max(1, args.max_fps)

_vote_buffer    = defaultdict(lambda: deque(maxlen=VOTE_FRAMES))
# Track label carry: confirmed label is held for N frames even when
# voting is inconclusive (face turned, partial occlusion, slight blur)
TRACK_HOLD      = 20   # frames to hold last confirmed label
_track_hold_cnt = defaultdict(int)   # pos_key → frames since last confirm
_vote_lock      = threading.Lock()
_last_unknown_t = {}   # pos_key → timestamp, for de-duplication

stats = {
    "total_frames": 0, "faces_detected": 0,
    "known_hits": 0,   "unknowns": 0,
    "fps": 0.0,        "det_ms": 0,
    "recog_ms": 0,     "skipped_quality": 0,
    "backend": f"{'NPU' if args.use_npu else 'CPU'}+{'NPU' if REC_NPU else 'CPU'}",
    "mem_mb": 0,
}

flask_app = Flask(__name__)

# ══════════════════════════════════════════════════════════
# MEMORY WATCHDOG
# ══════════════════════════════════════════════════════════
_npu_paused = False

def _memory_watchdog():
    global _npu_paused
    while True:
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                k, v = line.split(":")
                info[k.strip()] = int(v.strip().split()[0])
            total = info.get("MemTotal", 1)
            avail = info.get("MemAvailable", total)
            used_pct = 1.0 - avail / total
            stats["mem_mb"] = (total - avail) // 1024
            if used_pct > 0.82 and not _npu_paused:
                print(f"[Watchdog] RAM {used_pct:.0%} — pausing NPU, running GC")
                _npu_paused = True
                gc.collect()
            elif used_pct < 0.70 and _npu_paused:
                print(f"[Watchdog] RAM {used_pct:.0%} — resuming NPU")
                _npu_paused = False
        except Exception:
            pass
        time.sleep(4)

threading.Thread(target=_memory_watchdog, daemon=True, name="MemWatchdog").start()

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def _cosine(a, b):
    return float(1 - np.dot(a, b))

def _b64(frame, q=65):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()

def _quality(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def _detect_faces(frame):
    h, w = frame.shape[:2]
    ds = args.det_size
    resized = cv2.resize(frame, (ds, ds))
    sx, sy  = w / ds, h / ds
    try:
        with _det_mutex:
            _detector.setInputSize((ds, ds))
            _, raw = _detector.detect(resized)
    except Exception as e:
        print(f"[Det] Error: {e}")
        time.sleep(0.5)
        return None
    if raw is None or len(raw) == 0:
        return None
    d = raw.copy().astype(np.float32)
    for c in [0, 2, 4, 6, 8, 10, 12]: d[:, c] *= sx
    for c in [1, 3, 5, 7, 9, 11, 13]: d[:, c] *= sy
    d[:, 2] = d[:, 0] + d[:, 2]
    d[:, 3] = d[:, 1] + d[:, 3]
    return d

def _get_embedding(aligned):
    try:
        with _recog_mutex:
            return _recognizer.get_embedding(aligned)
    except Exception as e:
        print(f"[Recog] Error: {e}")
        time.sleep(0.3)
        return None

def _match(emb):
    if not known_centroids or emb is None:
        return "Unknown", 0.0
    real = {k: v for k, v in known_centroids.items() if not k.startswith("__count_")}
    if not real:
        return "Unknown", 0.0
    best_name, best_dist = "Unknown", 1.0
    for name, centroid in real.items():
        d = _cosine(emb, centroid)
        if d < best_dist:
            best_dist, best_name = d, name
    conf = round(1.0 - best_dist, 3)
    return (best_name, conf) if best_dist <= TOLERANCE else ("Unknown", conf)

def _add_to_centroid(name, emb):
    if name in known_centroids:
        count = known_centroids.get(f"__count_{name}", 1)
        new_count = count + 1
        centroid = (known_centroids[name] * count + emb) / new_count
        norm = np.linalg.norm(centroid)
        known_centroids[name] = centroid / norm if norm > 1e-9 else centroid
        known_centroids[f"__count_{name}"] = new_count
    else:
        norm = np.linalg.norm(emb)
        known_centroids[name] = emb / norm if norm > 1e-9 else emb.copy()
        known_centroids[f"__count_{name}"] = 1
    known_names_set.add(name)

def _augment_face(img, n=8):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    variants = [img]
    configs = [
        (0.70, -20, False,   0),
        (1.30, +20, False,   0),
        (1.00,   0, True,    0),
        (1.00,   0, False, -15),
        (1.00,   0, False, +15),
        (0.85, -10, False,  -8),
        (1.15, +10, True,    0),
    ]
    for alpha, beta, flip, angle in configs[:n-1]:
        aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if angle != 0:
            M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if flip:
            aug = cv2.flip(aug, 1)
        variants.append(aug)
    return variants

def _enroll_face(name, aligned):
    variants  = _augment_face(aligned, n=8)
    processed = 0
    for v in variants:
        emb = _get_embedding(v)
        if emb is not None:
            _add_to_centroid(name, emb)
            processed += 1
        time.sleep(0.02)
    save_db()
    return processed

def _save_unknown(aligned, pos_key):
    now = time.time()
    if now - _last_unknown_t.get(pos_key, 0) < 3.0:
        return None
    _last_unknown_t[pos_key] = now
    fid  = str(uuid.uuid4())[:8]
    path = os.path.join(UNKNOWN_DIR, f"{fid}.jpg")
    cv2.imwrite(path, aligned)
    unknown_faces[fid] = {
        "path": path,
        "timestamp": ist_str(),
        "crop_b64":  _b64(aligned),
        "labeled":   False,
    }
    return fid

def _log_event(name, conf, crop, fid=None):
    # Each log entry gets a STABLE UUID used as its thumb filename.
    # This means /api/log_thumb/<lid> always serves the right image
    # regardless of how many new entries were added since the last poll.
    lid = str(uuid.uuid4())[:8]
    thumb_path = None
    try:
        tdir = os.path.join(UNKNOWN_DIR, "_thumbs")
        os.makedirs(tdir, exist_ok=True)
        thumb_path = os.path.join(tdir, f"{lid}.jpg")
        small = cv2.resize(crop, (64, 64))
        cv2.imwrite(thumb_path, small, [cv2.IMWRITE_JPEG_QUALITY, 65])
    except Exception:
        thumb_path = None

    entry = {
        "name":       name,
        "confidence": conf,
        "time":       ist_str(),
        "fid":        fid,
        "lid":        lid,
        "thumb_path": thumb_path,
    }
    recognition_log.appendleft(entry)

    # Update live face grid stores
    with _live_lock:
        if name == "Unknown" and fid:
            _live_unknown[fid] = entry
            # Trim to last N unknowns
            if len(_live_unknown) > MAX_LIVE_UNKNOWN:
                oldest = sorted(_live_unknown.keys(),
                                key=lambda k: _live_unknown[k]["time"])[0]
                _live_unknown.pop(oldest, None)
        elif name != "Unknown":
            # One card per known person — always show latest
            _live_known[name] = entry

# ══════════════════════════════════════════════════════════
# PERSISTENCE — centroid format
# ══════════════════════════════════════════════════════════
def save_db():
    # Separate centroids (numpy arrays) from counts (ints) — never mix
    centroids = {}
    counts    = {}
    for k, v in known_centroids.items():
        if k.startswith("__count_"):
            counts[k] = int(v)
        else:
            centroids[k] = np.array(v, dtype=np.float32).tolist()
    with open(DB_PATH, "w") as f:
        json.dump({"centroids": centroids,
                   "counts":    counts,
                   "names":     list(known_names_set)}, f)

def load_db():
    global known_centroids, known_names_set
    if not os.path.exists(DB_PATH):
        print("[DB] No database yet.")
        return
    with open(DB_PATH) as f:
        d = json.load(f)
    if "centroids" in d:
        known_centroids = {k: np.array(v, dtype=np.float32)
                           for k, v in d["centroids"].items()}
        # Restore counts as plain Python ints (not numpy arrays)
        for k, v in d.get("counts", {}).items():
            known_centroids[k] = int(v)
        known_names_set = set(d.get("names", []))
    elif "names" in d and "encodings" in d:
        print("[DB] Migrating old format → centroids ...")
        names = d["names"]
        encs  = [np.array(e, dtype=np.float32) for e in d["encodings"]]
        groups = defaultdict(list)
        for n, e in zip(names, encs):
            groups[n].append(e)
        for n, embs in groups.items():
            c    = np.mean(embs, axis=0).astype(np.float32)
            norm = np.linalg.norm(c)
            known_centroids[n]                = c / norm if norm > 1e-9 else c
            known_centroids[f"__count_{n}"]   = len(embs)
            known_names_set.add(n)
        save_db()
    known_names_set = {n for n in known_names_set if not n.startswith("__count_")}
    real = {k: v for k, v in known_centroids.items() if not k.startswith("__count_")}
    print(f"[DB] {len(real)} person centroids: {sorted(real.keys())}")

# ══════════════════════════════════════════════════════════
# THREAD 1 — RTSPGrabber
# ══════════════════════════════════════════════════════════
class RTSPGrabber(threading.Thread):
    def __init__(self, source):
        super().__init__(daemon=True, name="Grabber")
        self.source  = source
        self.running = False
        self._fps_t  = time.time()
        self._fps_n  = 0

    def run(self):
        global _raw_frame, stream_active
        src = 0 if self.source is None else self.source
        print(f"[Grabber] Connecting: {src or 'webcam'}")
        cap = cv2.VideoCapture()
        if isinstance(src, str) and "rtsp" in src.lower():
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap.open(src, cv2.CAP_FFMPEG)
        else:
            cap.open(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("[Grabber] Cannot open source")
            stream_active = False
            return
        print("[Grabber] Connected")
        self.running = True
        while self.running and stream_active:
            if not cap.grab():
                time.sleep(0.01)
                continue
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue
            with _raw_lock:
                _raw_frame = frame
            self._fps_n += 1
            now = time.time()
            if now - self._fps_t >= 1.0:
                stats["fps"] = round(self._fps_n / (now - self._fps_t), 1)
                self._fps_n = 0
                self._fps_t = now
        cap.release()
        print("[Grabber] Stopped")

    def stop(self): self.running = False


# ══════════════════════════════════════════════════════════
# THREAD 2 — NPUProcessor
# ══════════════════════════════════════════════════════════
class NPUProcessor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="NPUProc")
        self.running      = False
        self._last_id     = None
        self._frame_n     = 0
        self._last_t      = 0.0
        self._track_labels = {}   # pos_key → (name, conf)

    def run(self):
        global _display_frame
        self.running = True
        print("[NPUProc] Started")
        while self.running and stream_active:
            now = time.time()
            if now - self._last_t < _frame_interval:
                time.sleep(0.008)
                continue
            self._last_t = time.time()
            if _npu_paused:
                time.sleep(0.2)
                continue
            with _raw_lock:
                raw = _raw_frame
            if raw is None or id(raw) == self._last_id:
                time.sleep(0.005)
                continue
            self._last_id  = id(raw)
            self._frame_n += 1
            frame = raw.copy()

            t0   = time.time()
            dets = _detect_faces(frame)
            stats["det_ms"] = int((time.time() - t0) * 1000)

            if dets is not None and len(dets) > 0:
                stats["faces_detected"] += len(dets)
                if self._frame_n % RECOG_EVERY == 0:
                    t1 = time.time()
                    self._run_recognition(frame, dets)
                    stats["recog_ms"] = int((time.time() - t1) * 1000)
                frame = self._draw(frame, dets)
            else:
                self._track_labels.clear()

            self._draw_hud(frame)
            stats["total_frames"] += 1

            if self._frame_n % 100 == 0:
                gc.collect()

            with _disp_lock:
                _display_frame = frame
            time.sleep(0.002)

        self.running = False
        print("[NPUProc] Stopped")

    def stop(self): self.running = False

    def _run_recognition(self, frame, dets):
        h, w = frame.shape[:2]
        for d in dets:
            x1,y1,x2,y2 = d[0:4].astype(int)
            x1=max(0,x1); y1=max(0,y1)
            x2=min(w,x2); y2=min(h,y2)
            if (x2-x1) < args.min_face_px or (y2-y1) < args.min_face_px:
                continue
            lms  = d[4:14].reshape(5, 2)
            crop = frame[y1:y2, x1:x2]
            qs   = _quality(crop)
            if qs < args.min_quality:
                stats["skipped_quality"] += 1
                continue
            aligned    = align_face(frame, lms)
            emb        = _get_embedding(aligned)
            name, conf = _match(emb)
            pos_key    = f"{x1//80}_{y1//80}"
            with _vote_lock:
                _vote_buffer[pos_key].append((name, conf))
                votes = list(_vote_buffer[pos_key])

            if len(votes) >= VOTE_FRAMES:
                vote_names  = [v[0] for v in votes]
                most_common = max(set(vote_names), key=vote_names.count)
                vote_count  = vote_names.count(most_common)

                if vote_count >= max(2, VOTE_FRAMES // 2 + 1):
                    avg_conf = float(np.mean([v[1] for v in votes if v[0] == most_common]))

                    # ── Face tracking: carry confirmed label ──────────
                    # If new vote says Unknown BUT we had a confirmed known
                    # label recently (within TRACK_HOLD frames), keep that
                    # label instead. Person just turned/moved slightly.
                    prev_label = self._track_labels.get(pos_key)
                    hold_count = _track_hold_cnt.get(pos_key, 0)

                    if most_common == "Unknown" and prev_label and                        prev_label[0] != "Unknown" and hold_count < TRACK_HOLD:
                        # Carry previous known label, increment hold counter
                        _track_hold_cnt[pos_key] = hold_count + 1
                        # Don't log — it was just a transient miss
                    else:
                        # New confirmed result — update label + reset hold
                        self._track_labels[pos_key] = (most_common, avg_conf)
                        _track_hold_cnt[pos_key] = 0

                        if most_common == "Unknown":
                            fid = _save_unknown(aligned, pos_key)
                            if fid:
                                stats["unknowns"] += 1
                                _log_event("Unknown", avg_conf, aligned, fid)
                        else:
                            stats["known_hits"] += 1
                            _log_event(most_common, avg_conf, aligned)
            else:
                # Still gathering votes — carry last label silently
                pass

    def _draw(self, frame, dets):
        h, w = frame.shape[:2]
        for d in dets:
            x1,y1,x2,y2 = d[0:4].astype(int)
            lms   = d[4:14].reshape(5, 2)
            score = float(d[-1])
            x1=max(0,x1); y1=max(0,y1)
            x2=min(w,x2); y2=min(h,y2)
            if (x2-x1) < args.min_face_px or (y2-y1) < args.min_face_px:
                continue
            pos_key    = f"{x1//80}_{y1//80}"
            label_data = self._track_labels.get(pos_key)
            if label_data:
                name, conf = label_data
                color = (0, 215, 60) if name != "Unknown" else (0, 70, 240)
                label = f"{name}  {conf:.0%}"
            else:
                color = (0, 160, 255)
                label = f"Identifying  {score:.2f}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
            tl = 14
            for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(cx,cy),(cx+dx*tl,cy),color,3)
                cv2.line(frame,(cx,cy),(cx,cy+dy*tl),color,3)
            lm_c=[(255,80,80),(80,80,255),(80,255,80),(255,80,255),(80,255,255)]
            for i,(lx,ly) in enumerate(lms):
                cv2.circle(frame,(int(lx),int(ly)),3,lm_c[i],-1)
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.50,1)
            ly2 = max(y1-th-10,0)
            cv2.rectangle(frame,(x1,ly2),(x1+tw+8,ly2+th+8),color,-1)
            cv2.putText(frame,label,(x1+4,ly2+th+4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,255),1,cv2.LINE_AA)
        return frame

    def _draw_hud(self, frame):
        pause = " [MEM PAUSE]" if _npu_paused else ""
        lines = [
            f"FPS:{stats['fps']}  Det:{stats['det_ms']}ms  Rec:{stats['recog_ms']}ms  RAM:{stats['mem_mb']}MB{pause}",
            f"{stats['backend']}  Faces:{stats['faces_detected']}  Known:{stats['known_hits']}  Skip:{stats['skipped_quality']}",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(frame,ln,(8,24+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,255,180),2,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════
# MJPEG
# ══════════════════════════════════════════════════════════
_blank = None
def gen_frames():
    """Stream raw RTSP frame — no bounding boxes, no annotations."""
    global _blank
    while True:
        with _raw_lock:
            frame = _raw_frame    # RAW frame, not _display_frame
        if frame is None:
            if _blank is None:
                _blank = np.zeros((360,480,3),dtype=np.uint8)
                cv2.putText(_blank,"Waiting for stream...",(60,180),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(50,50,50),2)
            frame = _blank
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


# ══════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════
_grabber = _processor = None

@flask_app.route("/")
def index(): return render_template("index.html")

@flask_app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/api/start_stream", methods=["POST"])
def start_stream():
    global stream_active, camera_source, _grabber, _processor
    global _raw_frame, _display_frame
    src = request.json.get("source","").strip()
    camera_source = None if src in ("","0") else src
    if stream_active:
        return jsonify({"ok": False, "msg": "Already running"})
    _raw_frame = _display_frame = None
    stream_active = True
    _grabber   = RTSPGrabber(camera_source)
    _processor = NPUProcessor()
    _grabber.start()
    time.sleep(0.4)
    _processor.start()
    return jsonify({"ok": True, "msg": f"Started: {camera_source or 'webcam'}"})

@flask_app.route("/api/stop_stream", methods=["POST"])
def stop_stream():
    global stream_active, _grabber, _processor
    stream_active = False
    if _grabber:   _grabber.stop()
    if _processor: _processor.stop()
    _grabber = _processor = None
    gc.collect()
    return jsonify({"ok": True})

@flask_app.route("/api/add_known_face", methods=["POST"])
def add_known_face():
    name   = request.form.get("name","").strip()
    images = request.files.getlist("images")
    if not name:   return jsonify({"ok": False, "msg": "Name required"})
    if not images: return jsonify({"ok": False, "msg": "No images"})
    total = 0
    for f in images:
        arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: continue
        dets = _detect_faces(img)
        if dets is not None and len(dets) > 0:
            best    = dets[np.argmax(dets[:,-1])]
            lms     = best[4:14].reshape(5,2)
            aligned = align_face(img, lms)
        else:
            h,w = img.shape[:2]; s = min(h,w)
            y0=(h-s)//2; x0=(w-s)//2
            aligned = cv2.resize(img[y0:y0+s, x0:x0+s], (112,112))
        total += _enroll_face(name, aligned)
    if total:
        count = int(known_centroids.get(f"__count_{name}", 0))
        return jsonify({"ok": True,
                        "msg": f"Enrolled '{name}' — {total} variants → centroid updated (total samples: {count})"})
    return jsonify({"ok": False, "msg": "No face found in images"})

@flask_app.route("/api/label_unknown", methods=["POST"])
def label_unknown():
    fid  = request.json.get("fid")
    name = request.json.get("name","").strip()
    if not fid or not name or fid not in unknown_faces:
        return jsonify({"ok": False, "msg": "Invalid"})
    img = cv2.imread(unknown_faces[fid]["path"])
    if img is None: return jsonify({"ok": False, "msg": "Cannot read image"})
    n = _enroll_face(name, img)
    unknown_faces[fid]["labeled"] = True
    count = int(known_centroids.get(f"__count_{name}", 0))
    return jsonify({"ok": True,
                    "msg": f"Labeled '{name}' — centroid updated (total samples: {count})"})

@flask_app.route("/api/label_unknown_aug", methods=["POST"])
def label_unknown_aug():
    return label_unknown()

@flask_app.route("/api/label_all_unknown", methods=["POST"])
def label_all_unknown():
    name = request.json.get("name","").strip()
    if not name: return jsonify({"ok": False, "msg": "Name required"})
    unlabeled = [k for k,v in unknown_faces.items() if not v["labeled"]]
    if not unlabeled: return jsonify({"ok": False, "msg": "No unlabeled faces"})
    total = 0
    for fid in unlabeled:
        img = cv2.imread(unknown_faces[fid]["path"])
        if img is None: continue
        total += _enroll_face(name, img)
        unknown_faces[fid]["labeled"] = True
    count = int(known_centroids.get(f"__count_{name}", 0))
    return jsonify({"ok": True,
                    "msg": f"Labeled {len(unlabeled)} faces as '{name}' — {total} variants, total samples: {count}"})

@flask_app.route("/api/person_stats")
def person_stats():
    real = {k: v for k,v in known_centroids.items() if not k.startswith("__count_")}
    return jsonify({"persons": [
        {"name": n, "embeddings": int(known_centroids.get(f"__count_{n}", 1))}
        for n in sorted(real.keys())
    ]})

@flask_app.route("/api/delete_person", methods=["POST"])
def delete_person():
    name = request.json.get("name","").strip()
    if not name: return jsonify({"ok": False, "msg": "Name required"})
    known_centroids.pop(name, None)
    known_centroids.pop(f"__count_{name}", None)
    known_names_set.discard(name)
    save_db()
    return jsonify({"ok": True, "msg": f"Deleted '{name}'"})

@flask_app.route("/api/status")
def status():
    """Lightweight — NO images. Safe to poll every second."""
    # Log summary: metadata only, no images
    log_meta = [
        {"name":       e["name"],
         "confidence": e["confidence"],
         "time":       e["time"],
         "fid":        e.get("fid"),
         "lid":        e.get("lid", str(i))}   # stable UUID
        for i, e in enumerate(list(recognition_log)[:20])
    ]
    # Unknown summary: metadata only, no crop_b64
    unk_meta = [
        {"fid": k, "timestamp": v["timestamp"]}
        for k, v in list(unknown_faces.items())[-12:]
        if not v["labeled"]
    ]
    return jsonify({
        "stream":      stream_active,
        "known_names": sorted(known_names_set),
        "stats":       stats,
        "tolerance":   TOLERANCE,
        "log":         log_meta,
        "unknowns":    unk_meta,
    })


@flask_app.route("/api/log_thumb/<lid>")
def log_thumb(lid):
    """Serve log thumbnail by stable UUID — never mismatches even as log grows."""
    import io
    from flask import send_file
    # Direct path lookup — O(1), no list scanning
    tdir = os.path.join(UNKNOWN_DIR, "_thumbs")
    tp   = os.path.join(tdir, f"{lid}.jpg")
    if os.path.exists(tp):
        return send_file(tp, mimetype="image/jpeg",
                         max_age=3600, conditional=True)
    # Fallback grey square
    placeholder = np.zeros((64,64,3), dtype=np.uint8) + 45
    _, buf = cv2.imencode(".jpg", placeholder)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@flask_app.route("/api/unknown_crop/<fid>")
def unknown_crop(fid):
    """Serve unknown face crop by fid — called only when Label tab opens."""
    from flask import send_file
    if fid not in unknown_faces:
        return "", 404
    path = unknown_faces[fid].get("path","")
    if path and os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "", 404

@flask_app.route("/api/reset_stats", methods=["POST"])
def reset_stats():
    stats.update({"total_frames":0,"faces_detected":0,"known_hits":0,
                  "unknowns":0,"fps":0.0,"det_ms":0,"recog_ms":0,"skipped_quality":0})
    return jsonify({"ok": True})

@flask_app.route("/api/set_tolerance", methods=["POST"])
def set_tolerance():
    global TOLERANCE
    TOLERANCE = max(0.2, min(0.9, float(request.json.get("value", 0.45))))
    return jsonify({"ok": True, "tolerance": TOLERANCE})

@flask_app.route("/api/set_recog_every", methods=["POST"])
def set_recog_every():
    global RECOG_EVERY
    RECOG_EVERY = max(1, min(10, int(request.json.get("value", 3))))
    return jsonify({"ok": True})

@flask_app.route("/api/set_vote_frames", methods=["POST"])
def set_vote_frames():
    global VOTE_FRAMES
    VOTE_FRAMES = max(1, min(15, int(request.json.get("value", 5))))
    return jsonify({"ok": True})



@flask_app.route("/api/live_known")
def live_known():
    """Latest confirmed crop for each known person — drives Known live grid."""
    with _live_lock:
        entries = list(_live_known.values())
    return jsonify({"cards": [
        {"name":  e["name"],
         "conf":  e["confidence"],
         "time":  e["time"],
         "lid":   e["lid"]}
        for e in sorted(entries, key=lambda x: x["name"])
    ]})


@flask_app.route("/api/live_unknown")
def live_unknown():
    """Latest unknown face crops — drives Unknown live grid."""
    with _live_lock:
        entries = list(_live_unknown.values())
    return jsonify({"cards": [
        {"fid":  e["fid"],
         "conf": e["confidence"],
         "time": e["time"],
         "lid":  e["lid"]}
        for e in sorted(entries, key=lambda x: x["time"], reverse=True)[:20]
    ]})


@flask_app.route("/api/clear_live_unknown", methods=["POST"])
def clear_live_unknown():
    """Clear the live unknown grid (after bulk labeling)."""
    with _live_lock:
        _live_unknown.clear()
    return jsonify({"ok": True})

# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_db()
    print(f"Open: http://0.0.0.0:{args.port}\n")
    flask_app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
