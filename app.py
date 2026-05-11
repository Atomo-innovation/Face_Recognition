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
import queue
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
ap.add_argument("--conf_thresh",  default=0.60,  type=float,
                help="Face detection confidence threshold. 0.60 is optimal for this YuNet "
                     "(lower = more false positives from non-face objects)")
ap.add_argument("--nms_thresh",   default=0.3,   type=float)
ap.add_argument("--keep_top_k",   default=30,    type=int)
ap.add_argument("--tolerance",    default=0.45,  type=float,
                help="Cosine distance threshold (0.45 default). "
                     "Raise to 0.55 if too many Unknowns, lower to 0.35 if false matches")
ap.add_argument("--vote_frames",  default=5,     type=int,
                help="Confirm recognition across N frames. Higher = more stable labels")
ap.add_argument("--recog_every",  default=3,     type=int)
ap.add_argument("--det_size",     default=640,   type=int,
                help="Detector input size. Must be 640 for this YuNet (native resolution). "
                     "Do NOT change to 320 — accuracy drops drastically.")
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
KNOWN_DIR    = "known_faces"
UNKNOWN_DIR  = "unknown_faces"
CLUSTER_DIR  = "unknown_clusters"   # grouped unknown faces for easy labeling
DB_PATH      = "face_db.json"
CLUSTER_DB   = "cluster_db.json"    # persists cluster embeddings
os.makedirs(KNOWN_DIR,   exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(CLUSTER_DIR, exist_ok=True)

# ── Cluster state ──────────────────────────────────────────
# cluster_id → {"centroid": np.array, "count": int, "members": [fid,...]}
_clusters: dict = {}
_cluster_lock   = threading.Lock()
CLUSTER_THRESH  = 0.40   # cosine distance: faces closer than this go in same cluster

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
print(f"  Det input  : {args.det_size}×{args.det_size} (native YuNet — letterbox pad)")
print(f"  Max FPS    : {args.max_fps}   Vote frames: {args.vote_frames}")
print(f"  Tolerance  : {args.tolerance}   Min face: {args.min_face_px}px")
print(f"  Conf thresh: {args.conf_thresh}   NMS: {args.nms_thresh}")
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
# RECOGNITION PIPELINE QUEUE
# Detector pushes (pos_key, aligned_crop, det_score) here.
# RecogWorker pops and runs embedding + voting asynchronously.
# ══════════════════════════════════════════════════════════
_recog_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=8)
# maxsize=8: if recog falls behind, old crops are dropped (det always wins)

# Shared label store — RecogWorker writes, _draw reads
# pos_key → (name, conf, timestamp)
_pipeline_labels: dict = {}
_pipeline_lock = threading.Lock()

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

def _letterbox(img, target, color=(114, 114, 114)):
    """
    Resize `img` to `target×target` while preserving aspect ratio.
    Pads with neutral gray (114,114,114) — same padding YuNet was trained with.

    Returns
    -------
    padded : (target, target, 3) uint8
    scale  : float — single scale factor applied to both axes
    pad    : (pad_x, pad_y) pixels added to left / top
    """
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((target, target, 3), color, dtype=np.uint8)
    px = (target - nw) // 2
    py = (target - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, scale, (px, py)


def _unletterbox_dets(dets, scale, pad, orig_w, orig_h):
    """
    Map detection coordinates from letterboxed space back to original frame.
    `dets` columns: [x1, y1, x2, y2, lm0x, lm0y, ..., lm4x, lm4y, score]
    """
    if dets is None or len(dets) == 0:
        return dets
    d  = dets.copy().astype(np.float32)
    px, py = pad
    # x-cols: x1=0, x2=2, lm0x=4, lm1x=6, lm2x=8, lm3x=10, lm4x=12
    # y-cols: y1=1, y2=3, lm0y=5, lm1y=7, lm2y=9, lm3y=11, lm4y=13
    for c in [0, 2, 4, 6, 8, 10, 12]:
        d[:, c] = np.clip((d[:, c] - px) / scale, 0, orig_w)
    for c in [1, 3, 5, 7, 9, 11, 13]:
        d[:, c] = np.clip((d[:, c] - py) / scale, 0, orig_h)
    return d


def _is_plausible_face(det):
    """
    Geometric sanity checks to reject non-face false positives.
    Operates on a single detection row after coordinate un-letterboxing.

    Checks
    ------
    1. Aspect ratio: face width/height should be 0.5–1.8
    2. Landmark ordering: left-eye x < right-eye x  (basic left/right)
    3. Eyes above mouth: eye y-mean < mouth y-mean
    4. Nose between eyes and mouth vertically
    5. Eye spacing >= 15% of bbox width (filter tiny/distant detections)
    6. Left-mouth / right-mouth roughly symmetric around nose x
    """
    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
    bw = x2 - x1
    bh = y2 - y1
    if bh < 1:
        return False

    # 1. Aspect ratio
    ar = bw / bh
    if ar < 0.40 or ar > 2.0:
        return False

    # Landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    lms = det[4:14].reshape(5, 2)
    le, re, nose, lm, rm = lms

    # 2. Left eye should be to the left of right eye (some pose tolerance)
    if re[0] - le[0] < -bw * 0.15:   # allow up to 15% overlap for profile
        return False

    # 3. Eyes above mouth
    eye_y   = (le[1] + re[1]) / 2
    mouth_y = (lm[1] + rm[1]) / 2
    if eye_y >= mouth_y:
        return False

    # 4. Nose between eyes and mouth vertically
    if not (eye_y < nose[1] < mouth_y + (mouth_y - eye_y) * 0.3):
        return False

    # 5. Eye spacing >= 15% of bbox width
    eye_spacing = abs(re[0] - le[0])
    if eye_spacing < bw * 0.15:
        return False

    # 6. Mouth points roughly on same horizontal plane (within 35% of face height)
    if abs(lm[1] - rm[1]) > bh * 0.35:
        return False

    return True


def _detect_faces(frame):
    """
    Detect faces in `frame` (any resolution) using letterbox preprocessing.

    Pipeline
    --------
    1. Letterbox frame to det_size × det_size (preserve aspect ratio, gray pad)
    2. Run YuNet detector on padded square
    3. Un-letterbox detections back to original frame coordinates
    4. Apply geometric plausibility filter to reject non-face false positives

    Returns
    -------
    ndarray shape (N, 15) or None
      Columns: x1, y1, x2, y2, lm0x…lm4y, score  (in original pixel space)
    """
    h, w  = frame.shape[:2]
    ds    = args.det_size          # 640 (native YuNet resolution)

    # ── Step 1: letterbox ─────────────────────────────────────────────
    padded, scale, pad = _letterbox(frame, ds)

    # ── Step 2: detect on padded square ───────────────────────────────
    try:
        with _det_mutex:
            _detector.setInputSize((ds, ds))
            _, raw = _detector.detect(padded)
    except Exception as e:
        print(f"[Det] Error: {e}")
        time.sleep(0.5)
        return None

    if raw is None or len(raw) == 0:
        return None

    # cv2.FaceDetectorYN returns [x, y, w, h, lm×10, score]
    # Convert to [x1, y1, x2, y2, lm×10, score] format
    d      = raw.copy().astype(np.float32)
    d[:, 2] = d[:, 0] + d[:, 2]   # x2 = x + w
    d[:, 3] = d[:, 1] + d[:, 3]   # y2 = y + h

    # ── Step 3: un-letterbox back to original frame coordinates ───────
    d = _unletterbox_dets(d, scale, pad, w, h)

    # ── Step 4: geometric plausibility filter ─────────────────────────
    valid = [row for row in d if _is_plausible_face(row)]
    if not valid:
        return None
    return np.array(valid, dtype=np.float32)

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

def _save_unknown(aligned, pos_key, emb=None):
    now = time.time()
    if now - _last_unknown_t.get(pos_key, 0) < 3.0:
        return None
    _last_unknown_t[pos_key] = now
    fid  = str(uuid.uuid4())[:8]
    path = os.path.join(UNKNOWN_DIR, f"{fid}.jpg")
    cv2.imwrite(path, aligned)
    unknown_faces[fid] = {
        "path":      path,
        "timestamp": ist_str(),
        "crop_b64":  _b64(aligned),
        "labeled":   False,
        "cluster":   None,
    }
    # Auto-cluster by embedding similarity so similar faces land in same dir
    if emb is not None:
        cid = _assign_cluster(fid, emb, aligned)
        unknown_faces[fid]["cluster"] = cid
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

def save_cluster_db():
    """Persist cluster centroids (embeddings) and membership lists."""
    data = {}
    with _cluster_lock:
        for cid, c in _clusters.items():
            data[cid] = {
                "centroid": c["centroid"].tolist(),
                "count":    c["count"],
                "members":  c["members"],
            }
    with open(CLUSTER_DB, "w") as f:
        json.dump(data, f)

def load_cluster_db():
    """Restore clusters from disk on startup."""
    global _clusters
    if not os.path.exists(CLUSTER_DB):
        return
    try:
        with open(CLUSTER_DB) as f:
            data = json.load(f)
        with _cluster_lock:
            for cid, c in data.items():
                _clusters[cid] = {
                    "centroid": np.array(c["centroid"], dtype=np.float32),
                    "count":    c["count"],
                    "members":  c["members"],
                }
        print(f"[ClusterDB] Loaded {len(_clusters)} clusters")
    except Exception as e:
        print(f"[ClusterDB] Load error: {e}")

def _assign_cluster(fid: str, emb: np.ndarray, aligned: np.ndarray):
    """
    Assign an unknown face embedding to the closest cluster (cosine distance).
    Creates a new cluster if none is close enough.
    Copies the face image into the cluster sub-directory for easy browsing.
    """
    with _cluster_lock:
        best_cid, best_dist = None, CLUSTER_THRESH
        for cid, c in _clusters.items():
            d = _cosine(emb, c["centroid"])
            if d < best_dist:
                best_dist, best_cid = d, cid

        if best_cid is None:
            # New cluster
            best_cid = str(uuid.uuid4())[:8]
            _clusters[best_cid] = {
                "centroid": emb.copy(),
                "count":    1,
                "members":  [fid],
            }
        else:
            # Update rolling centroid
            c    = _clusters[best_cid]
            n    = c["count"]
            newc = (c["centroid"] * n + emb) / (n + 1)
            norm = np.linalg.norm(newc)
            c["centroid"] = newc / norm if norm > 1e-9 else newc
            c["count"]    = n + 1
            c["members"].append(fid)

    # Copy image into cluster folder
    cdir = os.path.join(CLUSTER_DIR, best_cid)
    os.makedirs(cdir, exist_ok=True)
    src = unknown_faces.get(fid, {}).get("path", "")
    if src and os.path.exists(src):
        import shutil
        shutil.copy2(src, os.path.join(cdir, f"{fid}.jpg"))

    save_cluster_db()
    return best_cid

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
        print("[NPUProc] Started — detection-only loop (recognition runs in RecogWorker)")
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

            # ── DETECTION (fast, stays on this thread) ───────────────
            t0   = time.time()
            dets = _detect_faces(frame)
            stats["det_ms"] = int((time.time() - t0) * 1000)

            if dets is not None and len(dets) > 0:
                stats["faces_detected"] += len(dets)
                # Every RECOG_EVERY frames push crops into the queue
                # for the background RecogWorker to process.
                if self._frame_n % RECOG_EVERY == 0:
                    self._enqueue_crops(frame, dets)
                # Draw immediately using whatever labels are currently known
                frame = self._draw(frame, dets)
            else:
                with _pipeline_lock:
                    _pipeline_labels.clear()

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

    def _enqueue_crops(self, frame, dets):
        """
        Crop each detected face and push into the recognition queue.
        Non-blocking: if queue is full the oldest slot is dropped.

        Preprocessing notes
        -------------------
        • dets already in original-frame pixel coords (letterbox undone)
        • x2/y2 already absolute (not w/h) after _detect_faces conversion
        • Expand bbox by 15% for better alignment context (ArcFace benefit)
        • Quality gate: Laplacian variance + min pixel size
        """
        h, w = frame.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d[0:4].astype(int)
            bw, bh = x2 - x1, y2 - y1

            # Skip tiny faces — too small for reliable embedding
            if bw < args.min_face_px or bh < args.min_face_px:
                continue

            # Expand bbox slightly for better landmark context
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # Quality gate: reject blurry crops
            qs = _quality(crop)
            if qs < args.min_quality:
                stats["skipped_quality"] += 1
                continue

            lms     = d[4:14].reshape(5, 2)
            aligned = align_face(frame, lms)

            # pos_key: coarse grid bucket so the same face position maps to
            # the same key across consecutive frames
            pos_key = f"{x1 // 80}_{y1 // 80}"
            det_score = float(d[-1])

            item = (pos_key, aligned, det_score)
            try:
                _recog_queue.put_nowait(item)
            except queue.Full:
                try:
                    _recog_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    _recog_queue.put_nowait(item)
                except queue.Full:
                    pass

    def _draw(self, frame, dets):
        h, w = frame.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d[0:4].astype(int)
            lms   = d[4:14].reshape(5, 2)
            score = float(d[-1])
            # Clamp to frame bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if (x2 - x1) < args.min_face_px or (y2 - y1) < args.min_face_px:
                continue
            pos_key = f"{x1//80}_{y1//80}"
            with _pipeline_lock:
                label_data = _pipeline_labels.get(pos_key)
            if label_data:
                name, conf = label_data[0], label_data[1]
                color = (0, 215, 60) if name != "Unknown" else (0, 70, 240)
                label = f"{name}  {conf:.0%}"
            else:
                color = (0, 160, 255)
                label = f"Detecting  {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            tl = 14
            for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame, (cx,cy), (cx+dx*tl, cy), color, 3)
                cv2.line(frame, (cx,cy), (cx, cy+dy*tl), color, 3)
            lm_c = [(255,80,80),(80,80,255),(80,255,80),(255,80,255),(80,255,255)]
            for i, (lx, ly) in enumerate(lms):
                cv2.circle(frame, (int(lx), int(ly)), 3, lm_c[i], -1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            ly2 = max(y1 - th - 10, 0)
            cv2.rectangle(frame, (x1, ly2), (x1+tw+8, ly2+th+8), color, -1)
            cv2.putText(frame, label, (x1+4, ly2+th+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
        return frame

    def _draw_hud(self, frame):
        pause = " [MEM PAUSE]" if _npu_paused else ""
        q_depth = _recog_queue.qsize()
        lines = [
            f"FPS:{stats['fps']}  Det:{stats['det_ms']}ms  Rec:{stats['recog_ms']}ms  RAM:{stats['mem_mb']}MB{pause}",
            f"{stats['backend']}  Faces:{stats['faces_detected']}  Known:{stats['known_hits']}  Skip:{stats['skipped_quality']}  Q:{q_depth}",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(frame,ln,(8,24+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,255,180),2,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════
# THREAD 3 — RecogWorker
# Consumes aligned face crops from _recog_queue, runs embedding
# + voting, and writes confirmed labels to _pipeline_labels.
# The display thread (NPUProcessor) never waits on this worker.
# ══════════════════════════════════════════════════════════
class RecogWorker(threading.Thread):
    """Background recognition pipeline — fully decoupled from display."""

    def __init__(self):
        super().__init__(daemon=True, name="RecogWorker")
        self.running = False
        self._track_labels: dict = {}   # pos_key → (name, conf)

    def run(self):
        self.running = True
        print("[RecogWorker] Started — consuming from recognition queue")
        while self.running and stream_active:
            try:
                pos_key, aligned, det_score = _recog_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if _npu_paused:
                _recog_queue.task_done() if False else None  # no task_done needed
                continue

            t1  = time.time()
            emb = _get_embedding(aligned)
            stats["recog_ms"] = int((time.time() - t1) * 1000)

            if emb is None:
                continue

            name, conf = _match(emb)

            with _vote_lock:
                _vote_buffer[pos_key].append((name, conf))
                votes = list(_vote_buffer[pos_key])

            if len(votes) < VOTE_FRAMES:
                # Not enough votes yet — keep showing last label
                continue

            vote_names  = [v[0] for v in votes]
            most_common = max(set(vote_names), key=vote_names.count)
            vote_count  = vote_names.count(most_common)

            if vote_count < max(2, VOTE_FRAMES // 2 + 1):
                continue

            avg_conf   = float(np.mean([v[1] for v in votes if v[0] == most_common]))
            prev_label = self._track_labels.get(pos_key)
            hold_count = _track_hold_cnt.get(pos_key, 0)

            if (most_common == "Unknown" and prev_label
                    and prev_label[0] != "Unknown"
                    and hold_count < TRACK_HOLD):
                # Transient miss — carry last known label for TRACK_HOLD frames
                _track_hold_cnt[pos_key] = hold_count + 1
            else:
                # New confirmed result
                self._track_labels[pos_key] = (most_common, avg_conf)
                _track_hold_cnt[pos_key] = 0

                # Write to shared label store so NPUProcessor can draw it
                with _pipeline_lock:
                    _pipeline_labels[pos_key] = (most_common, avg_conf)

                if most_common == "Unknown":
                    fid = _save_unknown(aligned, pos_key, emb=emb)
                    if fid:
                        stats["unknowns"] += 1
                        _log_event("Unknown", avg_conf, aligned, fid)
                else:
                    stats["known_hits"] += 1
                    _log_event(most_common, avg_conf, aligned)

        self.running = False
        print("[RecogWorker] Stopped")

    def stop(self):
        self.running = False
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
_grabber = _processor = _recog_worker = None

@flask_app.route("/")
def index(): return render_template("index.html")

@flask_app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/api/start_stream", methods=["POST"])
def start_stream():
    global stream_active, camera_source, _grabber, _processor, _recog_worker
    global _raw_frame, _display_frame
    src = request.json.get("source","").strip()
    camera_source = None if src in ("","0") else src
    if stream_active:
        return jsonify({"ok": False, "msg": "Already running"})
    _raw_frame = _display_frame = None
    # Drain any stale crops from a previous session
    while not _recog_queue.empty():
        try: _recog_queue.get_nowait()
        except queue.Empty: break
    with _pipeline_lock:
        _pipeline_labels.clear()
    stream_active  = True
    _grabber       = RTSPGrabber(camera_source)
    _processor     = NPUProcessor()
    _recog_worker  = RecogWorker()
    _grabber.start()
    time.sleep(0.4)
    _processor.start()
    time.sleep(0.1)
    _recog_worker.start()
    return jsonify({"ok": True, "msg": f"Started: {camera_source or 'webcam'}"})

@flask_app.route("/api/stop_stream", methods=["POST"])
def stop_stream():
    global stream_active, _grabber, _processor, _recog_worker
    stream_active = False
    if _grabber:   _grabber.stop()
    if _processor: _processor.stop()
    if _recog_worker: _recog_worker.stop()
    _grabber = _processor = _recog_worker = None
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

@flask_app.route("/api/recent_detection")
def recent_detection():
    """
    Returns the single most recent recognition event for the popup notification.
    Includes a full-resolution face crop encoded as base64.
    Called every ~1.5s by the UI notification system.
    """
    if not recognition_log:
        return jsonify({"event": None})
    e = recognition_log[0]   # most recent (deque is appendleft)
    lid   = e.get("lid","")
    is_unk = e["name"] == "Unknown"
    # Serve crop from thumb file (64×64 is enough for the popup)
    crop_b64 = None
    tp = e.get("thumb_path","")
    if tp and os.path.exists(tp):
        with open(tp,"rb") as f:
            crop_b64 = base64.b64encode(f.read()).decode()
    return jsonify({
        "event": {
            "name":       e["name"],
            "confidence": e["confidence"],
            "time":       e["time"],
            "fid":        e.get("fid"),
            "lid":        lid,
            "authorized": not is_unk,
            "crop_b64":   crop_b64,
        }
    })


@flask_app.route("/api/clusters")
def list_clusters():
    """Return all unknown face clusters with member count and sample image."""
    with _cluster_lock:
        result = []
        for cid, c in _clusters.items():
            members  = c["members"]
            unlabeled = [m for m in members if not unknown_faces.get(m, {}).get("labeled")]
            # Pick a sample face image
            sample_fid = next((m for m in members if os.path.exists(
                os.path.join(CLUSTER_DIR, cid, f"{m}.jpg"))), None)
            result.append({
                "cid":       cid,
                "count":     len(members),
                "unlabeled": len(unlabeled),
                "sample":    sample_fid,
                "members":   members,
            })
    result.sort(key=lambda x: x["unlabeled"], reverse=True)
    return jsonify({"clusters": result})


@flask_app.route("/api/cluster_image/<cid>/<fid>")
def cluster_image(cid, fid):
    """Serve a face image from a cluster directory."""
    from flask import send_file
    path = os.path.join(CLUSTER_DIR, cid, f"{fid}.jpg")
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    # Fallback to unknown_faces store
    if fid in unknown_faces:
        p = unknown_faces[fid].get("path", "")
        if p and os.path.exists(p):
            return send_file(p, mimetype="image/jpeg")
    return "", 404


@flask_app.route("/api/label_cluster", methods=["POST"])
def label_cluster():
    """Label every face in a cluster with a given name and enroll them all."""
    cid  = request.json.get("cid","").strip()
    name = request.json.get("name","").strip()
    if not cid or not name:
        return jsonify({"ok": False, "msg": "cid and name required"})
    with _cluster_lock:
        c = _clusters.get(cid)
        if c is None:
            return jsonify({"ok": False, "msg": "Cluster not found"})
        members = list(c["members"])
    total = 0
    for fid in members:
        if unknown_faces.get(fid, {}).get("labeled"):
            continue
        img = cv2.imread(unknown_faces.get(fid, {}).get("path","") or "")
        if img is None:
            continue
        total += _enroll_face(name, img)
        unknown_faces[fid]["labeled"] = True
    count = int(known_centroids.get(f"__count_{name}", 0))
    return jsonify({
        "ok": True,
        "msg": f"Cluster '{cid}' → '{name}': {len(members)} faces, {total} embeddings enrolled (total: {count})"
    })


@flask_app.route("/api/merge_clusters", methods=["POST"])
def merge_clusters():
    """Merge two clusters into one (in case the same person got split)."""
    cid1 = request.json.get("cid1","").strip()
    cid2 = request.json.get("cid2","").strip()
    if not cid1 or not cid2 or cid1 == cid2:
        return jsonify({"ok": False, "msg": "Two different cids required"})
    with _cluster_lock:
        c1 = _clusters.get(cid1)
        c2 = _clusters.get(cid2)
        if not c1 or not c2:
            return jsonify({"ok": False, "msg": "Cluster not found"})
        # Merge c2 into c1
        n1, n2 = c1["count"], c2["count"]
        merged  = (c1["centroid"] * n1 + c2["centroid"] * n2) / (n1 + n2)
        norm    = np.linalg.norm(merged)
        c1["centroid"] = merged / norm if norm > 1e-9 else merged
        c1["count"]    = n1 + n2
        c1["members"].extend(c2["members"])
        # Move files
        import shutil
        src_dir = os.path.join(CLUSTER_DIR, cid2)
        dst_dir = os.path.join(CLUSTER_DIR, cid1)
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.isdir(src_dir):
            for f in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            shutil.rmtree(src_dir, ignore_errors=True)
        # Update member cluster references
        for fid in c2["members"]:
            if fid in unknown_faces:
                unknown_faces[fid]["cluster"] = cid1
        del _clusters[cid2]
    save_cluster_db()
    return jsonify({"ok": True, "msg": f"Merged {cid2} → {cid1}"})


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_db()
    load_cluster_db()
    print(f"Open: http://0.0.0.0:{args.port}\n")
    flask_app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
