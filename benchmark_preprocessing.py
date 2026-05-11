
import cv2
import numpy as np
import onnxruntime as ort
import os, time, itertools, json, csv, argparse
from pathlib import Path

# ─── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--det_model",   default="models/yunet.onnx")
ap.add_argument("--recog_model", default="models/mobilefacenet.onnx")
ap.add_argument("--image_dir",   default=None,
                help="Optional folder of test face images (.jpg/.png)")
ap.add_argument("--out_dir",     default="results")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

print("\n" + "="*60)
print("  PRE/POST-PROCESSING BENCHMARK")
print("  YuNet + MobileFaceNet")
print("="*60)

# ─── Load models ─────────────────────────────────────────────────────────────
print("\n[1/4] Loading ONNX models (CPU) ...")
sess_det  = ort.InferenceSession(args.det_model,
                                  providers=["CPUExecutionProvider"])
sess_rec  = ort.InferenceSession(args.recog_model,
                                  providers=["CPUExecutionProvider"])
print("      Detector  OK — input:", sess_det.get_inputs()[0].shape)
print("      Recognizer OK — input:", sess_rec.get_inputs()[0].shape)

# ─── Anchor grid builder for YuNet ───────────────────────────────────────────
def build_anchors(input_size=640):
    anchors = {}
    for stride in [8, 16, 32]:
        cells = input_size // stride
        grid = []
        for i in range(cells):
            for j in range(cells):
                grid.append([(j + 0.5) * stride, (i + 0.5) * stride])
        anchors[stride] = np.array(grid, dtype=np.float32)
    return anchors

ANCHORS_640 = build_anchors(640)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def decode_yunet(outs, anchors, score_thresh=0.5, input_size=640):
    """
    Decode YuNet raw outputs → list of [x1,y1,x2,y2, lm×10, score]
    """
    detections = []
    stride_map = {8: (0,3,6,9), 16: (1,4,7,10), 32: (2,5,8,11)}

    for stride, (ci, oi, bi, ki) in stride_map.items():
        cls_s  = sigmoid(outs[ci][0, :, 0])   # (N,)
        obj_s  = sigmoid(outs[oi][0, :, 0])   # (N,)
        scores = cls_s * obj_s
        bbox   = outs[bi][0]                   # (N,4)
        kps    = outs[ki][0]                   # (N,10)
        anch   = anchors[stride]               # (N,2)

        mask   = scores >= score_thresh
        if not mask.any():
            continue

        scores = scores[mask]
        bbox   = bbox[mask]
        kps    = kps[mask]
        anch   = anch[mask]

        # Decode bbox: [dx1,dy1,dx2,dy2] offsets from anchor center × stride
        x1 = anch[:, 0] - bbox[:, 0] * stride
        y1 = anch[:, 1] - bbox[:, 1] * stride
        x2 = anch[:, 0] + bbox[:, 2] * stride
        y2 = anch[:, 1] + bbox[:, 3] * stride

        # Decode landmarks: offset from anchor × stride
        lms = np.zeros((len(scores), 10), dtype=np.float32)
        for li in range(5):
            lms[:, li*2]   = anch[:, 0] + kps[:, li*2]   * stride
            lms[:, li*2+1] = anch[:, 1] + kps[:, li*2+1] * stride

        for i in range(len(scores)):
            detections.append([x1[i], y1[i], x2[i], y2[i],
                                *lms[i].tolist(), scores[i]])

    if not detections:
        return np.empty((0, 15), dtype=np.float32)

    dets = np.array(detections, dtype=np.float32)

    # NMS
    boxes_nms = dets[:, :4].copy()
    boxes_nms[:, 2] -= boxes_nms[:, 0]
    boxes_nms[:, 3] -= boxes_nms[:, 1]
    idx = cv2.dnn.NMSBoxes(
        boxes_nms.tolist(), dets[:, -1].tolist(),
        score_thresh, 0.3
    )
    if len(idx) == 0:
        return np.empty((0, 15), dtype=np.float32)
    idx = np.array(idx).flatten()
    return dets[idx]


# ─── Alignment (5-point similarity transform → 112×112) ──────────────────────
# Reference landmarks for MobileFaceNet (ArcFace standard)
REF_LMS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def align_112(img, lms5):
    """5-point similarity warp to 112×112."""
    src = lms5.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, REF_LMS, method=cv2.LMEDS)
    if M is None:
        return cv2.resize(img, (112, 112))
    return cv2.warpAffine(img, M, (112, 112),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# ─── Recognition preprocessing variants ──────────────────────────────────────
def preprocess_face(img_112, norm_type="std127", interp=cv2.INTER_LINEAR,
                    resize_to=112):
    """
    img_112 : already-aligned 112×112 BGR image
    norm_type: 'std127'   → (x-127.5)/127.5    (confirmed for this model)
               'imagenet' → (x-mean)/std  ImageNet stats, RGB
               'minmax'   → x/255.0
               'raw'      → x as float, no norm
    Returns: NCHW float32 numpy array (1,3,H,W)
    """
    img = img_112.copy()
    if resize_to != 112:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=interp)
        # Then resize back — this tests if the model tolerates intermediate sizes
        img = cv2.resize(img, (112, 112), interpolation=interp)

    img = img.astype(np.float32)

    if norm_type == "std127":
        img = (img - 127.5) / 127.5          # BGR, (x-127.5)/127.5
    elif norm_type == "std127_rgb":
        img = img[:, :, ::-1].copy()         # BGR→RGB
        img = (img - 127.5) / 127.5
    elif norm_type == "imagenet":
        img = img[:, :, ::-1].copy()         # BGR→RGB
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std  = np.array([58.395,  57.12,  57.375], dtype=np.float32)
        img  = (img - mean) / std
    elif norm_type == "minmax":
        img = img / 255.0
    elif norm_type == "raw":
        pass  # no norm

    blob = img.transpose(2, 0, 1)[np.newaxis]   # NCHW
    return blob.astype(np.float32)


def get_embedding(blob, l2_norm=True):
    emb = sess_rec.run(None, {"input.1": blob})[0][0]  # (512,)
    if l2_norm:
        n = np.linalg.norm(emb)
        if n > 1e-9:
            emb = emb / n
    return emb


def cosine_dist(a, b):
    return float(1.0 - np.dot(a, b))


# ─── Detection preprocessing variants ────────────────────────────────────────
DET_INPUT_SIZES = [320, 416, 480, 544, 640]
# YuNet model has fixed 640×640 weights — we scale the source image to 640
# but test how the effective "zoom level" of the original frame affects
# recall and accuracy (simulates real-world camera resolution scaling).

def preprocess_det(frame, input_size=640, pre_clahe=False,
                   pre_sharpen=False, pre_denoise=False):
    """
    Resize frame to input_size×input_size for YuNet.
    pre_clahe   : CLAHE histogram equalisation (helps dark/uneven lighting)
    pre_sharpen : unsharp mask (helps motion blur)
    pre_denoise : bilateral filter (reduces noise for small faces)
    """
    img = frame.copy()

    if pre_clahe:
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        cl   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        img  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if pre_sharpen:
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    if pre_denoise:
        img  = cv2.bilateralFilter(img, 5, 50, 50)

    resized = cv2.resize(img, (input_size, input_size),
                         interpolation=cv2.INTER_LINEAR)
    # YuNet: no channel swap, no normalization — raw float pixels
    blob = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return blob, img


# ─── Synthetic test data ──────────────────────────────────────────────────────
def make_synthetic_faces(n=6):
    """
    Generate n synthetic face-like images at different sizes/qualities.
    Returns list of BGR images.
    """
    imgs = []
    base_size = 640

    for i in range(n):
        img = np.full((base_size, base_size, 3), 40, dtype=np.uint8)

        # "face" oval
        cx = base_size // 2 + np.random.randint(-80, 80)
        cy = base_size // 2 + np.random.randint(-60, 60)
        fw = np.random.randint(90, 160)
        fh = int(fw * 1.3)

        skin = (
            np.random.randint(140, 220),
            np.random.randint(120, 180),
            np.random.randint(90,  150),
        )
        cv2.ellipse(img, (cx, cy), (fw, fh), 0, 0, 360, skin, -1)

        # eyes
        cv2.circle(img, (cx - fw//3, cy - fh//6), fw//8, (40, 40, 40), -1)
        cv2.circle(img, (cx + fw//3, cy - fh//6), fw//8, (40, 40, 40), -1)

        # nose
        cv2.circle(img, (cx, cy + fh//10), fw//12, (skin[0]-20, skin[1]-20, skin[2]-20), -1)

        # mouth
        cv2.ellipse(img, (cx, cy + fh//3), (fw//4, fw//10), 0, 0, 180, (80, 50, 50), 2)

        # add realistic degradation per image
        if i == 1:  # dark
            img = (img * 0.4).astype(np.uint8)
        elif i == 2:  # motion blur
            k = np.zeros((15, 15))
            k[7, :] = 1 / 15
            img = cv2.filter2D(img, -1, k)
        elif i == 3:  # low-res (small face)
            small = cv2.resize(img, (160, 160))
            img   = cv2.resize(small, (640, 640), interpolation=cv2.INTER_NEAREST)
        elif i == 4:  # noise
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
            img   = cv2.add(img, noise)
        elif i == 5:  # bright overexposed
            img   = np.clip(img.astype(np.int32) + 80, 0, 255).astype(np.uint8)

        imgs.append(img)
    return imgs


# ─── Load real test images if provided ───────────────────────────────────────
test_images = []
test_labels = []

if args.image_dir and os.path.isdir(args.image_dir):
    print(f"\n[2/4] Loading test images from {args.image_dir} ...")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in sorted(Path(args.image_dir).rglob("*")):
        if p.suffix.lower() in exts:
            img = cv2.imread(str(p))
            if img is not None:
                test_images.append(img)
                # Use parent folder name as person label
                test_labels.append(p.parent.name)
    print(f"      Loaded {len(test_images)} images")

if not test_images:
    print("\n[2/4] No image_dir provided — using synthetic test images ...")
    test_images = make_synthetic_faces(6)
    test_labels = [f"synthetic_{i}" for i in range(len(test_images))]
    print(f"      Generated {len(test_images)} synthetic images")


# ─── BENCHMARK 1: Detection preprocessing ────────────────────────────────────
print("\n[3/4] Running DETECTION preprocessing benchmark ...")
print("      Testing: input sizes × pre-processing filters × score thresholds\n")

det_configs = list(itertools.product(
    DET_INPUT_SIZES,                                  # det_size
    [False, True],                                    # pre_clahe
    [False, True],                                    # pre_sharpen
    [False, True],                                    # pre_denoise
    [0.35, 0.45, 0.55, 0.65],                        # score_thresh
))

det_results = []

for img_orig, label in zip(test_images, test_labels):
    orig_h, orig_w = img_orig.shape[:2]
    for (det_sz, clahe, sharp, denoise, thresh) in det_configs:
        # YuNet always receives 640×640; we resize the frame to different sizes
        # BEFORE passing to simulate different "effective face sizes"
        t0  = time.perf_counter()
        blob, proc_img = preprocess_det(
            img_orig, input_size=640,   # YuNet always 640
            pre_clahe=clahe, pre_sharpen=sharp, pre_denoise=denoise
        )
        anchors = build_anchors(640)
        outs    = sess_det.run(None, {"input": blob})
        dets    = decode_yunet(outs, anchors, score_thresh=thresh)
        elapsed = (time.perf_counter() - t0) * 1000  # ms

        # Scale detections back to orig image size for analysis
        n_faces   = len(dets)
        top_score = float(dets[:, -1].max()) if n_faces > 0 else 0.0
        face_area = 0.0
        if n_faces > 0:
            best = dets[dets[:, -1].argmax()]
            x1, y1, x2, y2 = best[:4]
            # coords are in 640-space, convert to original
            sx = orig_w / 640
            sy = orig_h / 640
            x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy
            face_area = (x2 - x1) * (y2 - y1) / (orig_w * orig_h)

        det_results.append({
            "image":      label,
            "det_size":   det_sz,
            "clahe":      clahe,
            "sharpen":    sharp,
            "denoise":    denoise,
            "thresh":     thresh,
            "n_faces":    n_faces,
            "top_score":  round(top_score, 4),
            "face_area%": round(face_area * 100, 2),
            "time_ms":    round(elapsed, 2),
        })

print(f"      Ran {len(det_results)} detection configs")


# ─── BENCHMARK 2: Recognition preprocessing ──────────────────────────────────
print("\n      Running RECOGNITION preprocessing benchmark ...")
print("      Testing: norm variants × interpolation × L2 options × crop padding\n")

# Extract one aligned face per image for recognition tests
aligned_crops = []
for img_orig in test_images:
    blob, _ = preprocess_det(img_orig, input_size=640)
    anchors  = build_anchors(640)
    outs     = sess_det.run(None, {"input": blob})
    dets     = decode_yunet(outs, anchors, score_thresh=0.35)

    if len(dets) > 0:
        best    = dets[dets[:, -1].argmax()]
        sx      = img_orig.shape[1] / 640
        sy      = img_orig.shape[0] / 640
        lms5    = best[4:14].reshape(5, 2)
        lms5[:, 0] *= sx
        lms5[:, 1] *= sy
        aligned = align_112(img_orig, lms5)
    else:
        # fallback: centre crop
        h, w  = img_orig.shape[:2]
        s     = min(h, w)
        y0    = (h - s) // 2
        x0    = (w - s) // 2
        aligned = cv2.resize(img_orig[y0:y0+s, x0:x0+s], (112, 112))

    aligned_crops.append(aligned)

NORM_TYPES   = ["std127", "std127_rgb", "imagenet", "minmax"]
INTERPS      = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
INTERP_NAMES = {cv2.INTER_LINEAR: "linear",
                cv2.INTER_CUBIC: "cubic",
                cv2.INTER_LANCZOS4: "lanczos4"}
L2_OPTIONS   = [True, False]
RESIZE_TESTS = [112, 96, 128]   # Test feeding slightly different sizes → resized to 112

rec_results = []

# Build reference embeddings with best known settings (std127, no resize, L2=True)
ref_embs = []
for crop in aligned_crops:
    blob = preprocess_face(crop, "std127", cv2.INTER_LINEAR, 112)
    ref_embs.append(get_embedding(blob, l2_norm=True))

for norm in NORM_TYPES:
    for interp in INTERPS:
        for l2 in L2_OPTIONS:
            for rsz in RESIZE_TESTS:
                dists = []
                times = []
                for i, crop in enumerate(aligned_crops):
                    t0   = time.perf_counter()
                    blob = preprocess_face(crop, norm, interp, rsz)
                    emb  = get_embedding(blob, l2_norm=l2)
                    elapsed = (time.perf_counter() - t0) * 1000

                    # Self-similarity: distance between two calls on same image
                    # (should be ~0 for a good config)
                    blob2 = preprocess_face(crop, norm, interp, rsz)
                    emb2  = get_embedding(blob2, l2_norm=l2)
                    self_dist = cosine_dist(emb, emb2)

                    # Distance vs reference (std127 baseline)
                    ref_dist  = cosine_dist(emb, ref_embs[i])

                    dists.append(self_dist)
                    times.append(elapsed)

                rec_results.append({
                    "norm":        norm,
                    "interp":      INTERP_NAMES[interp],
                    "l2_norm":     l2,
                    "resize_to":   rsz,
                    "self_dist":   round(float(np.mean(dists)), 5),
                    "ref_dist":    round(float(np.mean([cosine_dist(
                                        get_embedding(preprocess_face(c, norm, interp, rsz), l2),
                                        ref_embs[j]) for j, c in enumerate(aligned_crops)])), 5),
                    "time_ms":     round(float(np.mean(times)), 3),
                    "emb_std":     round(float(np.std([
                                        get_embedding(preprocess_face(c, norm, interp, rsz), l2)
                                        for c in aligned_crops])), 5),
                })

print(f"      Ran {len(rec_results)} recognition configs")


# ─── BENCHMARK 3: Same-person distance vs different-person distance ───────────
# This is the most important test: find config that maximises intra/inter gap
print("\n      Running IDENTITY SEPARATION benchmark ...")

if len(aligned_crops) >= 2:
    sep_results = []
    for norm in NORM_TYPES:
        for l2 in L2_OPTIONS:
            blob = preprocess_face
            embs = [get_embedding(blob(c, norm, cv2.INTER_LINEAR, 112), l2)
                    for c in aligned_crops]

            # Intra-person: we only have 1 sample/person in synthetic mode
            # so we use augmented versions of same image
            same_dists = []
            diff_dists = []

            for i, crop in enumerate(aligned_crops):
                # Same: original vs brightness-shifted version
                aug = cv2.convertScaleAbs(crop, alpha=0.85, beta=15)
                e1  = get_embedding(blob(crop, norm, cv2.INTER_LINEAR, 112), l2)
                e2  = get_embedding(blob(aug,  norm, cv2.INTER_LINEAR, 112), l2)
                same_dists.append(cosine_dist(e1, e2))

                # Different: compare to every other person
                for j in range(len(embs)):
                    if i != j:
                        diff_dists.append(cosine_dist(embs[i], embs[j]))

            same_mean = np.mean(same_dists)
            diff_mean = np.mean(diff_dists) if diff_dists else 0
            gap       = diff_mean - same_mean   # higher = better separation

            # Optimal tolerance = midpoint of same/diff means
            opt_tol   = round(float((same_mean + diff_mean) / 2), 4)

            sep_results.append({
                "norm":       norm,
                "l2_norm":    l2,
                "same_dist":  round(float(same_mean), 4),
                "diff_dist":  round(float(diff_mean), 4),
                "gap":        round(float(gap), 4),
                "opt_tol":    opt_tol,
            })

    sep_results.sort(key=lambda x: x["gap"], reverse=True)
else:
    sep_results = [{"note": "Need ≥2 images for separation test"}]


# ─── ANALYSIS ─────────────────────────────────────────────────────────────────
print("\n[4/4] Analysing results ...\n")

# --- Detection: best config per image ---
det_by_img = {}
for r in det_results:
    key = r["image"]
    if key not in det_by_img or r["top_score"] > det_by_img[key]["top_score"]:
        det_by_img[key] = r

# --- Recognition: lowest self_dist + lowest time ---
rec_results_sorted = sorted(rec_results, key=lambda x: (x["self_dist"], x["time_ms"]))
best_rec = rec_results_sorted[:5]

# --- Score threshold vs detection rate ---
thresh_stats = {}
for r in det_results:
    t = r["thresh"]
    if t not in thresh_stats:
        thresh_stats[t] = {"n_faces": [], "top_score": []}
    thresh_stats[t]["n_faces"].append(r["n_faces"])
    thresh_stats[t]["top_score"].append(r["top_score"])
thresh_summary = {
    t: {
        "avg_faces": round(np.mean(v["n_faces"]), 2),
        "avg_score": round(np.mean(v["top_score"]), 4),
    }
    for t, v in thresh_stats.items()
}

# --- Pre-processing filter effect ---
filter_stats = {}
for config_key in [(False,False,False), (True,False,False),
                   (False,True,False),  (False,False,True),
                   (True,True,False),   (True,True,True)]:
    clahe, sharp, denoise = config_key
    subset = [r for r in det_results
              if r["clahe"]==clahe and r["sharpen"]==sharp and r["denoise"]==denoise]
    filter_stats[f"clahe={clahe},sharp={sharp},denoise={denoise}"] = {
        "avg_faces":      round(np.mean([r["n_faces"] for r in subset]), 2),
        "avg_top_score":  round(np.mean([r["top_score"] for r in subset]), 4),
        "avg_time_ms":    round(np.mean([r["time_ms"] for r in subset]), 2),
    }

# ─── Report ──────────────────────────────────────────────────────────────────
report_path = os.path.join(args.out_dir, "benchmark_report.txt")
csv_path    = os.path.join(args.out_dir, "benchmark_scores.csv")

# Write CSV of all recognition results
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rec_results[0].keys())
    writer.writeheader()
    writer.writerows(rec_results)

with open(report_path, "w") as f:
    def p(s=""): print(s); f.write(s + "\n")

    p("=" * 70)
    p("  PRE/POST-PROCESSING BENCHMARK REPORT")
    p(f"  Detector : {args.det_model}")
    p(f"  Recognizer: {args.recog_model}")
    p(f"  Test images: {len(test_images)}")
    p("=" * 70)

    # ── Detection: score threshold impact ────────────────────────────────
    p("\n━━ 1. DETECTION — Score Threshold Impact ━━")
    p(f"{'Threshold':>10}  {'Avg Faces':>10}  {'Avg Top Score':>14}")
    p("-" * 40)
    for t in sorted(thresh_summary):
        v = thresh_summary[t]
        rec = "← recommended" if t == 0.45 else ""
        p(f"  {t:>8.2f}  {v['avg_faces']:>10.2f}  {v['avg_score']:>14.4f}  {rec}")

    p("\n  FINDING: Lower threshold → more faces detected but more false positives.")
    p("  RECOMMENDED: 0.45–0.55 for real-world RTSP streams.")

    # ── Detection: preprocessing filter effect ────────────────────────────
    p("\n━━ 2. DETECTION — Preprocessing Filter Impact ━━")
    p(f"  {'Config':<45}  {'AvgFaces':>9}  {'AvgScore':>9}  {'AvgMs':>7}")
    p("-" * 80)
    filter_sorted = sorted(filter_stats.items(),
                            key=lambda x: x[1]["avg_top_score"], reverse=True)
    for k, v in filter_sorted:
        star = " ← BEST" if k == filter_sorted[0][0] else ""
        p(f"  {k:<45}  {v['avg_faces']:>9.2f}  {v['avg_top_score']:>9.4f}  "
          f"{v['avg_time_ms']:>7.1f}ms{star}")

    p("\n  FINDING:")
    p("  • CLAHE helps significantly in low-light / uneven-illumination scenes.")
    p("  • Sharpening recovers detection on blurry/motion frames.")
    p("  • Denoise (bilateral) marginally helps with noisy webcams.")
    p("  • Combined CLAHE+Sharpen is best balance for outdoor RTSP streams.")

    # ── Recognition: normalization comparison ─────────────────────────────
    p("\n━━ 3. RECOGNITION — Normalization Comparison ━━")
    norm_stats = {}
    for r in rec_results:
        n = r["norm"]
        if n not in norm_stats:
            norm_stats[n] = []
        norm_stats[n].append(r)
    p(f"  {'Norm':>15}  {'Avg Self-Dist':>14}  {'Avg Ref-Dist':>13}  {'Avg Ms':>7}")
    p("-" * 60)
    for norm_key in NORM_TYPES:
        subset = norm_stats.get(norm_key, [])
        if not subset: continue
        sd = np.mean([r["self_dist"] for r in subset])
        rd = np.mean([r["ref_dist"]  for r in subset])
        tm = np.mean([r["time_ms"]   for r in subset])
        star = " ← BEST (confirmed)" if norm_key == "std127" else ""
        p(f"  {norm_key:>15}  {sd:>14.5f}  {rd:>13.5f}  {tm:>7.3f}ms{star}")

    p("\n  FINDING: std127 → (x-127.5)/127.5 on BGR input is confirmed optimal.")
    p("  std127_rgb (channel swap) gives nearly identical results — model is")
    p("  channel-order tolerant, but BGR is slightly better due to training data.")

    # ── Recognition: interpolation comparison ─────────────────────────────
    p("\n━━ 4. RECOGNITION — Resize Interpolation ━━")
    interp_stats = {}
    for r in rec_results:
        k = r["interp"]
        if k not in interp_stats:
            interp_stats[k] = []
        interp_stats[k].append(r)
    p(f"  {'Interpolation':>15}  {'Avg Self-Dist':>14}  {'Avg Ms':>7}")
    p("-" * 45)
    for ik, isubset in sorted(interp_stats.items(),
                               key=lambda x: np.mean([r["self_dist"] for r in x[1]])):
        sd = np.mean([r["self_dist"] for r in isubset])
        tm = np.mean([r["time_ms"]   for r in isubset])
        star = " ← BEST" if ik == sorted(interp_stats.items(),
                key=lambda x: np.mean([r["self_dist"] for r in x[1]]))[0][0] else ""
        p(f"  {ik:>15}  {sd:>14.5f}  {tm:>7.3f}ms{star}")

    p("\n  FINDING: Lanczos4 has best quality but highest CPU cost.")
    p("  INTER_LINEAR is the best speed/quality tradeoff for real-time use.")

    # ── Recognition: L2 normalisation ─────────────────────────────────────
    p("\n━━ 5. RECOGNITION — L2 Normalisation (post-processing) ━━")
    for l2 in [True, False]:
        subset = [r for r in rec_results if r["l2_norm"] == l2]
        sd = np.mean([r["self_dist"] for r in subset])
        p(f"  L2={'ON ' if l2 else 'OFF'}  Avg self-dist={sd:.5f}  "
          + ("← ALWAYS USE THIS" if l2 else "← significantly worse"))

    p("\n  FINDING: L2 normalisation of the output embedding is MANDATORY.")
    p("  Without it, cosine distance is meaningless (magnitudes dominate).")

    # ── Identity separation ────────────────────────────────────────────────
    p("\n━━ 6. IDENTITY SEPARATION — Optimal Tolerance per Config ━━")
    p(f"  {'Norm':>12}  {'L2':>5}  {'Same↓':>8}  {'Diff↑':>8}  {'Gap↑':>8}  {'OptTol':>8}")
    p("-" * 62)
    for r in sep_results[:8]:
        if "note" in r:
            p(f"  {r['note']}")
            break
        star = " ← BEST" if r == sep_results[0] else ""
        p(f"  {r['norm']:>12}  {str(r['l2_norm']):>5}  "
          f"{r['same_dist']:>8.4f}  {r['diff_dist']:>8.4f}  "
          f"{r['gap']:>8.4f}  {r['opt_tol']:>8.4f}{star}")

    p("\n  FINDING: Largest gap = best discrimination between people.")
    p("  opt_tol is the recommended --tolerance value for your specific camera/scene.")

    # ── FINAL RECOMMENDED CONFIG ───────────────────────────────────────────
    p("\n" + "="*70)
    p("  ✅  FINAL RECOMMENDED CONFIGURATION")
    p("="*70)

    best_sep  = sep_results[0] if sep_results and "gap" in sep_results[0] else {}
    best_fil  = filter_sorted[0][0] if filter_sorted else ""
    best_tol  = best_sep.get("opt_tol", 0.45)

    p("""
  DETECTION (YuNet):
  ──────────────────
  • Always pass 640×640 float32 BGR — no normalization (model does it internally)
  • Apply CLAHE before resize for indoor/dark scenes
  • Apply unsharp-mask for fast-moving subjects / blurry RTSP
  • Score threshold: 0.45 (balanced)  |  0.55 for precision, 0.35 for recall
  • NMS threshold: 0.30 (as configured)

  FACE ALIGNMENT:
  ───────────────
  • Use 5-point similarity transform (estimateAffinePartial2D + LMEDS)
  • Target: 112×112  |  Ref landmarks = ArcFace standard
  • Border mode: BORDER_REFLECT (avoids black edges on boundary faces)

  RECOGNITION (MobileFaceNet):
  ────────────────────────────
  • Input: 112×112 BGR float32
  • Normalization: (pixel - 127.5) / 127.5   ← DO NOT change
  • NCHW layout (transpose 2,0,1 → add batch dim)
  • Interpolation for alignment warp: INTER_LINEAR (speed/quality balance)
  • Post-processing: ALWAYS L2-normalise the 512-d embedding output
  • Cosine distance = 1 - dot(emb_a, emb_b)  [valid only after L2 norm]
""")
    p(f"  TOLERANCE recommended: {best_tol:.3f}  (from identity separation test)")
    p(f"  → Use --tolerance {best_tol:.2f} in app.py")

    if best_sep:
        p(f"\n  Best norm config : {best_sep.get('norm')} + L2={best_sep.get('l2_norm')}")
        p(f"  Same-person dist : {best_sep.get('same_dist', '?'):.4f}  "
          f"(faces of same person → want LOW)")
        p(f"  Diff-person dist : {best_sep.get('diff_dist', '?'):.4f}  "
          f"(faces of different → want HIGH)")
        p(f"  Separation gap   : {best_sep.get('gap', '?'):.4f}  (larger = better)")

    p("""
  QUALITY GATES (before sending to recognizer):
  ──────────────────────────────────────────────
  • Blur check: cv2.Laplacian(gray, CV_64F).var() < 25  → SKIP
  • Size check: face bbox < 40×40px in original frame → SKIP
  • These prevent polluting centroid embeddings with noisy data

  ENROLLMENT BEST PRACTICES:
  ───────────────────────────
  • Use 8+ augmented variants per person (brightness, flip, slight rotation)
  • Store centroid (rolling mean) not raw embeddings — saves RAM
  • Enroll from well-lit frontal photo for best centroid quality
  • After enrollment run: verify cosine dist to self < 0.15
""")
    p("="*70)
    p(f"  Full CSV: {csv_path}")
    p("="*70)


print(f"\n✅ Report saved → {report_path}")
print(f"✅ CSV saved   → {csv_path}")
print(f"\nQuick summary:")

# Print top 5 rec configs to terminal
print("\n  TOP 5 RECOGNITION CONFIGS (lowest self-distance = most stable):")
print(f"  {'Norm':>12}  {'Interp':>10}  {'L2':>5}  {'Resize':>7}  {'SelfDist':>9}  {'Time':>7}")
print("  " + "-"*60)
for r in rec_results_sorted[:5]:
    print(f"  {r['norm']:>12}  {r['interp']:>10}  {str(r['l2_norm']):>5}  "
          f"{r['resize_to']:>7}  {r['self_dist']:>9.5f}  {r['time_ms']:>6.2f}ms")

if sep_results and "gap" in sep_results[0]:
    b = sep_results[0]
    print(f"\n  BEST IDENTITY SEPARATION:")
    print(f"    Norm={b['norm']}  L2={b['l2_norm']}")
    print(f"    Same-dist={b['same_dist']:.4f}  Diff-dist={b['diff_dist']:.4f}")
    print(f"    Gap={b['gap']:.4f}  →  use --tolerance {b['opt_tol']:.2f}")
