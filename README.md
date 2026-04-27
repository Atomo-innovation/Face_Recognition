# NPU Real-Time RTSP Face Recognition

Detection + Recognition fully on NPU using OpenCV DNN with **DNN_BACKEND_TIMVX / DNN_TARGET_NPU**  
Same backend as your reference code — both models run on the same NPU.

---

## Project Structure

```
npu_face_recognition/
├── app.py           # Main Flask app — 3-thread real-time pipeline
├── priorbox.py      # Anchor decoder (same as reference)
├── aligner.py       # 5-landmark face alignment → 112×112
├── recognizer.py    # NPU recognition model wrapper
├── utils.py         # Draw helpers
├── templates/
│   └── index.html   # Web dashboard
├── models/          # Put your ONNX models here
├── known_faces/     # Auto-created
├── unknown_faces/   # Auto-created
└── face_db.json     # Auto-created
```

---

## Step 1 — Install dependencies

```bash
pip install flask opencv-python numpy pillow
```

> OpenCV must be built with **TIM-VX support** for NPU to work.  
> On Rockchip / Khadas / NXP boards this is usually pre-installed.  
> Check: `python -c "import cv2; print(cv2.dnn.DNN_BACKEND_TIMVX)"`

---

## Step 2 — Download models

### Detection model (libfacedetection)
```bash
# YuNet — lightweight, fast, works great with NPU
wget -O models/yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

### Recognition model
```bash
# MobileFaceNet — fast, 128-dim embedding (~1 MB)
wget -O models/mobilefacenet.onnx \
  https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/releases/download/v1.0/MobileFaceNet.onnx

# OR ArcFace R50 — more accurate, 512-dim (~90 MB)
# Download from: https://drive.google.com/file/d/1KG1H6e-wFVvFV8lfAJBgOWS1MBc8T_j3
```

---

## Step 3 — Run

### NPU mode (default)
```bash
python app.py \
  --det_model   models/yunet.onnx \
  --recog_model models/mobilefacenet.onnx \
  --recog_type  mobilefacenet \
  --use_npu     true
```

### CPU fallback (if NPU not available)
```bash
python app.py \
  --det_model   models/yunet.onnx \
  --recog_model models/mobilefacenet.onnx \
  --recog_type  mobilefacenet \
  --use_npu     false
```

### With your own libfacedetection model
```bash
python app.py \
  --det_model   models/facedetection_yunet_2022mar.onnx \
  --recog_model models/mobilefacenet.onnx \
  --recog_type  mobilefacenet \
  --use_npu     true \
  --conf_thresh 0.6 \
  --nms_thresh  0.3
```

Open: **http://localhost:5050**

---

## All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--det_model` | required | Path to detection ONNX |
| `--recog_model` | required | Path to recognition ONNX |
| `--recog_type` | `mobilefacenet` | `mobilefacenet` or `arcface_r50` |
| `--use_npu` | `true` | Use TIM-VX NPU or CPU fallback |
| `--conf_thresh` | `0.6` | Detection confidence threshold |
| `--nms_thresh` | `0.3` | NMS threshold |
| `--keep_top_k` | `200` | Max detections per frame |
| `--tolerance` | `0.5` | Recognition distance (lower = stricter) |
| `--recog_every` | `2` | Run recognition every N frames |
| `--port` | `5050` | Web server port |

---

## How the pipeline works

```
RTSP Camera
    │
    ▼
RTSPGrabber (Thread 1)
  cap.grab() loop — drains buffer at full FPS
  cap.retrieve() — decodes only the freshest frame
  → _raw_frame (always latest)
    │
    ▼
NPUProcessor (Thread 2)
  Detection:   blobFromImage → det_net.forward() on NPU → PriorBox.decode() → NMS
  Alignment:   5-landmark affine warp → 112×112 aligned crop
  Recognition: blobFromImage → recog_net.forward() on NPU → L2 embedding
  Matching:    cosine distance vs known_encodings
  → _display_frame (annotated)
    │
    ▼
Flask MJPEG (Thread 3)
  gen_frames() → multipart/x-mixed-replace → browser
```

---

## RTSP URL examples

| Brand | URL |
|-------|-----|
| Hikvision | `rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101` |
| Dahua | `rtsp://admin:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0` |
| Reolink | `rtsp://admin:pass@192.168.1.200:554//h264Preview_01_main` |
| Generic | `rtsp://user:pass@192.168.1.100:554/stream1` |

---

## Verify NPU is being used

```python
import cv2
print("TIM-VX backend ID:", cv2.dnn.DNN_BACKEND_TIMVX)   # should print a number
print("NPU target ID    :", cv2.dnn.DNN_TARGET_NPU)        # should print a number
```

If these fail, your OpenCV was not built with TIM-VX support.  
Use `--use_npu false` to fall back to CPU.
