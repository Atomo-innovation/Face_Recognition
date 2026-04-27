"""
NPU-accelerated Face Recognizer
================================
Uses OpenCV DNN with DNN_BACKEND_TIMVX + DNN_TARGET_NPU for recognition,
mirroring exactly how the reference detection code runs on the NPU.

Supported recognition models (ONNX):
  • MobileFaceNet  — fast, 128-dim embedding  (~1 MB)
  • ArcFace R50    — accurate, 512-dim         (~90 MB)

Download links printed on first run if model not found.
"""

import cv2
import numpy as np
import os


# ── Model configs ──────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "mobilefacenet": {
        "input_size":  (112, 112),
        "input_name":  "input",
        "output_name": "output",
        "mean":        (127.5, 127.5, 127.5),
        "scale":       1.0 / 127.5,
        "dim":         128,
        "url": "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/"
               "releases/download/v1.0/MobileFaceNet.onnx",
    },
    "arcface_r50": {
        "input_size":  (112, 112),
        "input_name":  "input.1",
        "output_name": "683",
        "mean":        (127.5, 127.5, 127.5),
        "scale":       1.0 / 127.5,
        "dim":         512,
        "url": "https://drive.google.com/file/d/1KG1H6e-wFVvFV8lfAJBgOWS1MBc8T_j3",
    },
}


class NPUFaceRecognizer:
    """
    Wraps an ONNX recognition model running on NPU via OpenCV DNN.

    Parameters
    ----------
    model_path  : path to .onnx recognition model
    model_type  : 'mobilefacenet' | 'arcface_r50'
    use_npu     : True = TIM-VX NPU, False = CPU fallback
    """

    def __init__(self, model_path: str,
                 model_type: str = "mobilefacenet",
                 use_npu: bool = True):

        if not os.path.exists(model_path):
            cfg = MODEL_CONFIGS.get(model_type, {})
            url = cfg.get("url", "N/A")
            raise FileNotFoundError(
                f"\nRecognition model not found: {model_path}\n"
                f"Download '{model_type}' from:\n  {url}\n"
                f"Then pass --recog_model <path>"
            )

        self.cfg = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["mobilefacenet"])
        self.net = cv2.dnn.readNetFromONNX(model_path)

        if use_npu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_TIMVX)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)
            print(f"[Recognizer] ✓ NPU backend (TIM-VX) — {model_type}")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[Recognizer] CPU backend — {model_type}")

        self.use_npu    = use_npu
        self.model_type = model_type
        self.dim        = self.cfg["dim"]

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        aligned_face : (112, 112, 3) BGR — output of aligner.align_face()

        Returns
        -------
        embedding : (dim,) float32, L2-normalized
        """
        h, w = self.cfg["input_size"]
        inp  = cv2.resize(aligned_face, (w, h))

        blob = cv2.dnn.blobFromImage(
            inp,
            scalefactor=self.cfg["scale"],
            size=(w, h),
            mean=self.cfg["mean"],
            swapRB=False,   # already BGR
            crop=False,
        )

        self.net.setInput(blob)
        emb = self.net.forward().flatten().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb /= norm

        return emb
