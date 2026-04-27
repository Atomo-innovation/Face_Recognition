"""
Face alignment using 5 landmarks detected by libfacedetection.
Produces a 112×112 aligned face crop — standard input for ArcFace/FaceNet.
"""
import cv2
import numpy as np


# Reference landmark positions for a 112×112 aligned face (ArcFace standard)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # right eye
    [73.5318, 51.5014],   # left eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # right mouth corner
    [70.7299, 92.2041],   # left mouth corner
], dtype=np.float32)


def align_face(img: np.ndarray, landmarks: np.ndarray,
               output_size: int = 112) -> np.ndarray:
    """
    Warp face crop to canonical 112×112 using similarity transform.

    Parameters
    ----------
    img       : BGR image (full frame)
    landmarks : (5, 2) array — [RE, LE, Nose, RM, LM] in pixel coords
    output_size : side length of output square (default 112)

    Returns
    -------
    aligned   : (output_size, output_size, 3) BGR image
    """
    scale  = output_size / 112.0
    dst    = REFERENCE_LANDMARKS * scale

    src    = landmarks.astype(np.float32)
    M, _   = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    if M is None:
        # Fallback: crude crop + resize around centroid
        cx = int(landmarks[:, 0].mean())
        cy = int(landmarks[:, 1].mean())
        h, w = img.shape[:2]
        half = output_size // 2
        x1 = max(0, cx - half); y1 = max(0, cy - half)
        x2 = min(w, cx + half); y2 = min(h, cy + half)
        crop = img[y1:y2, x1:x2]
        return cv2.resize(crop, (output_size, output_size))

    aligned = cv2.warpAffine(img, M, (output_size, output_size),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return aligned
