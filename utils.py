"""
Draw utilities — bounding boxes, landmarks, scores.
"""
import cv2
import numpy as np


def draw(img, bboxes, landmarks, scores):
    """Draw boxes + 5-point landmarks on img (in-place copy)."""
    out = img.copy()
    for bbox, lms, score in zip(bboxes, landmarks, scores):
        x1, y1, x2, y2 = bbox.astype(int)
        # box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # score label
        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # 5 landmarks: RE, LE, Nose, RMouth, LMouth
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
        for idx, (lx, ly) in enumerate(lms):
            cv2.circle(out, (int(lx), int(ly)), 3, colors[idx], -1)
    return out
