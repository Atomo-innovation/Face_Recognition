"""
PriorBox — anchor generation for libfacedetection ONNX model.
Identical to the reference implementation.
"""
import numpy as np
from itertools import product


class PriorBox:
    def __init__(self, input_shape, output_shape=None, strides=(8, 16, 32, 64),
                 min_sizes=((10, 16, 24), (32, 48), (64, 96), (128, 192, 256))):
        self.input_shape  = input_shape   # (w, h)
        self.output_shape = output_shape if output_shape else input_shape
        self.strides      = strides
        self.min_sizes    = min_sizes
        self.priors       = self._generate()

    def _generate(self):
        W, H = self.input_shape
        anchors = []
        for k, stride in enumerate(self.strides):
            fh = int(np.ceil(H / stride))
            fw = int(np.ceil(W / stride))
            for i, j in product(range(fh), range(fw)):
                for min_size in self.min_sizes[k]:
                    s_kx = min_size / W
                    s_ky = min_size / H
                    cx   = (j + 0.5) * stride / W
                    cy   = (i + 0.5) * stride / H
                    anchors.append([cx, cy, s_kx, s_ky])
        return np.array(anchors, dtype=np.float32)

    def decode(self, loc, conf, iou, conf_thresh=0.6):
        """
        Decode raw network output into [x1,y1,x2,y2, lm*10, score] detections.
        loc  : (1, #anchors, 14)
        conf : (1, #anchors, 2)
        iou  : (1, #anchors, 1)
        """
        loc  = loc.squeeze(0)   # (#anchors, 14)
        conf = conf.squeeze(0)  # (#anchors, 2)
        iou  = iou.squeeze(0)   # (#anchors, 1)

        # face class score  (geometric mean of softmax + iou)
        cls_scores  = conf[:, 1]
        iou_scores  = iou[:, 0].clip(0, 1)
        scores      = np.sqrt(cls_scores * iou_scores)

        # keep only confident
        mask        = scores > conf_thresh
        scores      = scores[mask]
        loc         = loc[mask]
        priors      = self.priors[mask]

        W, H = self.input_shape
        OW, OH = self.output_shape

        # Decode boxes
        bboxes      = np.empty((len(scores), 4), dtype=np.float32)
        bboxes[:, 0] = (priors[:, 0] + loc[:, 0] * 0.1 * priors[:, 2]) * OW
        bboxes[:, 1] = (priors[:, 1] + loc[:, 1] * 0.1 * priors[:, 3]) * OH
        bboxes[:, 2] = (priors[:, 2] * np.exp(loc[:, 2] * 0.2)) * OW
        bboxes[:, 3] = (priors[:, 3] * np.exp(loc[:, 3] * 0.2)) * OH
        # cx,cy,w,h → x1,y1,x2,y2
        bboxes[:, 0] -= bboxes[:, 2] / 2
        bboxes[:, 1] -= bboxes[:, 3] / 2
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

        # Decode 5 landmarks (x,y) × 5
        landmarks = np.empty((len(scores), 10), dtype=np.float32)
        for n in range(5):
            landmarks[:, n*2]   = (priors[:, 0] + loc[:, 4 + n*2]   * 0.1 * priors[:, 2]) * OW
            landmarks[:, n*2+1] = (priors[:, 1] + loc[:, 4 + n*2+1] * 0.1 * priors[:, 3]) * OH

        return np.concatenate([bboxes, landmarks, scores[:, np.newaxis]], axis=1)
