from collections import defaultdict
import numpy as np
import cv2
import torch

from utils.utils import _nms, _top1
from utils.image import get_affine_transform, affine_transform
import matplotlib.pyplot as plt


class BlurBallPostprocessor(object):
    def __init__(self, cfg):
        # print(cfg['detector']['postprocessor'])
        self._score_threshold = cfg["detector"]["postprocessor"]["score_threshold"]
        self._model_name = cfg["model"]["name"]
        self._scales = cfg["detector"]["postprocessor"]["scales"]
        self._blob_det_method = cfg["detector"]["postprocessor"]["blob_det_method"]
        self._use_hm_weight = cfg["detector"]["postprocessor"]["use_hm_weight"]
        # self._xy_comp_method  = cfg['detector']['postprocessor']['xy_comp_method']
        # print(self._score_threshold, self._scales)

        # self._hm_type = cfg['target_generator']['type']
        # self._sigmas  = cfg['target_generator']['sigmas']
        self._sigmas = cfg["dataloader"]["heatmap"]["sigmas"]
        # self._mags    = cfg['target_generator']['mags']
        # self._min_values = cfg['target_generator']['min_values']
        # print(hm_type, sigmas, mags, min_values)

    def _detect_blob_concomp(self, hm):
        xys, ls, angles, scores = [], [], [], []
        if np.max(hm) > self._score_threshold:
            visi = True
            th, hm_th = cv2.threshold(hm, self._score_threshold, 1, cv2.THRESH_BINARY)
            n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
            for m in range(1, n_labels):
                ys, xs = np.where(labels == m)
                ws = hm[ys, xs]
                if self._use_hm_weight:
                    score = ws.sum()
                    x = np.sum(np.array(xs) * ws) / np.sum(ws)
                    y = np.sum(np.array(ys) * ws) / np.sum(ws)
                    mask = (labels == m).astype(np.uint8)
                    moments = cv2.moments(mask, binaryImage=True)
                    # Calculate the orientation angle
                    if moments["mu20"] - moments["mu02"] == 0:
                        angle = 0
                    else:
                        angle = 0.5 * np.arctan2(
                            2 * moments["mu11"], moments["mu20"] - moments["mu02"]
                        )

                    # Convert angle from radians to degrees
                    angle_degrees = np.degrees(angle)

                    # Find the coordinates of the non-zero pixels
                    coords = np.column_stack(np.where(mask > 0))

                    # print(coords)
                    # Apply PCA
                    mean, eigenvectors = cv2.PCACompute(
                        coords.astype(np.float32), mean=None
                    )

                    # The principal axis is the eigenvector corresponding to the largest eigenvalue
                    principal_axis = eigenvectors[0]

                    # Project the coordinates onto the principal axis to find the extent
                    projected = np.dot(coords - mean, principal_axis)

                    # Calculate the length of the major axis
                    l = projected.max() - projected.min()
                else:
                    score = ws.shape[0]
                    x = np.sum(np.array(xs)) / ws.shape[0]
                    y = np.sum(np.array(ys)) / ws.shape[0]
                    # print(xs, ys)
                    # print(score, x, y)
                xys.append(np.array([x, y]))
                ls.append(l / 2)
                angles.append(angle_degrees)
                scores.append(score)
        return xys, angles, ls, scores

    def run(self, preds, affine_mats):
        results = defaultdict(lambda: defaultdict(dict))
        for scale in self._scales:
            preds_ = preds[scale]
            affine_mats_ = affine_mats[scale].cpu().numpy()
            hms_ = preds_.sigmoid_().cpu().numpy()

            b, s, h, w = hms_.shape
            for i in range(b):
                for j in range(s):
                    # print(i,j)
                    """
                    if self._xy_comp_method=='center':
                        assert 0, 'not yet'
                        #xy_, visi_, blob_size_ = self._detect_blob_center(hms_[i,j])
                        xy_, visi_, blob_score_ = self._detect_blob_center(hms_[i,j])
                    elif self._xy_comp_method=='gravity':
                        #xy_, visi_, blob_size_ = self._detect_blob_gravity(hms_[i,j])
                        #xy_, visi_, blob_score_ = self._detect_blob_gravity(hms_[i,j])
                        xys_, scores_ = self._detect_blob_gravity(hms_[i,j])
                    """
                    if self._blob_det_method == "concomp":
                        xys_, angles_, ls_, scores_ = self._detect_blob_concomp(
                            hms_[i, j]
                        )
                    elif self._blob_det_method == "nms":
                        xys_, scores_ = self._detect_blob_nms(
                            hms_[i, j], self._sigmas[scale]
                        )
                    else:
                        raise ValueError(
                            "undefined xy_comp_method: {}".format(self._xy_comp_method)
                        )
                    xys_t_ = []
                    for xy_ in xys_:
                        xys_t_.append(affine_transform(xy_, affine_mats_[i]))
                    # results[i][j][scale] = {'xy': xy_, 'visi': visi_, 'blob_size': blob_size_}
                    # results[i][j][scale] = {'xy': xy_, 'visi': visi_, 'blob_score': blob_score_}
                    results[i][j][scale] = {
                        "xys": xys_t_,
                        "angles": angles_,
                        "lengths": ls_,
                        "scores": scores_,
                        "hm": hms_[i, j],
                        "trans": affine_mats_[i],
                    }

        # print(results)
        return results
