from .online import Track
import numpy as np


class OnlineTrackerBlur:
    def __init__(self, cfg):
        self._max_disp = cfg["tracker"]["max_disp"]
        self._fid = 0
        self._track = Track()

    def _select_best(self, frame_dets):
        best_score = -np.inf
        visi = False
        x, y = -np.inf, -np.inf
        angle, length = -np.inf, -np.inf

        xy_pred = None

        for det in frame_dets:
            score = det["score"]
            if xy_pred is not None:
                qscore = self._compute_quality(
                    xy_pred, det["xy"], self._track.xy(self._fid - 1)
                )
                score += qscore

            if score > best_score:
                best_score = score
                xy = det["xy"]
                x, y = xy[0], xy[1]
                visi = True
                angle = det["angle"]
                length = det["length"]
        return x, y, angle, length, visi, best_score

    def _select_not_too_far(self, frame_dets):
        if (self._fid == 0) or (not self._track.is_visible(self._fid - 1)):
            return frame_dets

        frame_dets_ = []
        for det in frame_dets:
            if (
                np.linalg.norm(det["xy"] - self._track.xy(self._fid - 1))
                < self._max_disp
            ):
                frame_dets_.append(det)
        return frame_dets_

    def _compute_quality(self, xy1, xy2, xy3):
        return -np.linalg.norm(xy1 - xy2)

    def update(self, frame_dets):
        frame_dets = self._select_not_too_far(frame_dets)
        x, y, angle, length, visi, score = self._select_best(frame_dets)
        self._track.add(self._fid, x, y, visi, score)

        self._fid += 1
        return {
            "x": x,
            "y": y,
            "angle": angle,
            "length": length,
            "visi": visi,
            "score": score,
        }

    def refresh(self):
        self._fid = 0
        self._track = Track()
