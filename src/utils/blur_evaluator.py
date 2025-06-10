import numpy as np
import logging

log = logging.getLogger(__name__)


class BlurEvaluator(object):
    def __init__(self, cfg):
        self._dist_threshold = cfg["runner"]["eval"]["dist_threshold"]
        self._tp = 0
        self._fp1 = 0
        self._fp2 = 0
        self._tn = 0
        self._fn = 0
        self._ses = []  # squared error
        self._l_aes = []  # Blur abs error
        self._angle_aes = []  # Angle abs error
        self._scores = []
        self._ys = []

    def eval_single_frame(
        self,
        xy_pred,
        angle_pred,
        l_pred,
        visi_pred,
        score_pred,
        xy_gt,
        angle_gt,
        l_gt,
        visi_gt,
    ):
        tp, fp1, fp2, tn, fn = 0, 0, 0, 0, 0
        se = None
        if visi_gt:
            if visi_pred:
                if (
                    np.linalg.norm(np.array(xy_pred) - np.array(xy_gt))
                    < self._dist_threshold
                ):
                    tp += 1
                else:
                    fp1 += 1
                se = np.linalg.norm(np.array(xy_pred) - np.array(xy_gt)) ** 2
                l_ae = np.abs((l_pred - l_gt))
                if l_pred > 4:
                    if angle_gt > 90:
                        angle_ae = np.abs((angle_pred + 180 - angle_gt))
                    elif angle_gt < -90:
                        angle_ae = np.abs((angle_pred - 180 - angle_gt))
                    else:
                        angle_ae = np.abs((angle_pred - angle_gt))
                    self._angle_aes.append(angle_ae)
                # print(angle_pred)
                # print(angle_gt)
                # print(angle_ae)
                # print()
                self._ses.append(se)
                self._l_aes.append(l_ae)
            else:
                fn += 1
        else:
            if visi_pred:
                fp2 += 1
            else:
                tn += 1
        self._tp += tp
        self._fp1 += fp1
        self._fp2 += fp2
        self._tn += tn
        self._fn += fn

        if tp > 0 or fp1 > 0 or fp2 > 0:
            if tp > 0:
                self._ys.append(1)
            else:
                self._ys.append(0)
            self._scores.append(score_pred)

        return {"tp": tp, "tn": tn, "fp1": fp1, "fp2": fp2, "fn": fn, "se": se}

    @property
    def dist_threshold(self):
        return self._dist_threshold

    @property
    def tp_all(self):
        return self._tp

    @property
    def fp1_all(self):
        return self._fp1

    @property
    def fp2_all(self):
        return self._fp2

    @property
    def fp_all(self):
        return self.fp1_all + self.fp2_all

    @property
    def tn_all(self):
        return self._tn

    @property
    def fn_all(self):
        return self._fn

    @property
    def prec(self):
        prec = 0.0
        if (self.tp_all + self.fp_all) > 0.0:
            prec = self.tp_all / (self.tp_all + self.fp_all)
        return prec

    @property
    def recall(self):
        recall = 0.0
        if (self.tp_all + self.fn_all) > 0.0:
            recall = self.tp_all / (self.tp_all + self.fn_all)
        return recall

    @property
    def f1(self):
        f1 = 0.0
        if self.prec + self.recall > 0.0:
            f1 = 2 * self.prec * self.recall / (self.prec + self.recall)
        return f1

    @property
    def accuracy(self):
        accuracy = 0.0
        if self.tp_all + self.tn_all + self.fp_all + self.fn_all > 0.0:
            accuracy = (self.tp_all + self.tn_all) / (
                self.tp_all + self.tn_all + self.fp_all + self.fn_all
            )
        return accuracy

    @property
    def sq_errs(self):
        return self._ses

    @property
    def ap(self):
        inds = np.argsort(-1 * np.array(self._scores)).tolist()
        tp = 0
        r2p = {}
        for i, ind in enumerate(inds, start=1):
            tp += self._ys[ind]
            p = tp / i
            r = tp / (self.tp_all + self.fn_all)
            if not r in r2p.keys():
                r2p[r] = p
            else:
                if r2p[r] < p:
                    r2p[r] = p
        prev_r = 0
        ap = 0.0
        for r, p in r2p.items():
            ap += (r - prev_r) * p
            prev_r = r
        return ap

    @property
    def rmse(self):
        _rmse = -np.inf
        if len(self.sq_errs) > 0:
            _rmse = np.sqrt(np.array(self.sq_errs).mean())
        return _rmse

    @property
    def l_mae(self):
        _mae = -np.inf
        if len(self._l_aes) > 0:
            _mae = np.mean(np.array(self._l_aes))
        return _mae

    @property
    def l_std(self):
        _std = -np.inf
        if len(self._l_aes) > 0:
            _std = np.std(np.array(self._l_aes))
        return _std

    @property
    def angle_mae(self):
        _mae = -np.inf
        if len(self._angle_aes) > 0:
            _mae = np.mean(np.array(self._angle_aes))
        return _mae

    @property
    def angle_std(self):
        _std = -np.inf
        if len(self._l_aes) > 0:
            _std = np.std(np.array(self._angle_aes))
        return _std

    def print_results(self, txt=None, elapsed_time=0.0, num_frames=0, with_ap=True):
        if txt is not None:
            log.info("{}".format(txt))
        if num_frames > 0:
            log.info(
                "Elapsed time: {}, FPS: {} ({}/{})".format(
                    elapsed_time, num_frames / elapsed_time, num_frames, elapsed_time
                )
            )
        if with_ap:
            log.info(
                "| TP   | TN   | FP1   | FP2   | FP   | FN   | Prec       | Recall       | F1       | Accuracy       | RMSE | AP  |"
            )
            log.info(
                "| ---- | ---- | ----- | ----- | ---- | ---- | ---------- | ------------ | -------- | -------------- | ---- | ----- |"
            )
            log.info(
                "| {tp} | {tn} | {fp1} | {fp2} | {fp} | {fn} | {prec:.4f} | {recall:.4f} | {f1:.4f} | {accuracy:.4f} | {rmse:.2f}({num_ses}) | {ap:.4f} |".format(
                    tp=self.tp_all,
                    tn=self.tn_all,
                    fp1=self.fp1_all,
                    fp2=self.fp2_all,
                    fp=self.fp_all,
                    fn=self.fn_all,
                    prec=self.prec,
                    recall=self.recall,
                    f1=self.f1,
                    accuracy=self.accuracy,
                    rmse=self.rmse,
                    num_ses=len(self.sq_errs),
                    ap=self.ap,
                )
            )
            log.info(f"L RMSE: {self.l_mae} +/- {self.l_std}")
            log.info(f"Angle RMSE: {self.angle_mae} +/- {self.angle_std}")
        else:
            log.info(
                "| TP   | TN   | FP1   | FP2   | FP   | FN   | Prec       | Recall       | F1       | Accuracy       | RMSE |"
            )
            log.info(
                "| ---- | ---- | ----- | ----- | ---- | ---- | ---------- | ------------ | -------- | -------------- | ---- |"
            )
            log.info(
                "| {tp} | {tn} | {fp1} | {fp2} | {fp} | {fn} | {prec:.4f} | {recall:.4f} | {f1:.4f} | {accuracy:.4f} | {rmse:.2f}({num_ses}) |".format(
                    tp=self.tp_all,
                    tn=self.tn_all,
                    fp1=self.fp1_all,
                    fp2=self.fp2_all,
                    fp=self.fp_all,
                    fn=self.fn_all,
                    prec=self.prec,
                    recall=self.recall,
                    f1=self.f1,
                    accuracy=self.accuracy,
                    rmse=self.rmse,
                    num_ses=len(self.sq_errs),
                )
            )
            log.info(f"L RMSE: {self.l_mae} +/- {self.l_std}")
            log.info(f"Angle RMSE: {self.angle_mae} +/- {self.angle_std}")
