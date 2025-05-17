import os
import os.path as osp
import matplotlib.pyplot as plt
import shutil
import torchvision.transforms as T
import pandas as pd
from pathlib import Path
import time
import logging
from collections import defaultdict
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt

from dataloaders import build_dataloader
from detectors import build_detector
from trackers import build_tracker
from utils import mkdir_if_missing, draw_frame, gen_video, Center, Evaluator
from utils.image import get_affine_transform, affine_transform
from utils.preprocess import process_video

from .base import BaseRunner


# # Build the dataloader
# transform_train = T.Compose(
#     [
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )


@torch.no_grad()
def inference_video(
    detector,
    tracker,
    input_video_path,
    frame_dir,
    cfg,
    vis_frame_dir=None,
    vis_hm_dir=None,
    vis_traj_path=None,
    dist_thresh=10.0,
):
    frames_in = detector.frames_in
    frames_out = detector.frames_out

    # +---------------
    t_start = time.time()

    det_results = []
    hm_results = []
    num_frames = 0
    print("Starting********")

    # Get all frames
    imgs_paths = sorted(Path(frame_dir).glob("*.png"))

    cap = cv2.VideoCapture(input_video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    s = max(h, w) * 1.0
    trans = np.stack(
        [
            get_affine_transform(
                c,
                s,
                0,
                [cfg["model"]["inp_width"], cfg["model"]["inp_height"]],
                inv=1,
            )
            for _ in range(3)
        ],
        axis=0,
    )
    trans = torch.tensor(trans)[None, :]
    preprocess_frame = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((cfg["model"]["inp_height"], cfg["model"]["inp_width"])),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    step = cfg["detector"]["step"]
    det_results = defaultdict(list)
    hm_results = defaultdict(list)
    img_paths_buffer = []
    frames_buffer = []
    for img_path in imgs_paths:
        # cv2.imshow("test", frame)
        # cv2.waitKey(1)
        # num_frames += imgs.shape[0] * frames_in
        frame = cv2.imread(str(img_path))
        frames_buffer.append(frame)
        img_paths_buffer.append(str(img_path))
        if len(frames_buffer) == cfg["model"]["frames_in"]:
            # Preprocess the frames
            frames_processed = [preprocess_frame(f) for f in frames_buffer]
            input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(
                0
            )  # .to(device)
            batch_results, hms_vis = detector.run_tensor(input_tensor, trans)

            for ie in batch_results[0].keys():
                path = img_paths_buffer[ie]
                preds = batch_results[0][ie]
                det_results[path].extend(preds)
                hm_results[path].extend(hms_vis[0][ie])
            if step == 1:
                frames_buffer.pop(0)
                img_paths_buffer.pop(0)
            elif step == 3:
                img_paths_buffer = []
                frames_buffer = []

    tracker.refresh()
    result_dict = {}
    print("Running tracker")
    for img_path, preds in det_results.items():
        result_dict[img_path] = tracker.update(preds)
    print("Finished tracking")

    # print(result_dict)
    t_elapsed = time.time() - t_start
    # +---------------

    cm_pred = plt.get_cmap("Reds", len(result_dict))

    x_fin, y_fin, vis_fin = [], [], []
    if cfg["model"]["name"] == "blurball":
        l_fin, theta_fin = ([], [])

    cnt = 0
    for cnt, img_path in enumerate(result_dict.keys()):
        # xy_pred = (result_dict[cnt]["x"], result_dict[cnt]["y"])
        x_pred = result_dict[img_path]["x"]
        y_pred = result_dict[img_path]["y"]
        visi_pred = result_dict[img_path]["visi"]
        score_pred = result_dict[img_path]["score"]
        if cfg["model"]["name"] == "blurball":
            angle_pred = result_dict[img_path]["angle"]
            length_pred = result_dict[img_path]["length"]

        # Save the predictions
        x_fin.append(int(min(max(x_pred, 0), 100000)))
        y_fin.append(int(min(max(y_pred, 0), 100000)))
        vis_fin.append(int(visi_pred))
        if cfg["model"]["name"] == "blurball":
            theta_fin.append(angle_pred)
            l_fin.append(length_pred)

        # cv2.imshow("test", 250 * hm_results[img_path][0]["hm"])
        # cv2.waitKey(800)

        if vis_frame_dir is not None:
            vis_frame_path = (
                osp.join(vis_frame_dir, osp.basename(img_path))
                if vis_frame_dir is not None
                else None
            )
            hm_path = (
                osp.join(vis_hm_dir, osp.basename(img_path))
                if vis_frame_dir is not None
                else None
            )
            vis_gt = cv2.imread(img_path)
            vis_pred = cv2.imread(img_path)

            for cnt2, img_path2 in enumerate(result_dict.keys()):
                if cnt2 != cnt:
                    continue
                if cnt2 > cnt:
                    break

                x_pred = result_dict[img_path2]["x"]
                y_pred = result_dict[img_path2]["y"]
                visi_pred = result_dict[img_path2]["visi"]
                score_pred = result_dict[img_path2]["score"]
                if cfg["model"]["name"] == "blurball":
                    angle_pred = result_dict[img_path2]["angle"]
                    length_pred = result_dict[img_path2]["length"]

                color_pred = (
                    int(cm_pred(cnt2)[2] * 255),
                    int(cm_pred(cnt2)[1] * 255),
                    int(cm_pred(cnt2)[0] * 255),
                )

                color_pred = (255, 0, 0)
                if cfg["model"]["name"] == "blurball":
                    vis_pred = draw_frame(
                        vis_pred,
                        center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                        color=color_pred,
                        radius=3,
                        angle=angle_pred,
                        l=length_pred,
                    )
                    vis_hm_pred = cv2.cvtColor(
                        (255 * hm_results[img_path][0]["hm"]).astype(np.uint8),
                        cv2.COLOR_GRAY2RGB,
                    )
                    vis_hm_pred = cv2.resize(vis_hm_pred, (1280, 720))
                    vis_hm_pred = draw_frame(
                        vis_hm_pred,
                        center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                        color=color_pred,
                        radius=3,
                        angle=angle_pred,
                        l=length_pred,
                    )
                else:
                    vis_pred = draw_frame(
                        vis_pred,
                        center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                        color=color_pred,
                        radius=3,
                    )
                    vis_hm_pred = cv2.cvtColor(
                        (255 * hm_results[img_path][0]["hm"]).astype(np.uint8),
                        cv2.COLOR_GRAY2RGB,
                    )
                    vis_hm_pred = draw_frame(
                        vis_hm_pred,
                        center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                        color=color_pred,
                        radius=3,
                    )

            # vis = np.hstack((vis_gt, vis_pred))
            vis = vis_pred
            cv2.imwrite(vis_frame_path, vis)
            cv2.imwrite(hm_path, vis_hm_pred)

        # if vis_traj_path is not None:
        #     color_pred = (
        #         int(cm_pred(cnt)[2] * 255),
        #         int(cm_pred(cnt)[1] * 255),
        #         int(cm_pred(cnt)[0] * 255),
        #     )
        #     vis = visualizer.draw_frame(
        #         vis,
        #         center_gt=center_gt,
        #         color_gt=color_gt,
        #     )

    if vis_frame_dir is not None:
        video_path = "{}.mp4".format(vis_frame_dir)
        gen_video(video_path, vis_frame_dir, fps=25.0)
        print("Saving video at " + video_path)

    # Save the evaluation results
    if cfg["model"]["name"] == "blurball":
        df = pd.DataFrame(
            {
                "Frame": x_fin,
                "X": x_fin,
                "Y": y_fin,
                "Visibility": vis_fin,
                "L": l_fin,
                "Theta": theta_fin,
            }
        )
    else:
        df = pd.DataFrame(
            {"Frame": x_fin, "X": x_fin, "Y": y_fin, "Visibility": vis_fin}
        )
    df["Frame"] = df.index
    df.to_csv(osp.join(frame_dir, "traj.csv"), index=False)
    print("Saving csv at " + osp.join(frame_dir, "traj.csv"))

    return {"t_elapsed": t_elapsed, "num_frames": num_frames}


class NewVideosInferenceRunner(BaseRunner):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(cfg)
        # print(cfg["input_vid"])

        self._vis_result = cfg["runner"]["vis_result"]
        self._vis_hm = cfg["runner"]["vis_hm"]
        self._vis_traj = cfg["runner"]["vis_traj"]
        self._input_vid_path = Path(cfg["input_vid"])

    def run(self, model=None, model_dir=None):
        return self._run_model(model=model)

    def _run_model(self, model=None):
        detector = build_detector(self._cfg, model=model)
        tracker = build_tracker(self._cfg)

        # Generate frames directory for processing
        frame_dir = process_video(self._input_vid_path)
        print("Finished preprocess_video")

        t_elapsed_all = 0.0
        num_frames_all = 0

        vis_frame_dir, vis_hm_dir, vis_traj_path = None, None, None
        if self._vis_result:
            vis_frame_dir = osp.join(self._input_vid_path.parent, "frames")
            mkdir_if_missing(vis_frame_dir)
        if self._vis_hm:
            vis_hm_dir = osp.join(self._input_vid_path.parent, "hm")
            mkdir_if_missing(vis_hm_dir)
        # if self._vis_traj:
        #     vis_traj_dir = osp.join(self._output_dir, "vis_traj")
        #     mkdir_if_missing(vis_traj_dir)
        #     vis_traj_path = osp.join(vis_traj_dir, "{}_{}.png".format(match, clip_name))

        tmp = inference_video(
            detector,
            tracker,
            self._input_vid_path,
            frame_dir,
            self._cfg,
            vis_frame_dir=vis_frame_dir,
            vis_hm_dir=vis_hm_dir,
        )

        t_elapsed_all += tmp["t_elapsed"]
        num_frames_all += tmp["num_frames"]

        return
