import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from .train_and_test import Trainer
from .eval import VideosInferenceRunner
from .eval_blurball import BlurVideosInferenceRunner
from .inference import NewVideosInferenceRunner
from .extract_frame import ExtractFrameRunner

log = logging.getLogger(__name__)

__runner_factory = {
    "train": Trainer,
    "eval": VideosInferenceRunner,
    "eval_blurball": BlurVideosInferenceRunner,
    "extract_frame": ExtractFrameRunner,
    "inference": NewVideosInferenceRunner,
}


def select_runner(
    cfg: DictConfig,
):
    runner_name = cfg["runner"]["name"]
    if not runner_name in __runner_factory.keys():
        raise KeyError("unknown runner: {}".format(runner_name))
    return __runner_factory[runner_name](cfg)
