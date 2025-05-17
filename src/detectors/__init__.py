from .detector import TracknetV2Detector
from .deepball_detector import DeepBallDetector
from .blurball_detector import BlurBallDetector

__factory = {
    "tracknetv2": TracknetV2Detector,
    "deepball": DeepBallDetector,
    "blurball": BlurBallDetector,
}


def build_detector(cfg, model=None):
    detector_name = cfg["detector"]["name"]
    if not detector_name in __factory.keys():
        raise KeyError("invalid detector: {}".format(detector_name))
    return __factory[detector_name](cfg, model=model)
