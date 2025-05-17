from .intra_frame_peak import IntraFramePeakTracker
from .online import OnlineTracker
from .online_blur import OnlineTrackerBlur

__tracker_factory = {
    "intra_frame_peak": IntraFramePeakTracker,
    "online": OnlineTracker,
    "online_blur": OnlineTrackerBlur,
}


def build_tracker(cfg):
    tracker_name = cfg["tracker"]["name"]
    if tracker_name not in __tracker_factory.keys():
        raise KeyError("unknown tracker: {}".format(tracker_name))
    return __tracker_factory[tracker_name](cfg)
