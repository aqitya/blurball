from .heatmaps import (
    BinaryFixedSizeMapGenerator,
    PrototypeBasedBinaryMapGenerator,
    BinaryLineFixedSizeMapGenerator,
)

__hm_generator_factory = {
    "binary_fixed_size": BinaryFixedSizeMapGenerator,
    "binary_line_fixed_size": BinaryLineFixedSizeMapGenerator,
    "binary_prototype": PrototypeBasedBinaryMapGenerator,
}


def select_heatmap_generator(cfg):
    hm_generator_name = cfg["name"]
    if hm_generator_name not in __hm_generator_factory.keys():
        raise KeyError("unknown hm_generator: {}".format(hm_generator_name))
    return __hm_generator_factory[hm_generator_name](cfg)
