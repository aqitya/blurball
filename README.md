# BlurBall

Code & dataset repository for the paper: **[BlurBall: Ball detection with blur estimation]()**

![demo](demo.gif)

This repo is forked from [WASB: Widely Applicable Strong Baseline for Sports Ball Detection and Tracking](https://github.com/nttcom/WASB-SBDT/)
We added the training scripts and and other modifications.

## Dataset

The table tennis ball dataset includes both the positions of the balls (either midpoint or endpoint) and the motion blur associated (length and orientation).

It can be dowloaded from here: [NextCloud](https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7)

## Weights

All trained model weights for BlurBall, WASB, TrackNetv2, ResTrackNetv2, BallSeg, DeepBall, DeepBall large and Monotrack are available here: [Nextcloud](https://cloud.cs.uni-tuebingen.de/index.php/s/6Z8TpM3sXRKHzGC)


## Inference

Because the BlurBall is multiple input multiple output, it is quite sensitive to duplicated frames. This often happens on online recordings where videos recorded at 25fps are encoded at 30fps. It will generate a directory with the unique frames at the same location as the input video.

Run inference on a video:

```
python main.py --config-name=inference_<model> detector.model_path=<path to corresponding model> +input_vid=<path to vid>
```

## Training
To come


<!-- ## Citation -->

<!-- If you find this work useful, please consider to cite our paper: -->

<!-- ``` -->
<!-- @inproceedings{gossard2025, -->
<!-- 	title={BlurBall: Ball detector with blur estimation}, -->
<!-- 	author={Gossar}, -->
<!-- 	booktitle={}, -->
<!-- 	year={2023} -->
<!-- } -->
<!-- ``` -->

