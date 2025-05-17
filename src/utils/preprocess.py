"""
Script to seperate the images in unique frames
Some videos are filmed at 25 fps but encoded at 30 fps which leads to duplicated frames that interfere with MIMO of BlurBall
"""
import cv2
import os
import argparse
from skimage.metrics import structural_similarity as ssim

# Parameters
SSIM_THRESHOLD = 0.99  # SSIM threshold for detecting similar frames (1.0 is identical)


def process_video(video_path):
    video_path = os.path.abspath(video_path)
    video_dir = os.path.dirname(video_path)
    video_name, _ = os.path.splitext(os.path.basename(video_path))

    # Output directory relative to the video file
    frames_dir = os.path.join(video_dir, "frames_" + video_name)
    os.makedirs(frames_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    prev_frame_gray = None
    unique_index = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame_gray is None:
            # Save first frame
            frame_filename = f"{unique_index:05d}.png"
            cv2.imwrite(os.path.join(frames_dir, frame_filename), frame)
            unique_index += 1
        else:
            score = ssim(prev_frame_gray, frame_gray)
            if score < SSIM_THRESHOLD:
                frame_filename = f"{unique_index:05d}.png"
                cv2.imwrite(os.path.join(frames_dir, frame_filename), frame)
                unique_index += 1

        prev_frame_gray = frame_gray

    cap.release()

    removed = total_frames - unique_index
    removed_ratio = removed / total_frames if total_frames > 0 else 0
    # print(f"Processed {video_name}:")
    # print(f"  Total frames: {total_frames}")
    # print(f"  Unique frames saved: {unique_index}")
    # print(f"  Frames removed (similar): {removed}")
    # print(f"  Ratio removed: {removed_ratio:.2%}")
    return frames_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract unique frames from a video using SSIM."
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    args = parser.parse_args()

    process_video(args.video_path)
