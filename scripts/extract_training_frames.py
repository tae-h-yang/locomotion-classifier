import os
import re
import cv2
from tqdm import tqdm

# Path setup
this_path = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(this_path, "../data/videos")
OUTPUT_DIR = os.path.join(this_path, "../data/train")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
FRAME_INTERVAL = 5
RGB_PREFIX = "rgb"
DEPTH_PREFIX = "depth"
RGB_PATTERN = re.compile(r"^rgb_\d{3}_[a-z]+\.mp4$")
DEPTH_PATTERN = re.compile(r"^depth_\d{3}_[a-z]+\.mp4$")


# Helper function
def extract_frames(video_path, prefix, label, counter):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    with tqdm(
        total=total_frames, desc=f"Extracting {prefix}_{label}", unit="frame"
    ) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_INTERVAL == 0:
                filename = f"{prefix}_{counter:04d}_{label}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)
                counter += 1
            frame_idx += 1
            pbar.update(1)
    cap.release()
    return counter


# List and validate videos
rgb_videos = sorted([f for f in os.listdir(VIDEO_DIR) if RGB_PATTERN.match(f)])
depth_videos = sorted([f for f in os.listdir(VIDEO_DIR) if DEPTH_PATTERN.match(f)])

# Main loop
rgb_counter = 0
depth_counter = 0

for rgb_file, depth_file in zip(rgb_videos, depth_videos):
    label_rgb = rgb_file.split("_")[-1].replace(".mp4", "")
    label_depth = depth_file.split("_")[-1].replace(".mp4", "")
    if label_rgb != label_depth:
        print(f"Label mismatch: {rgb_file}, {depth_file}")
        continue

    rgb_counter = extract_frames(
        os.path.join(VIDEO_DIR, rgb_file), "rgb", label_rgb, rgb_counter
    )
    depth_counter = extract_frames(
        os.path.join(VIDEO_DIR, depth_file), "depth", label_depth, depth_counter
    )

print(f"\nSaved {rgb_counter} RGB and {depth_counter} depth images to {OUTPUT_DIR}")
