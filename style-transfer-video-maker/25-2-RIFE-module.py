### ❗❗❗ Important Ntice ❗❗❗
### should run this on conda env which is intalled via 
# conda_rife_env_backup.yaml  env backup file.

import os
import sys
import subprocess

PYTHON_EXEC = "/home/alphaai/miniconda3/envs/rife/bin/python"
RIFE_ROOT_DIR_PATH = "/home/alphaai/Documents/ECCV2022-RIFE"
RIFE_SCRIPT_PATH = "/home/alphaai/Documents/ECCV2022-RIFE/inference_video.py"
RIFE_INPUT_FRAMES_PATH = "/home/alphaai/Documents/FirstAIPractice/data/videos/style-transfer/style-transferred-frames"
RIFE_EXP_FACTOR = 3
RIFE_OUTPUT_DIR_PATH = "/home/alphaai/Documents/FirstAIPractice/data/videos/style-transfer/rife-frames"

FPS = 30
RIFE_RESULT_MP4_DIR = "/home/alphaai/Documents/FirstAIPractice/data/videos/style-transfer/rife-frames/rife_result.mp4"

RIFE_VID_DIR = "/home/alphaai/Documents/ECCV2022-RIFE/vid_out"


### apply RIFE on style-transferred frame images.
rife_command = [
    PYTHON_EXEC,
    RIFE_SCRIPT_PATH,
    "--img", RIFE_INPUT_FRAMES_PATH,
    "--exp", str(RIFE_EXP_FACTOR),
    "--output", RIFE_OUTPUT_DIR_PATH
]

rife_process = subprocess.Popen(
    rife_command,
    cwd="/home/alphaai/Documents/ECCV2022-RIFE",  # ✅ Required for relative model paths to work
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in rife_process.stdout:
    print(line.strip())

rife_process.wait()

if rife_process.returncode != 0:
    print(f"❌ RIFE failed with return code {rife_process.returncode}.")
    sys.exit("Aborting the script")
else:
    print("✅ RIFE frame interpolation completed successfully.")



### apply ffmpeg on RIFE result image files and makes a result mp4.
ffmpeg_cmd = [
    "ffmpeg",
    "-framerate", str(FPS), "-i", "vid_out/%07d.png", "-c:v",
    "libx264", "-pix_fmt", "yuv420p",
    RIFE_RESULT_MP4_DIR
]

ffmpeg_process = subprocess.Popen(
    ffmpeg_cmd,
    cwd=RIFE_ROOT_DIR_PATH,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in ffmpeg_process.stdout:
    print(line.strip())

ffmpeg_process.wait()

if ffmpeg_process.returncode == 0:
    print("✅ FFmpeg completed successfully.")
else:
    print(f"❌ FFmpeg failed with return code {ffmpeg_process.returncode}.")
    sys.exit("Aborting the script")


### clean up RIFE result images.
for filename in os.listdir(RIFE_VID_DIR):
    file_path = os.path.join(RIFE_VID_DIR, filename)
    os.remove(file_path)