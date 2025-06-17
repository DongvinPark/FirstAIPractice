### ❗❗❗ Important Ntice ❗❗❗
"""
ubuntu linux에서 RIFE 사용하는 방법
1. codna env를 python 3.10 버전으로 새롭게 하나 만들고, activate시킨다.
2. RIFE git repository를 깃 클론한 후, 해당 리포지토리의 root 디렉토리로 이동하여 CLI 설치 방법을 따라서 설치한다. https://github.com/hzwer/ECCV2022-RIFE 깃 리포지토리의 리드미 문서 내의 CLI Usage 부분에서 자세히 설명하고 있다.
3. Pre Trained AI 모델을 다운로드 해서 RIFE 루트 디렉토리 내에 train_log 디렉토리를 통째로 복사 붙여넣기 한다. 여기가 정확하게 실행되지 않으면 모델  inference_video.py 파일 내의 import 부분에서 에러가 뜬다.
4. RIFE 프로젝트 디렉토리 내의 RIFE.py 파일 안에 다음과 같은 코드가 존재한다.
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
            
여기에서 self.flownet ... 부분을 아래와 같은 새로운 코드로 교체한다.
	self.flownet.load_state_dict(convert(torch.load(f'{path}/flownet.pkl', weights_only=False)))
그래야 오류 없이 사용할 수 있다. 혹시라도 잘 안 될 경우, 해당 에러 로그를 chat gpt에게 전달하면서 질문해보면 된다.
5. RIFE은 00000.png ~ 00099.png 와 같은 형식의 png 파일만을 입력으로 받는다. 이 형식이 지켜지지 않을 경우, 'index out of range' 에러가 뜬다.
"""
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

origin_video_frames_output_dir = './data/videos/style-transfer/frames'


### apply RIFE on style-transferred images.
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

if rife_process.returncode == 0:
    print("✅ RIFE frame interpolation completed successfully.")
else:
    print(f"❌ RIFE failed with return code {rife_process.returncode}.")
    sys.exit("Aborting the script...")


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
    sys.exit("Aborting the script...")


### clean up all input source images.
for filename in os.listdir(RIFE_VID_DIR):
    file_path = os.path.join(RIFE_VID_DIR, filename)
    os.remove(file_path)

for filename in os.listdir(RIFE_INPUT_FRAMES_PATH):
    file_path = os.path.join(RIFE_INPUT_FRAMES_PATH, filename)
    os.remove(file_path)

for filename in os.listdir(origin_video_frames_output_dir):
    file_path = os.path.join(origin_video_frames_output_dir, filename)
    os.remove(file_path)