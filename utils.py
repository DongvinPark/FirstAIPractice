from time import gmtime, strftime
import os
import cv2
#import torch
from PIL import Image, ImageDraw
import numpy as np
import re
#import logging
#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def is_chinese(text:str)->bool:
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return chinese_char_pattern.search(text) is not None

def gettime():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def extract_frames_to_files(video_path, frame_dir, ext='png',max_frame:int = None):
    """Extracts frames from video.
        Input arg: video path, output directory for images.
    """
    if not os.path.exists(frame_dir):
        print("Tasik, no directory available for frames, exit")
        return -1
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"{frame_count:04d}.{ext}")
        success = cv2.imwrite(frame_path, frame)
        if not success:
            print("Tasik, image write fails. exit.")
            return -1,-1,-1
        frame_count += 1
        if max_frame is not None and max_frame==frame_count:
            break
                            
    cap.release()
    print(f"Tasik, Extracted {frame_count} frames.")
    return frame_count, width, height

def extract_frame(video_path, max_frame:int = None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frame is not None and max_frame==len(frames):
            break
        frames.append(frame)
    cap.release()
    return frames

def get_filename_only(file_path):
    if file_path is None or os.path.isdir(file_path):return None
    return os.path.splitext(os.path.basename(file_path))[0]
    
def get_extension_only(file_path):
    if file_path is None or os.path.isdir(file_path):return None
    return os.path.splitext(os.path.basename(file_path))[1]

def get_all_files(dir_path, ext=None):
    if dir_path is None: return None
    files=[]
    for f in sorted(os.listdir(dir_path)):
        _ext = get_extension_only(f)
        if ext is not None and _ext != ext: continue
        files.append(f)     
    return files   

def xprint(*args, **kwargs):
    print(gettime()," ".join(map(str,args)), **kwargs)

#def findGpuFloatType():
#    if torch.cuda.is_available():        
#        dtype = torch.float16
#        device = 'cuda'
#    elif torch.backends.mps.is_available():
#        dtype = torch.float16   
#        device ='mps'
#    else:
#        dtype=torch.float32
#        device = 'cpu'
#    return dtype, device     

#def hasGpu():
#    _,device = findGpuFloatType()
#    return device=='cuda'

#def hasMps():
#    _,device = findGpuFloatType()
#    return device=='mps'


def process_canny(video_path):
    frames = extract_frame(video_path)
    canny_frames=[]
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result = Image.fromarray(edges)
        canny_frames.append(result) # pil type
    return canny_frames, frames

def make_video(output_path, output_frames, width, height, fps:float = 30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in output_frames:
        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    out.release()

#def clearDevice(device):
#    if device=='cuda':
#        torch.cuda.empty_cache()    
#    elif device=='mps':
#        torch.mps.empty_cache()    

# this is valid only with YOLO
def draw_yolo_pose_frame(yolo_keypoints, image_size):
    # OpenPose-style keypoint connections (COCO format)
    yolo_connections = [
        (5, 7), (7, 9),  # Left arm (Shoulder → Elbow → Wrist)
        (6, 8), (8, 10),  # Right arm (Shoulder → Elbow → Wrist)
        (5, 6),  # Shoulder connection
        (11, 13), (13, 15),  # Left leg (Hip → Knee → Ankle)
        (12, 14), (14, 16),  # Right leg (Hip → Knee → Ankle)
        (11, 12),  # Hip connection
        (5, 11), (6, 12)  # Spine (Shoulders to Hips)
    ]
    # yolo_keypoints' shape is [1, frame_number, 3], for 3 -->[x,y,confidence]
    # so we need remove the 1st element of batch size.
    yolo_keypoints = yolo_keypoints.squeeze(0)

    # Colors for visualization
    KEYPOINT_COLOR = (0, 255, 0)  # Green
    STICK_COLOR = (0, 0, 255)     # Red
    RADIUS = 5                    # Radius for keypoints
    THICKNESS = 2     
    
    # Create a black canvas
    width, height = image_size
    pose_frame = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(pose_frame)
    
    # Convert YOLO keypoints (filter out low-confidence points)
    confidence = 0.2
    keypoints = {i: (int(kp[0]), int(kp[1])) 
                 for i, kp in enumerate(yolo_keypoints) 
                 if kp[2] > confidence}

    # Draw skeleton lines
    for p1, p2 in yolo_connections:
        if p1 in keypoints and p2 in keypoints:
            draw.line([keypoints[p1], keypoints[p2]], fill='red', width=5)

    # Draw keypoints as white circles
    for x, y in keypoints.values():
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red', outline="red")

    #pose_frame.show()
    return pose_frame   