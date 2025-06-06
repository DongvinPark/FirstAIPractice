import cv2
import os
import time
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm  # optional: shows progress bar

######### 입력 비디오에서 프레임들을 추출해서 저장한다.
input_video_path = './data/videos/style-transfer/input512.mp4'
origin_video_frames_output_dir = './data/videos/style-transfer/frames'
os.makedirs(origin_video_frames_output_dir, exist_ok=True)

cap = cv2.VideoCapture(input_video_path)
frame_cnt = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break;
    cv2.imwrite(f'{origin_video_frames_output_dir}/frame_{frame_cnt:05d}.jpg', frame)
    frame_cnt += 1

cap.release()



######### 추출된 프레임들마다 개별적으로 style transfer를 해준다.
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(device)

imsize = 512
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

unloader = transforms.ToPILImage()


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a*b, c*d)
    G = torch.mm(features, features.t())
    return G.div(a*b*c*d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses



def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward() # 역전파

            run[0] += 1

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img

# 스타일 이미지는 상수다. 변하지 않는다.
style_img = image_loader("./data/images/neural-style/picasso.jpg")
styled_output_frame_dir = "./data/videos/style-transfer/style-transferred-frames"
os.makedirs(styled_output_frame_dir, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(origin_video_frames_output_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"Found {len(image_files)} images.")

for filename in tqdm(image_files):
    input_path = os.path.join(origin_video_frames_output_dir, filename)
    output_path = os.path.join(styled_output_frame_dir, f"stylized_{filename}")

    # Load content image (frame)
    content_img = image_loader(input_path)
    input_img = content_img.clone()

    # Apply style transfer
    output = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_img, style_img, input_img
    )

    # Convert and save result
    output_image = output.cpu().clone().squeeze(0)
    output_image = unloader(output_image)
    output_image.save(output_path)

print("✅ All frames processed and saved.")


######### style-transfer를 마친 프레임들을 하나의 비디오 파일로 병합시킨다.
frame_rate = 25

my_input_path = "./data/videos/style-transfer/style-transferred-frames"
# relative to styled_output_frame_dir
fianl_result_mp4_dir = "../stylized_output_video_h264.mp4"

# FFmpeg command as a list
command = [
    "ffmpeg",
    "-framerate", str(frame_rate),
    "-i", "stylized_frame_%05d.jpg",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    fianl_result_mp4_dir
]

# Run FFmpeg with Popen and stream the output
process = subprocess.Popen(
    command,
    cwd=my_input_path,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True  # Ensures output is decoded as text instead of bytes
)

# Print FFmpeg output line-by-line
for line in process.stdout:
    print(line.strip())

# Wait for process to complete
process.wait()

# Check exit status
if process.returncode == 0:
    print("✅ FFmpeg completed successfully.")
else:
    print(f"❌ FFmpeg failed with return code {process.returncode}.")


























