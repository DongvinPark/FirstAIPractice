import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

import time

import copy

# this implementation can be found at this official Pytorch tutorial
# https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html

# CNN 중에서도 미리 빌드된 VGG19 신경망이 사용되었다.
# gpt 한테 물어본 결과, 아래와 같은 3 가지의 이유로 VGG19가 사용되었다고 한다.
#	•	Deep enough to capture meaningful structure and texture
#	•	Simple architecture (just conv + relu + pooling)
#	•	Commonly used in style transfer because it works very well empirically
# Style Transfer에 CNN이 사용되는 이유는, CNN이 합성곱 층과 풀링 층을 이용해서 이미지의 특징들을
# 잡아내는 것을 잘하기 때문이다.

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(device)

# 이미지 사이즈를 가로와 세로 모두 512로 규정한다.
imsize = 512

# 트랜스포머는 torchvision 라이브러리 내부의 image preprocessing utility 모음이다.
# 이미지 사이즈 재조정, 이미지를 텐서로 변환, 일련의 작업들을 하나의 Chain으로 바인드
# 하는 기능이 현재의 코드에서 사용되었다.
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader("./data/images/neural-style/picasso.jpg")
content_img = image_loader("./data/images/neural-style/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size!"

unloader = transforms.ToPILImage()

plt.ion()

def imshow(tensor, title=None):
    # PIL 및 matplotlib 는 CPU를 통해서 작동시키기 때문에, tensor 데이터를 cpu로 로드시켜야 한다.
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# 스타일 loss를 계산할 때 사용된다.
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

# conv_1 ~ conv_5는 VGG19의 고유한 layer들이다. 정리해보면 다음과 같다.
# Layer         Name         What it captures
# conv_1       Shallow       Edges, colors, fine textures
# conv_2                     Patterns, basic shapes
# conv_3                     Object parts, more global texture
# conv_4       Mid/Deep      Object structure, general layout
# conv_5        Deep         Abstract patterns, high-level info
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# 앞에서 다운로드 받았던 VGG19 신경망에 별도의 '층'들을 넣어서 새로운 모델을 만든다.
# 그 후, Content Loss와 Style Loss를 계산한다.
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    # 이 반복문을 통해서 다음과 같이 VGG19 신경망에 수정을 가한다.
# [Normalization] → [Conv1] → [StyleLoss] → [ReLU] → [Conv2] → [StyleLoss] → ...
#                                                  ↑
#                                       ContentLoss or StyleLoss inserted here
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


input_img = content_img.clone()
plt.figure()
imshow(input_img, title='Input Image')


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # model을 Freeze하는 부분이다.
    # VGG19라는 이미 만들어져 있는 모델을 사용하되, 해당 모델 내부의 weights & bias 는 변형을 가해서는 안 된다.
    # 이러한 상태에서 VGG19 모델은 feature extractor로서 사용되고 있으며, 이러한 방식으로
    # pre built model을 사용하는 것을 'interface mode에서 모델을 사용한다'라고 한다.
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    # 순전파와 역전파를 반복하면서 input_img를 변형시키는 것을 반복한다.
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                # '클램핑(Clamping)'이라고 한다. image 또는 tensor 를 구성하는 값이
                # 0 보다 작으면 0이 되고, 1보다 크면 1이 되게 만든다.
                # 음수 값 또는 1을 초과하는 값은 유효한 이미지 픽셀 값이 아니기 때문이다.
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
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

start_time = time.time()
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')
print("Total train & test time(sec):", time.time() - start_time)

output_image = output.cpu().clone().squeeze(0)
output_image = unloader(output_image)
output_image.save("./data/videos/style-transfer/style-transferred-frames/output_stylized.jpg")
print("Output image saved as 'output_stylized.jpg'")

plt.ioff()
plt.show()