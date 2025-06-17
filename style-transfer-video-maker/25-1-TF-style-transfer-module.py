### ❗❗❗ Important Ntice ❗❗❗
### should run this on conda env which is intalled via 
# conda_neural_style_transfer_env_backup.yaml env backup file.

import cv2
import os
import time
import subprocess

from tqdm import tqdm # show progress bar

import tensorflow as tf

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub




######### Define Constants
input_video_path = './data/videos/style-transfer/input.mp4'
origin_video_frames_output_dir = './data/videos/style-transfer/frames'
os.makedirs(origin_video_frames_output_dir, exist_ok=True)

fps = 25
gop = 75
style_transfer_target_factor = 7
style_image_path = './data/images/neural-style/kandinsky.jpg'
styled_output_frame_dir = "./data/videos/style-transfer/style-transferred-frames"
os.makedirs(styled_output_frame_dir, exist_ok=True)

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Model optimization parameters
style_weight=1e-2
content_weight=1e4
total_variation_weight = 100
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_transfer_traning_cnt = 700



######### 입력 비디오에서 프레임들을 추출해서 저장한다.
cap = cv2.VideoCapture(input_video_path)
frame_cnt = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break;
    cv2.imwrite(f'{origin_video_frames_output_dir}/{frame_cnt:05d}.png', frame)
    frame_cnt += 1

cap.release()



######### 프레임들을 style_transfer_target_factor size 간격으로 추출해준다.
### (ex : frame number 0, 7, 14, ...)
for filename in os.listdir(origin_video_frames_output_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        # Extract the numeric part of the filename: e.g., "frame_000123.jpg" → 123
        number_part = filename.split(".")[0]
        try:
            frame_num = int(number_part)
            if frame_num % style_transfer_target_factor != 0:
                os.remove(os.path.join(origin_video_frames_output_dir, filename))
        except ValueError:
            # Skip files that don't match the expected pattern
            continue

image_files = sorted([
    f for f in os.listdir(origin_video_frames_output_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Rename files to 00000.png, 00001.png, ...
for new_idx, filename in enumerate(image_files):
    ext = os.path.splitext(filename)[1]  # Keep extension (e.g., .png)
    new_name = f"{new_idx:05d}{ext}"
    
    src_path = os.path.join(origin_video_frames_output_dir, filename)
    dst_path = os.path.join(origin_video_frames_output_dir, new_name)

    os.rename(src_path, dst_path)



######### Do Style Transfer - used TensorFlow Tutorial and VGG19 model.
### url : https://www.tensorflow.org/tutorials/generative/style_transfer

# === FUNCTIONS ===
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# === STYLE MODEL SETUP ===
style_image = load_img(style_image_path)
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
  """ Creates a VGG model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
  return result / (num_locations)


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}


extractor = StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']


def style_content_loss(outputs, content_targets, image):
  style_outputs = outputs['style']
  content_outputs = outputs['content']
  style_loss = tf.add_n([
      tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
      for name in style_outputs.keys()
  ])
  style_loss *= style_weight / num_style_layers

  content_loss = tf.add_n([
      tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
      for name in content_outputs.keys()
  ])
  content_loss *= content_weight / num_content_layers

  total_variation_loss = tf.image.total_variation(image)
  total_variation_loss *= total_variation_weight

  return style_loss + content_loss + total_variation_loss

@tf.function()
def train_step(image, content_targets):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs, content_targets, image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


# === BATCH STYLE TRANSFER ===

frame_files = sorted([f for f in os.listdir(origin_video_frames_output_dir) if f.endswith('.png')])
""" ❗Use this when fails to import tqdm util❗
for i, fname in enumerate(frame_files):
    print(f"[{i+1}/{len(frame_files)}] Processing {fname}...")

    content_image = load_img(os.path.join(origin_video_frames_output_dir, fname))
    content_targets = extractor(content_image)['content']
    image = tf.Variable(content_image)

    # Run a few iterations (you can tune this)
    for step in range(style_transfer_traning_cnt):
        train_step(image, content_targets)

    output_image = tensor_to_image(image)
    output_path = os.path.join(styled_output_frame_dir, fname)
    output_image.save(output_path)
"""

for fname in tqdm(frame_files, desc="Style Transferring Frames"):
    content_image = load_img(os.path.join(origin_video_frames_output_dir, fname))
    content_targets = extractor(content_image)['content']
    image = tf.Variable(content_image)

    # Style transfer iterations per frame
    for step in range(style_transfer_traning_cnt):
        train_step(image, content_targets)

    output_image = tensor_to_image(image)
    output_path = os.path.join(styled_output_frame_dir, fname)
    output_image.save(output_path)

print("✅ Style transfer completed for all frames.")