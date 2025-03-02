import tensorflow as tf
import cv2

print("TensorFlow version:", tf.__version__)
print("CUDA enabled:", tf.test.is_built_with_cuda())
print("OpenCV version:", cv2.__version__)
print("GPU detected:", tf.config.list_physical_devices('GPU'))

# Check if cuDNN is available
try:
    from tensorflow.python.platform import build_info as tf_build_info
    print("cuDNN version:", tf_build_info.build_info['cudnn_version'])
except:
    print("cuDNN is NOT available")