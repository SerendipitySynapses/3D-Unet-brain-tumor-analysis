### Initialize GPU for mac
#conda create --name 3D-Unet-brain-tumor-analysis python=3.9
#conda activate 3D-Unet-brain-tumor-analysis
# conda install -c apple tensorflow-deps
# pip install tensorflow-macos
# pip install tensorflow-metal

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.config.list_physical_devices('GPU'))

#Is GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
import tensorflow.keras.backend as K
K.clear_session()

