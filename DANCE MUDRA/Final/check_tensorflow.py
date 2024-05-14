import tensorflow as tf

try:
    tf_version = tf.__version__
    print("TensorFlow version:", tf_version)
except ImportError:
    print("TensorFlow is not installed.")
