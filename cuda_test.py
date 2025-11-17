import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Test GPU
if tf.config.list_physical_devices('GPU'):
    print("\n✓ GPU is available and working!")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("GPU test passed:", c.numpy())
else:
    print("\n✗ No GPU detected")