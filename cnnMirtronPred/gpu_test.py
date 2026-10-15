import os
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# tf.keras.backend.set_session(tf.Session(config=config));

print('*' * 30)
gpu_available = tf.test.is_gpu_available()
print(gpu_available)

print('*' * 30)
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print(is_cuda_gpu_available)

print('*' * 30)
is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))
print(is_cuda_gpu_min_3)
