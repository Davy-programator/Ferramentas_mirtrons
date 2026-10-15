import tensorflow as tf

gpus = tf._config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf._config.experimental.set_memory_growth(gpu, True)
tf._config.experimental.set_virtual_device_configuration(gpus[0], [tf._config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])