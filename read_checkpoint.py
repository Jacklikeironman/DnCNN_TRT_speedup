import os
import path
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


print(" [*] Reading checkpoints...")
model_dir = "%s_%s" % ("dncnn", "sigma25")
checkpoint_dir = os.path.join('checkpoint_dncnn', model_dir)

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
ckpt_name = ckpt.model_checkpoint_path
print('ckptpath: %s' % ckpt_name)
reader=pywrap_tensorflow.NewCheckpointReader(ckpt_name)
var_to_shape_map=reader.get_variable_to_shape_map()

data_print=np.array([])
for key in var_to_shape_map:
    print('############################################################')
    print('tensor_name',key)
    ckpt_data=np.float64(np.array(reader.get_tensor(key)))#cast list to np arrary
    print(ckpt_data)


