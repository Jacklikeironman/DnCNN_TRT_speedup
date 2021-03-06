from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model_original import DNCNN
import pprint
import os

# ##############################################################################
# import pycuda.driver as cuda
# import pycuda.autoinit
# import argparse
#
# import uff
#
# import tensorrt as trt
# #from tensorflow.contrib import tensorrt as tfrt
# from tensorrt.parsers import uffparser
#
# MAX_WORKSPACE = 1 << 30
# G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
# MAX_BATCHSIZE = 1
# ###############################################################################

flags = tf.app.flags
flags.DEFINE_integer("epoch", 40, "Number of epoch [20]")
flags.DEFINE_integer("batch_size",256, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 40, "The size of image to use [40]")
flags.DEFINE_integer("label_size", 40, "The size of label to produce [40]")
flags.DEFINE_float("lr_init", 1e-3, "The learning rate of gradient descent algorithm [1e-3]")
flags.DEFINE_integer("depth", 17, "Depth of Network. [17]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_string("train_labels_dir", "data/Train400", "Name of train labels directory [data/Train400]")
flags.DEFINE_string("test_labels_dir", "data/Test/Set12", "Name of test labels directory [data/Test/Set12]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_dncnn", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("result_dir", "result_dncnn", "Name of result directory [result]")
flags.DEFINE_string("log_dir", 'Log_dncnn', "Name of log directory[Log]")
flags.DEFINE_boolean("is_train", True, "BN parameters and processing mode,True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    with tf.Session() as sess:
        dncnn = DNCNN(sess,FLAGS)
        if FLAGS.is_train:
            dncnn.train()
        else:
            dncnn.test()

if __name__ == '__main__':
    tf.app.run()