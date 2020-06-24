import os
import path
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib import tensorrt as trt
from tensorflow.python.platform import gfile


OUTPUT_NAMES = ["conv17/output"]
MAX_BATCHSIZE = 1
MAX_WORKSPACE = 1 << 30

sess = tf.Session()
with gfile.FastGFile('model_save_while_test.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')


writer = tf.summary.FileWriter('log_before_trt')
writer.add_graph(sess.graph)
writer.flush()

graphdef = tf.get_default_graph().as_graph_def()
frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, graphdef, OUTPUT_NAMES)
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph_def,
    outputs = OUTPUT_NAMES,
    max_batch_size=MAX_BATCHSIZE,
    max_workspace_size_bytes=MAX_WORKSPACE,
    minimum_segment_size=3
)
writer = tf.summary.FileWriter('log_after_trt')
writer.add_graph(sess.graph)
writer.flush()


