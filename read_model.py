import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import glob
import cv2
import scipy
import numpy as np


def imsave(image, path):
    return scipy.misc.imsave(path, image)



sess = tf.Session()
with gfile.FastGFile('model_save_while_test.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

# print(sess.graph.as_graph_def())

log_dir = 'log_model_while_test'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = tf.summary.FileWriter(log_dir)
writer.add_graph(sess.graph)
writer.flush()

GT_file_list = sorted(glob.glob('data/Test/Set12' + '/*.png'))

input_x = sess.graph.get_tensor_by_name('images:0')
op = sess.graph.get_tensor_by_name('conv17/output:0')

for i in range(len(GT_file_list)):
    GT_img = cv2.imread(GT_file_list[i], 0) / 255
    GT_img = GT_img.reshape(1, GT_img.shape[0], GT_img.shape[1], 1)
    noise = np.random.normal(0, 25 / 255.0, GT_img.shape)
    test_img = GT_img + noise

    result = sess.run(op, feed_dict={input_x: test_img})
    print(result.shape)
    result = result.squeeze()
    image_path = os.path.join(os.getcwd(), 'test_model_while_test')
    image_path1 = os.path.join(image_path, "test_model_image_" + str(i) + ".png")
    imsave(result, image_path1)