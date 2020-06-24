import numpy as np
import tensorflow as tf

import os
import time
import glob
import cv2
import scipy
import matplotlib.pyplot as plt

from tensorrt.lite import Engine
from tensorrt.infer import LogSeverity
import tensorrt

def imsave(image, path):
    return scipy.misc.imsave(path, image)


engine_single = Engine(PLAN="dncnn.engine")
GT_data_dir = 'data/Test/Set12'
GT_file_list = sorted(glob.glob(GT_data_dir + '/*.png'))


print("TensorRT Testing...")
for i in range(len(GT_file_list)):
    GT_img = cv2.imread(GT_file_list[i], 0) / 255
    GT_img_reshape = GT_img.reshape(1, GT_img.shape[0], GT_img.shape[1], 1)
    noise = np.random.normal(0, 25 / 255.0, GT_img_reshape.shape)
    test_img = GT_img_reshape + noise
    print(test_img.shape)
    test_img_transpose = np.transpose(test_img, [0, 3, 1, 2])

    start_time2 = time.time()
    result = engine_single.infer(test_img_transpose)  # 这是个list类型的输出
    print("process one image use: %4.4f s" % (time.time() - start_time2))


    result = np.array(result).squeeze(axis=0)
    result = np.transpose(result, [0, 2, 3, 1])

    MSE_DNCNN = np.float32(np.mean(np.square(result - GT_img_reshape)))
    PSNR_DNCNN = np.multiply(10.0, np.log(1.0 * 1.0 / MSE_DNCNN) / np.log(10.0))
    print('Picture[%d]   MSE:[%.8f]   PSNR:[%.4f] ---------DNCNN_trt' % ((i + 1), MSE_DNCNN, PSNR_DNCNN))

    result = result.squeeze()
    image_path = os.path.join(os.getcwd(), 'result_dncnn')
    image_path1 = os.path.join(image_path, "test_trt_image_" + str(i) + ".png")
    imsave(result, image_path1)




