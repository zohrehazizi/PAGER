#############################
#    Author: Xuejing Lei    #
# Modified by: Zohreh Azizi #
#############################

import numpy as np
import os
import sys
sys.path.insert(0, './TTUR')
import fid
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import pickle
from data import Normalize

tf_config = tf.ConfigProto(
    device_count = {'GPU':1}
    )

def calculate_FID(images, dsname):
    # FID: input:(HxWx3), normalized to [0, 255]
    # Given 10000 generated images and 10000 test images

    if np.abs(np.max(images)-255)>1e-5 or np.abs(np.min(images)-0)>1e-5:
        print("Image Range wrong: ({},{}), Normailizing to [0, 255]".format(np.min(images), np.max(images)))
        images = Normalize(images, a=0, b=255)
        images = np.round(images).astype(int)

    shuffle_idx = np.arange(images.shape[0])
    np.random.shuffle(shuffle_idx)
    images = images[shuffle_idx][:10000]
    if images.shape[3]==1:
        images = np.repeat(images, 3, axis=3)
    print("In calculate_FID: images shape {}".format(images.shape))

    # Paths
    # image_path = './local00/bioinf/tmp/' # set path to some generated images
    if dsname == 'lsun':
        stats_path = '../TTUR/fid_stats_{}_train.npz'.format(dsname) # training set statistics
    # elif dsname == 'lsun-bedroom':
    #     # randomly select 10000 samples from train set
    #     stats_path = '/media/mcl418-2/Data/Generative/TTUR/fid_stats_{}_test.npz'.format(dsname) # training set statistics
    else:
        stats_path = '../TTUR/fid_stats_{}_test.npz'.format(dsname) # training set statistics
    inception_path = fid.check_or_download_inception('../TTUR') # download inception network

    # loads all images into memory (this might require a lot of RAM!)
    # image_list = glob.glob(os.path.join(datapath, '*.jpg'))
    # images = np.moveaxis(images,1,3)
    # images = np.array([imread(str(fn)).astype(np.float32) for fn in files])

    # load precalculated training set statistics
    f = np.load(stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()

    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=50)

    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    print("======== FID: {}".format(fid_value))
    return fid_value

if __name__ == "__main__":
    """
    dsname = "cifar10"
    gen_path = "/media/mcl418-2/Data/Generative/ssl_cw1/results/0514/{}/models/".format(dsname)
    fw=open(gen_path+'generated_images_refined.pkl','rb')
    images = pickle.load(fw)
    fw.close()
    print("images size: ", images.shape)
    calculate_FID(images, dsname)

    fw=open(gen_path+'generated_images.pkl','rb')
    images = pickle.load(fw)
    fw.close()
    print("images size: ", images.shape)
    calculate_FID(images, dsname)
    """

    dsname = 'mnist'
    gen_path = sys.argv[1]
    images = np.load(gen_path)
    images = images.transpose(0,2,3,1)
    fid_score = calculate_FID(images, dsname)
    print(fid_score)