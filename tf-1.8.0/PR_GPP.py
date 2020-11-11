# Copyright 2020 Lawrence Livermore National Security, LLC and other authors: Rushil Anirudh, Suhas Lohit, Pavan Turaga
# SPDX-License-Identifier: MIT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from model import *
from utils import block_diagonal
import cPickle as pkl
from skimage.measure import compare_psnr

import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import pybm3d
import cPickle as pkl

def projector_tf(imgs,phi=None):
    csproj = tf.matmul(imgs,tf.squeeze(phi))
    return csproj

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def sample_Z(m, n):
    return np.random.uniform(-1,1,size=[m, n])


def run_calibration(test_img_name):


    I_x = I_y = 256
    d_x = d_y = 32
    dim_x = d_x*d_y
    batch_size = (I_x*I_y)//(dim_x)
    n_measure = 0.25
    lr_factor = 1.0#*batch_size//64

    n_img_plot = int(np.sqrt(batch_size))
    dim_z = 100
    dim_phi = int(n_measure*dim_x)
    nIter = 51

    iters = np.array(np.geomspace(10,10,nIter),dtype=int)


    modelsave ='./gan_models/gen_models_corrupt-cifar32'

    fname = '/p/lustre1/anirudh1/GAN/mimicGAN/IMAGENET/test_images/{}.tif'.format(test_img_name)
    savefolder = 'paper_expts/results_pr_{}/'.format(str(n_measure*100))

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    savename = savefolder+modelsave.split('_')[-1]+'_'+fname.split('/')[-1][:-4]+'.pkl'
    print(savename)
    # if os.path.exists(savename):
        # return
    x_test = Image.open(fname).convert(mode='L').resize((I_x,I_y))

    x_test_ = np.array(x_test)/255.
    x_test = []
    for i in range(n_img_plot):
        for j in range(n_img_plot):
            _x = x_test_[i*d_x:d_x*(i+1),j*d_y:d_y*(j+1)]
            x_test.append(_x)

    x_test = np.array(x_test)
    x_test = np.expand_dims(x_test,3)
    print(x_test.shape)
    test_images = x_test[:batch_size,:,:,:]
    imsave(test_images,[n_img_plot,n_img_plot],'cs_outs/gt_sample.png')

    tf.reset_default_graph()
    tf.set_random_seed(0)
    np.random.seed(4321)

    Y_obs_ph = tf.placeholder(tf.float32,[batch_size,dim_phi])
    phi_ph = tf.placeholder(tf.float32,[dim_x,dim_phi])

    tmp = tf.random_uniform([1000,dim_z],minval=-1.,maxval=1.)
    tmp = tf.expand_dims(tf.reduce_mean(tmp,axis=0),axis=0)

    z_ = tf.tile(tmp,[batch_size,1])
    z_prior_ = tf.Variable(z_,name="z_prior")

    G_sample = 0.5*generator(z_prior_,False)+0.5
    G_sample_re = tf.reshape(G_sample,[-1,dim_x])

    y_curr = projector_tf(G_sample_re,phi_ph)
    phase = tf.sign(y_curr)
    Y_obs2 = tf.multiply(phase,Y_obs_ph)

    phi_est = phi_ph
    proj_corrected = projector_tf(G_sample_re,phi_est)


    v = G_sample_re - 1e-1*tf.transpose(tf.matmul(phi_est,tf.transpose(proj_corrected-Y_obs2)))

    G_loss = tf.reduce_mean(tf.square(v - G_sample_re))

    t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    g_vars = [var for var in t_vars if 'Generator' in var.name]


    solution_opt = tf.train.RMSPropOptimizer(3e-2).minimize(G_loss, var_list=[z_prior_])


    saver = tf.train.Saver(g_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(modelsave)


        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("************ Prior restored! **************")

        nb = batch_size
        z_test = sample_Z(100,dim_z)
        phi_np = np.random.randn(dim_x,dim_phi)
        phi_test_np = 1.0*phi_np

        print(np.mean(x_test),np.min(x_test),np.max(x_test))
        y_obs = np.abs(np.matmul(test_images.reshape(-1,dim_x),phi_test_np))

        for i in xrange(nIter):

            if i %10 ==0:
                G_imgs,tr_loss = sess.run([G_sample,G_loss],feed_dict={phi_ph:phi_np,Y_obs_ph:y_obs})
                merged = merge(G_imgs,[n_img_plot,n_img_plot])
                merged_clean = pybm3d.bm3d.bm3d(merged, 0.25)
                psnr0 = compare_psnr(x_test_,merged,data_range=1.0)
                psnr1 = compare_psnr(x_test_,merged_clean,data_range=1.0)

                scipy.misc.imsave('cs_outs/inv_solution_{}.png'.format(str(i).zfill(3)),merged)
                scipy.misc.imsave('cs_outs/inv_bm3d_solution_{}.png'.format(str(i).zfill(3)),merged_clean)
                print('iter: {:d}, tr loss: {:.4f}, PSNR-raw: {:.4f}, PSNR-bm3d: {:.4f}'.format(i,tr_loss,psnr0,psnr1))

            fd = {phi_ph:phi_np,Y_obs_ph:y_obs}
            for j in range(iters[i]):
                _,tr_loss = sess.run([solution_opt,G_loss],feed_dict=fd)

        results_dict = {}
        results_dict['psnr'] = [psnr0,psnr1]
        results_dict['gprior'] = merged
        results_dict['gprior_bm3d'] = merged_clean
        pkl.dump(results_dict,open(savename,'wb'))

if __name__ == '__main__':
    test_images = ['barbara', 'Parrots','lena256','foreman','cameraman','house','Monarch']
    for test_img in test_images:
        run_calibration(test_img)
