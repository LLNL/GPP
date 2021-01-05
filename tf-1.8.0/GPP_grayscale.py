# Copyright 2020 Lawrence Livermore National Security, LLC and other authors: Rushil Anirudh, Suhas Lohit, Pavan Turaga
# SPDX-License-Identifier: MIT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from model import generator

from skimage.measure import compare_psnr
from skimage.io import imsave
from PIL import Image
import pybm3d
import os



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

def grid_imsave(images, size, path):
    image = merge(images, size)
    imsave(path, image)
    return

def sample_Z(m, n):
    return np.random.uniform(-1,1,size=[m, n])

def GPP_solve(test_img_name='parrots', USE_BM3D=False, savedir='outs_tf'):

    modelsave ='./gan_models/gen_models_corrupt-cifar32'
    fname = '/p/lustre1/anirudh1/GAN/mimicGAN/IMAGENET/test_images/{}.tif'.format(test_img_name)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    I_x = I_y = 256
    d_x = d_y = 32
    dim_x = d_x*d_y
    batch_size = (I_x*I_y)//(dim_x)
    n_measure = 0.1
    lr_factor = 1.0

    n_img_plot = int(np.sqrt(batch_size))
    dim_z = 100
    dim_phi = int(n_measure*dim_x)
    nIter = 151

    x_test = Image.open(fname).convert(mode='L').resize((I_x,I_y))

    x_test_ = np.array(x_test)/255.
    x_test = []

    for i in range(n_img_plot):
        for j in range(n_img_plot):
            _x = x_test_[i*d_x:d_x*(i+1),j*d_y:d_y*(j+1)]
            x_test.append(_x)

    x_test = np.array(x_test)
    x_test = np.expand_dims(x_test,3)
    test_images = x_test[:batch_size,:,:,:]

    grid_imsave(test_images,[n_img_plot,n_img_plot],'{}/gt_sample.png'.format(savedir))

    tf.reset_default_graph()
    tf.set_random_seed(0)
    np.random.seed(4321)

    Y_obs_ph = tf.placeholder(tf.float32,[batch_size,dim_phi])
    phi_ph = tf.placeholder(tf.float32,[dim_x,dim_phi])
    lr = tf.placeholder(tf.float32)

    tmp = tf.random_uniform([100,dim_z],minval=-1.0,maxval=1.0)
    tmp = tf.expand_dims(tf.reduce_mean(tmp,axis=0),axis=0)
    z_ = tf.tile(tmp,[batch_size,1])
    z_prior_ = tf.Variable(z_,name="z_prior")

    G_sample_ = 0.5*generator(z_prior_,False)+0.5

    G_sample = tf.image.resize_images(G_sample_,[d_x,d_y],tf.image.ResizeMethod.BICUBIC)
    G_sample_re = tf.reshape(G_sample,[-1,dim_x])

    phi_est = phi_ph

    proj_corrected = projector_tf(G_sample_re,phi_est)


    G_loss = tf.reduce_mean(tf.square(proj_corrected-Y_obs_ph))
    loss = tf.reduce_mean(tf.abs(proj_corrected-Y_obs_ph))
    opt_loss = G_loss


    t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    g_vars = [var for var in t_vars if 'Generator' in var.name]


    solution_opt = tf.train.RMSPropOptimizer(lr).minimize(opt_loss, var_list=[z_prior_])

    saver = tf.train.Saver(g_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(modelsave)


        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("************ Generator weights restored! **************")

        nb = batch_size
        z_test = sample_Z(100,dim_z)
        phi_np = np.random.randn(dim_x,dim_phi)
        phi_test_np = phi_np

        y_obs = np.matmul(test_images.reshape(-1,dim_x),phi_test_np)
        lr_start = 1e-3

        for i in range(nIter):
            lr_new = lr_start*0.99**(1.*i/nIter)
            if i %10 ==0:
                G_imgs,tr_loss = sess.run([G_sample,G_loss],feed_dict={phi_ph:phi_np,Y_obs_ph:y_obs})
                merged = merge(G_imgs,[n_img_plot,n_img_plot])
                psnr0 = compare_psnr(x_test_,merged,data_range=1.0)

                if USE_BM3D:
                    merged_clean = pybm3d.bm3d.bm3d(merged,0.25)
                    psnr1 = compare_psnr(x_test_,merged_clean,data_range=1.0)
                    merged_clean = np.array(merged_clean*255,dtype=np.uint8)
                    print('iter: {:d}, tr loss: {:.4f}, PSNR-raw: {:.4f}, PSNR-bm3d: {:.4f}'.format(i,tr_loss,psnr0,psnr1))
                    imsave('{}/inv_bm3d_solution_{}.png'.format(savedir,str(i).zfill(3)),merged_clean)

                else:
                    merged = np.array(merged*255,dtype=np.uint8)
                    print('iter: {:d}, tr loss: {:.4f}, PSNR-raw: {:.4f}'.format(i,tr_loss,psnr0))
                    imsave('{}/inv_solution_{}.png'.format(savedir,str(i).zfill(3)),merged)

            fd = {phi_ph:phi_np,Y_obs_ph:y_obs,lr:lr_new}

            for j in range(10):
                _,tr_loss = sess.run([solution_opt,G_loss],feed_dict=fd)

if __name__ == '__main__':
    # test_images = ['barbara', 'Parrots','lena256','foreman','cameraman','house','Monarch']
    test_img_name = 'Parrots'
    USE_BM3D = True
    savedir = './outs_tf'
    GPP_solve(test_img_name, USE_BM3D, savedir)
