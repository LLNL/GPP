# Copyright 2020 Lawrence Livermore National Security, LLC and other authors: Rushil Anirudh, Suhas Lohit, Pavan Turaga
# SPDX-License-Identifier: MIT
# coding=utf-8
import numpy as np
import tensorflow as tf

from model import generator_c

from skimage.measure import compare_psnr
from PIL import Image
import pybm3d
import os


from skimage.transform import rescale, resize
from skimage import color
from skimage import io
import scipy

'''
*** WARNING: This code is experimental ***

'''
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

def GPP_color(test_image):

    # I_x = I_y = 1024
    I_y = 1024
    # I_x = 1536
    I_x = 768
    d_x = d_y = 32
    dim_x = d_x*d_y
    batch_size = (I_x*I_y)//(dim_x)
    n_measure = 0.01
    lr_factor = 1.0#*batch_size//64
    dim_z = 100
    dim_phi = int(n_measure*dim_x)
    nIter = 201
    n_img_plot_x = I_x//d_x
    n_img_plot_y = I_y//d_y

    iters = np.array(np.geomspace(10,10,nIter),dtype=int)

    modelsave = './gan_models/gen_models_corrupt-colorcifar32'
    fname = './test_images/{}.jpg'.format(test_image)

    image = io.imread(fname)
    x_test = resize(image, (I_x, I_y),anti_aliasing=True,preserve_range=True,mode='reflect')
    x_test_ = np.array(x_test)/np.max(x_test)


    x_test = []
    for i in range(n_img_plot_x):
        for j in range(n_img_plot_y):
            _x = x_test_[i*d_x:d_x*(i+1),j*d_y:d_y*(j+1)]
            x_test.append(_x)

    x_test = np.array(x_test)
    test_images = x_test[:batch_size,:,:,:]

    imsave(test_images,[n_img_plot_x,n_img_plot_y],'cs_outs/gt_sample.png')

    tf.reset_default_graph()
    tf.set_random_seed(0)
    np.random.seed(4321)

    Y_obs_ph = tf.placeholder(tf.float32,[batch_size,dim_phi,3])
    phi_ph = tf.placeholder(tf.float32,[dim_x,dim_phi])
    lr = tf.placeholder(tf.float32)

    tmp = 0.*tf.random_uniform([batch_size,dim_z],minval=-1.0,maxval=1.0)

    z_prior_ = tf.Variable(tmp,name="z_prior")

    G_sample_ = 0.5*generator_c(z_prior_,False,dim_z=dim_z)+0.5
    G_sample = tf.image.resize_images(G_sample_,[d_x,d_y])

    G_sample_re_r = tf.reshape(G_sample[:,:,:,0],[-1,dim_x])
    G_sample_re_g = tf.reshape(G_sample[:,:,:,1],[-1,dim_x])
    G_sample_re_b = tf.reshape(G_sample[:,:,:,2],[-1,dim_x])

    phi_est = phi_ph

    proj_corrected_r = projector_tf(G_sample_re_r,phi_est)
    proj_corrected_g = projector_tf(G_sample_re_g,phi_est)
    proj_corrected_b = projector_tf(G_sample_re_b,phi_est)

    G_loss = 0
    G_loss += tf.reduce_mean(tf.square(proj_corrected_r-Y_obs_ph[:,:,0]))
    G_loss += tf.reduce_mean(tf.square(proj_corrected_g-Y_obs_ph[:,:,1]))
    G_loss += tf.reduce_mean(tf.square(proj_corrected_b-Y_obs_ph[:,:,2]))
    opt_loss = G_loss


    t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    g_vars = [var for var in t_vars if 'Generator' in var.name]


    solution_opt = tf.train.RMSPropOptimizer(lr).minimize(opt_loss, var_list=[z_prior_])

    saver = tf.train.Saver(g_vars)
    merged_imgs = []
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

        print(np.mean(x_test),np.min(x_test),np.max(x_test))
        y_obs = np.zeros((batch_size,dim_phi,3))
        for i in range(3):
            y_obs[:,:,i] = np.matmul(test_images[:,:,:,i].reshape(-1,dim_x),phi_test_np)
        # lr_start = 2e-3
        lr_start = 5e-3

        for i in xrange(nIter):
            lr_new = lr_start*0.99**(1.*i/nIter)
            if i %5 ==0:
                G_imgs,tr_loss = sess.run([G_sample,G_loss],feed_dict={phi_ph:phi_np,Y_obs_ph:y_obs})
                merged = merge(G_imgs,[n_img_plot_x,n_img_plot_y])
                scipy.misc.imsave('cs_outs/inv_solution_{}.png'.format(str(i).zfill(3)),merged)

                psnr0 = compare_psnr(x_test_,merged,data_range=1.0)

                print('iter: {:d}, tr loss: {:.4f}, PSNR-raw: {:.4f}, PSNR-bm3d: {:.4f}'.format(i,tr_loss,psnr0,psnr0))

                merged_imgs.append(merged)

            fd = {phi_ph:phi_np,Y_obs_ph:y_obs,lr:lr_new}
            for j in range(iters[i]):

                _,tr_loss = sess.run([solution_opt,G_loss],feed_dict=fd)

    return merged_imgs

if __name__ == '__main__':
    test_image = 'color_turtle'

    img_c = GPP_color(test_image)
