# coding=utf-8
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

import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import pybm3d

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



dim_x = 256
batch_size = 256
n_measure = 0.1

n_img_plot = int(np.sqrt(batch_size))
dim_z = 100
dim_phi = int(n_measure*dim_x)
nIter = 501
d_x = d_y = 16
iters = np.array(np.geomspace(10,10,nIter),dtype=int)

modelsave = './gen_models_imagenet'
# modelsave ='../CIFAR/gen_models_cifar'
# modelsave = './backup_genmodels'
fname = '/p/lustre1/anirudh1/GAN/mimicGAN/IMAGENET/test_images/Parrots.tif'
# fname = './building.jpg'
x_test = Image.open(fname).convert(mode='L').resize((256,256))

x_test_ = np.array(x_test)/255.
# x_test_ = 2*x_test_-1
x_test = []
for i in range(16):
    for j in range(16):
        _x = x_test_[i*d_x:d_x*(i+1),j*d_y:d_y*(j+1)]
        x_test.append(_x)

x_test = np.array(x_test)
x_test = np.expand_dims(x_test,3)

test_images = x_test[:batch_size,:,:,:]

imsave(test_images,[n_img_plot,n_img_plot],'cs_outs/gt_sample.png')

def mimic_correction_v2(phi_old,y_obs,G_curr,n_batch=batch_size):
    a_list = []
    b_list = []
    for i in range(n_batch):
        phi_block = block_diagonal(1*[phi_old])
        y_block = tf.reshape(y_obs[i,:],[1,1*dim_phi])
        G_block = tf.reshape(G_curr[i,:],[1,1*dim_x])
        I = tf.ones_like(phi_old)
        I_block = block_diagonal(1*[I])
        y_m = tf.matmul(G_block,I_block)

        y_hat = tf.matmul(G_block,phi_block)
        theta_1 = tf.squeeze(tf.matmul(y_hat,tf.transpose(y_hat)))
        theta_2 = tf.squeeze(tf.matmul(y_hat,tf.transpose(y_m)))
        C0 = tf.matmul(y_block,tf.transpose(y_hat))

        theta_4 = tf.matmul(y_m,tf.transpose(y_m))
        C1 = tf.matmul(y_block,tf.transpose(y_m))

        a_est = tf.squeeze((theta_4*C0-C1*theta_2)/(theta_1*theta_4 - theta_2*theta_2))
        b_est = tf.squeeze((C1 - a_est*theta_2)/theta_4)
        a_list.append(a_est)
        b_list.append(b_est)
    a_approx = tf.reduce_mean(a_list)
    b_approx = tf.reduce_mean(b_list)
    # a_est = tf.squeeze((C1-theta_4*b_est)/(theta_2))
    return a_approx,b_approx

tf.reset_default_graph()
tf.set_random_seed(0)
np.random.seed(4321)

Y_obs_ph = tf.placeholder(tf.float32,[batch_size,dim_phi])
phi_ph = tf.placeholder(tf.float32,[dim_x,dim_phi])

tmp = tf.random_uniform([1000,dim_z],minval=-1.,maxval=1.)
tmp = tf.expand_dims(tf.reduce_mean(tmp,axis=0),axis=0)

z_ = tf.tile(tmp,[batch_size,1])
z_prior_ = tf.Variable(z_,name="z_prior")

G_sample_ = 0.5*generator(z_prior_,False)+0.5
G_sample = G_sample_[:,::2,::2]
G_sample_re = tf.reshape(G_sample,[-1,dim_x])

'''************ Self-correction step ************'''
a_est,b_est = mimic_correction_v2(phi_ph,Y_obs_ph,G_sample_re)

# phi_est = a_est*phi_ph+b_est
phi_est = phi_ph

proj_corrected = projector_tf(G_sample_re,phi_est)

G_loss = tf.reduce_mean(tf.square(proj_corrected-Y_obs_ph))
loss = tf.reduce_mean(tf.square(proj_corrected-Y_obs_ph))
# z_clip = tf.clip_by_value(z_prior_,-1,1)

t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
g_vars = [var for var in t_vars if 'Generator' in var.name]


solution_opt = tf.train.RMSPropOptimizer(5e-2).minimize(loss, var_list=[z_prior_])


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
    phi_test_np = 1.0*phi_np - 0.

    print(np.mean(x_test),np.min(x_test),np.max(x_test))
    y_obs = np.matmul(test_images.reshape(-1,dim_x),phi_test_np)

    for i in xrange(nIter):

        if i %10 ==0:
            G_imgs,tr_loss,a_estimate,b_estimate = sess.run([G_sample,G_loss,a_est,b_est],feed_dict={phi_ph:phi_np,Y_obs_ph:y_obs})
            # print(np.mean(G_imgs),np.min(G_imgs),np.max(G_imgs))
            # print('iter: {:d}, tr loss: {:.4f}'.format(i,tr_loss))
            print('iter: {:d}, tr loss: {:.4f}, a*: {:.3f}, b* {:.3f}'.format(i,tr_loss,a_estimate,b_estimate))
            # imsave(G_imgs,[n_img_plot,n_img_plot],'cs_outs/inv_solution_{}.png'.format(str(i).zfill(3)))
            merged = merge(G_imgs,[n_img_plot,n_img_plot])
            merged_clean = pybm3d.bm3d.bm3d(merged, 0.05)
            scipy.misc.imsave('cs_outs/inv_solution_{}.png'.format(str(i).zfill(3)),merged)
            scipy.misc.imsave('cs_outs/inv_bm3d_solution_{}.png'.format(str(i).zfill(3)),merged_clean)


        fd = {phi_ph:phi_np,Y_obs_ph:y_obs}
        for j in range(iters[i]):
            _,tr_loss = sess.run([solution_opt,G_loss],feed_dict=fd)
