# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from model import *
from dataReader import dataReader



modelsave = './gen_models_imagenet'


imglist = []

with open("filenames.txt", "r") as f:
    for item in f:
        imglist.append(item[:-1])

imagenet = dataReader(imglist=imglist,img_dim=[32,32],batch_size=100,sub=-1)
imagenet.batch_idx = 0

x_batch = imagenet.get_train_next_batch_par()
print(x_batch.shape)


def sample_Z(m, n):
    return np.random.uniform(-1,1,size=[m, n])

batch_size = 100
dim_z = 100
nIter = 50

test_imgs = np.expand_dims(imagenet.get_random_test_batch(nb=100),axis=3)
imsave((test_imgs+1.)/2.,[10,10],'outs/gt_sample.png')


imgs = tf.placeholder(tf.float32,[None,32,32,1])
train_mode = tf.placeholder(tf.bool,name='train_mode')
# imgs_gray = tf.image.rgb_to_grayscale(imgs)

z_latent = encoder(imgs,dim_z,train_mode)
G_sample = generator(z_latent,train_mode)

loss = tf.reduce_mean(tf.square(G_sample-imgs))

t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
e_vars = [var for var in t_vars if 'Encoder' in var.name]
g_vars = [var for var in t_vars if 'Generator' in var.name]


g_opt = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(loss, var_list=g_vars+e_vars)

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(modelsave)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("************ Model restored! **************")

    for epoch in range(nIter):

        imagenet.batch_idx = 0

        while(imagenet.batch_idx is not -1):
            K = imagenet.batch_idx
            x_batch = np.expand_dims(imagenet.get_train_next_batch_par(),axis=3)

            _,gloss=sess.run([g_opt,loss], feed_dict={imgs:x_batch,train_mode:True})


            if K % 100 == 0:
                print('iter #',str(epoch),', batch #',K,'recon loss:',gloss)


            if K % 1000 == 0:
                samples = sess.run(G_sample,feed_dict={imgs:test_imgs,train_mode:False})
                imsave(samples,[10,10],'outs/{}_{}.png'.format(str(epoch).zfill(1),str(K).zfill(4)))
                save_path = saver.save(sess, modelsave+"/model_"+str(epoch)+"_batch_"+str(K)+".ckpt")
