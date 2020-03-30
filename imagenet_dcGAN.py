# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from model import *
from dataReader import dataReader



g_modelsave = './gen_models_imagenet'
d_modelsave = './disc_models_imagenet'

imglist = []

with open("filenames.txt", "r") as f:
    for item in f:
        imglist.append(item[:-1])

imagenet = dataReader(imglist=imglist,img_dim=[32,32],batch_size=100,sub=-1)
imagenet.batch_idx = 0

x_batch = imagenet.get_train_next_batch_par()
print(x_batch.shape)
imsave(x_batch[:,:,:,np.newaxis],[10,10],'outs/gt_sample.png')

def sample_Z(m, n):
    return np.random.uniform(-1,1,size=[m, n])

batch_size = 100
dim_z = 100
nIter = 2


imgs = tf.placeholder(tf.float32,[None,32,32,1])
# imgs_gray = tf.image.rgb_to_grayscale(imgs)
z = tf.placeholder(tf.float32, [None, dim_z], name='z')
train_mode = tf.placeholder(tf.bool,name='train_mode')

G_sample = generator(z,train_mode)
D_logit_real, D_real = discriminator(imgs,train_mode)
D_logit_fake, D_fake = discriminator(G_sample,train_mode,reuse=True)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
d_vars = [var for var in t_vars if 'Discriminator' in var.name]
g_vars = [var for var in t_vars if 'Generator' in var.name]


G_loss_reg = G_loss
D_loss_reg = D_loss

#
d_opt = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(D_loss, var_list=d_vars)
g_opt = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(G_loss, var_list=g_vars)

assert len(list(set(t_vars)-set(d_vars+g_vars)))==0
g_saver = tf.train.Saver(g_vars)
d_saver = tf.train.Saver(d_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    d_ckpt = tf.train.get_checkpoint_state(d_modelsave)
    g_ckpt = tf.train.get_checkpoint_state(g_modelsave)

    if g_ckpt and g_ckpt.model_checkpoint_path:
        g_saver.restore(sess, g_ckpt.model_checkpoint_path)
        print("************ Gen restored! **************")

    if d_ckpt and d_ckpt.model_checkpoint_path:
        d_saver.restore(sess, d_ckpt.model_checkpoint_path)
        print("************ Disc restored! **************")

    z_test_sample = sample_Z(100,100)
    for epoch in range(nIter):
        imagenet.batch_idx = 0

        while(imagenet.batch_idx is not -1):
            K = imagenet.batch_idx
            x_batch = np.expand_dims(imagenet.get_train_next_batch_par(),axis=3)
            batch_z = sample_Z(x_batch.shape[0],100)

            _,dloss=sess.run([d_opt,D_loss_reg], feed_dict={imgs:x_batch,z:batch_z,train_mode:True})
            # _,dloss=sess.run([d_opt,D_loss_reg], feed_dict={imgs:x_batch,z:batch_z,train_mode:True})
            _,gloss=sess.run([g_opt,G_loss_reg], feed_dict={imgs:x_batch,z:batch_z,train_mode:True})
            _,gloss=sess.run([g_opt,G_loss_reg], feed_dict={imgs:x_batch,z:batch_z,train_mode:True})


            if K % 100 == 0:
                print('iter #',str(epoch),', batch #',K,'g loss:',gloss,'d loss',dloss)


            if K % 100 == 0:
                samples = sess.run(G_sample,feed_dict={imgs:x_batch,z:z_test_sample,train_mode:False})
                imsave((samples+1.)/2.,[10,10],'outs/{}_{}.png'.format(str(epoch).zfill(1),str(K).zfill(4)))


            if K % 1000 == 0:
                save_path = g_saver.save(sess, g_modelsave+"/model_"+str(epoch)+"_batch_"+str(K)+".ckpt")
                save_path = d_saver.save(sess, d_modelsave+"/model_"+str(epoch)+"_batch_"+str(K)+".ckpt")
