#author: Rushil Anirudh

import sys
from os import listdir, walk
from os.path import isfile, join, isdir
import os
import math
import csv
import random
import numpy as np
import scipy
from scipy.misc import *
from scipy import ndimage as ndi
from sklearn import utils
import pickle as pkl
from random import shuffle
from sklearn.preprocessing import scale, MinMaxScaler
import time
from PIL import Image
import sys
sys.stdout.flush()
from multiprocessing import Process
from multiprocessing.queues import Queue
import multiprocessing
import copy
import glob

def read_file((filename,img_size)):
    filename = '/usr/workspace/anirudh1/imagenet-32x32/'+filename
    img = Image.open(filename).convert(mode='L')
    img = np.array(img,dtype=np.float32)/127.5 - 1 
    return img

class dataReader:

    def __init__(self,datapath = './',img_dim=[64,64],
                     batch_size=64,test_size=0.1,sub=-1,imglist=None,fmt="*.png"):

        self.datapath = datapath
        self.x_dim = img_dim
        self.n_batch = batch_size
        self.batch_idx = 0
        self.test_size = test_size
        if imglist is None:
            self.imglist = glob.glob(join(self.datapath,fmt))
        else:
            self.imglist = imglist


        self.num_samples = len(self.imglist)
        if sub==-1:
            sub = self.num_samples
        else:
            self.num_samples = sub

        N_train = int(self.num_samples * (1-self.test_size))

        self.n_train = N_train
        self.n_test = self.num_samples - self.n_train


        self.data_train_list = self.imglist[:self.n_train]
        self.data_test_list  = self.imglist[self.n_train:]


        print('Number of samples:',self.num_samples)


    def __read_file(self,filename):
        filename = '/usr/workspace/anirudh1/imagenet-32x32/'+filename
        img = Image.open(filename).convert(mode='L')
        img = np.array(img,dtype=np.float32)/127.5 - 1
        return img

    def get_train_next_batch_par(self):
        if self.batch_idx == -1:
            return None, None

        X_train = []
        idx = self.batch_idx
        nb = self.n_batch

        if (idx+1)*nb >= self.n_train:
            self.batch_idx = -1 #end of dataset
            train_img_batch = self.data_train_list[idx*nb:]

        else:
            train_img_batch = self.data_train_list[idx*nb:(idx+1)*nb]
            self.batch_idx += 1 #increment batch index

        pool = multiprocessing.Pool(8)
        job_xargs = [(t1, self.x_dim) for t1 in train_img_batch]

        # X_train = pool.map(read_file,job_xargs)
        X_train = pool.map_async(read_file,job_xargs).get(9999999)

        pool.close()

        X_out = np.array(X_train,dtype=np.float32)

        return X_out

    def get_random_test_batch(self,nb=None):
        X_test = []
        if nb is None:
            nb = self.n_batch

        idx = np.random.choice(len(self.data_test_list),size=(nb),replace=False)
        data_batch = [self.data_test_list[k] for k in idx]

        for t1 in data_batch:
            x = self.__read_file(t1)
            X_test.append(x[:,:])

        return np.array(X_test,dtype=np.float32)


if __name__=='__main__':
    datapath = sys.argv[1]
    ctr = dataReader(datapath=datapath,img_dim=[32,32],batch_size=100,sub=-1)

    print('Number of samples:',ctr.num_samples)
    ctr.batch_idx = 0
    X_train = ctr.get_train_next_batch_par()

    while(ctr.batch_idx is not -1):
        time0 = time.time()
        X_train = ctr.get_train_next_batch_par()
        # print(np.max(X_train),np.min(X_train),np.mean(X_train),np.std(X_train))
        print(X_train.shape,ctr.batch_idx,'time: ',time.time()-time0)


    ####
