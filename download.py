#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:31:59 2018
@author: chinwei
"""

import urllib
import _pickle as pickle
import gzip
import os
import numpy as np
import urllib.request



if __name__ == '__main__':

    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist'],
                        help='dataset name')
    parser.add_argument('--savedir', type=str, default='datasets', 
                        help='directory to save the dataset')
    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        
    if args.dataset == 'mnist':
        path = 'http://deeplearning.net/data/mnist'
        mnist_filename_all = 'mnist.pkl'
        local_filename = os.path.join(args.savedir, mnist_filename_all)
        urllib.request.urlretrieve(
            "{}/{}.gz".format(path,mnist_filename_all), local_filename+'.gz')
        print('filename:', local_filename)
        tr,va,te = pickle.load(gzip.open(local_filename+'.gz', 'rb'), encoding='latin1')
        o = open(local_filename+'.npy','wb')
        np.save(o, (tr,va,te))

        
