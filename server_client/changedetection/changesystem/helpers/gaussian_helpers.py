import numpy as np
import os
import cv2
import torch
import sys
env_path = os.path.dirname(__file__)
if env_path not in sys.path:
    sys.path.append(env_path)


class mylist:

    def __init__(self, l):
        self.l=l

    def __repr__(self): 
        return repr(self.l)

    def append(self, x):
        self.l.append(x)

def gkern(l, sig):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel) * 5*10**4

def softmax(vector):
	e = np.exp(vector.l)
	return e / e.sum()

def get_weights(image_shape, kernel_size, gaussian, stride):

    weights = np.array([np.array([mylist([]) for _ in range(image_shape[1])]) for _ in range(image_shape[0])])

    flag0 = (weights.shape[0] - kernel_size) % stride != 0
    flag1 = (weights.shape[1] - kernel_size) % stride != 0

    i = 0
    while i+kernel_size <= weights.shape[0]:
        j = 0
        while j+kernel_size <= weights.shape[1]:
            for p in range(kernel_size):
                for q in range(kernel_size):
                    weights[i+p, j+q].append(gaussian[p, q])
            j += stride
        #---------
        if flag1:
            for p in range(kernel_size):
                for q in range(kernel_size):
                    weights[i+p, weights.shape[1]+q-kernel_size].append(gaussian[p, q])
        
        i += stride

    if flag0:
        j = 0
        while j+kernel_size <= weights.shape[1]:
            for p in range(kernel_size):
                for q in range(kernel_size):
                    weights[weights.shape[0]+p-kernel_size, j+q].append(gaussian[p, q])
            j += stride
        #---------
        if flag1:
            for p in range(kernel_size):
                for q in range(kernel_size):
                    weights[weights.shape[0]+p-kernel_size, weights.shape[1]+q-kernel_size].append(gaussian[p, q])
        #---------
        

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if weights[i, j]:
                weights[i, j] = list(softmax(weights[i, j]))
    return weights

def make_generator(weight_col):
    def generator():
        i = 0
        length = len(weight_col)
        while True:
            yield weight_col[i]
            i  = (i+1) % length
    return generator()

def make_generators(weights):
    result = np.empty(weights.shape, dtype=object)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = make_generator(weights[i, j])
    return result

# ----------------------------------------------

def set_generators(window_size, stride, sigma, image_shape):

        gaussian = gkern(window_size, sigma)
        weights = get_weights(image_shape, window_size, gaussian, stride)
        return make_generators(weights)


def apply_window_prediction(result, predict_window, generators, i, j, height, width):
        # height, width = predict_window.shape[2], predict_window.shape[3]
        for p in range(predict_window.shape[2]):
            for q in range(predict_window.shape[3]):
                weight = torch.tensor(next(generators[i+p, j+q]))
                result[:, :, i+p, j+q] += predict_window[:, :, p, q] * weight
