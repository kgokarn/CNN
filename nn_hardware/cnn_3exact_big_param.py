#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

cnn_data = np.load('cnn_1train_weights_aftest.npz')

np.set_printoptions(precision=8)
np.set_printoptions(threshold=np.inf)

correct_predictions = 0


def print_max_err(i, d, id):
    if i == 0:
        exp = cnn_data[id][0]
        print(f'{np.max(np.abs(exp - d)):.3f}')


def pwl_activation(acti):
    condition1 = np.less(acti, -2)
    condition2 = np.multiply(np.greater_equal(acti, -2), np.less(acti, -0.6875))
    condition3 = np.multiply(np.greater_equal(acti, -0.6875), np.less(acti, 0.6875))
    condition4 = np.multiply(np.greater_equal(acti, 0.6875), np.less(acti, 2))
    condition5 = np.greater_equal(acti, 2)

    a = np.multiply(condition1, -1)
    b = np.multiply(condition2, np.multiply((acti - 2), 0.25))
    c = np.multiply(condition3, acti)
    d = np.multiply(condition4, np.multiply((acti + 2), 0.25))
    e = np.multiply(condition5, 1)

    return a + b + c + d + e


first_conv_max = 0
second_conv_max = 0
first_conv_avg = 0
second_conv_avg = 0

first_layer_kernels = 20
second_layer_kernels = 60
third_layer_kernels = 60
fourth_layer_kernels = 120
# assume square kernels
first_layer_kernel_size = 4
second_layer_kernel_size = 3
third_layer_kernel_size = 3
fourth_layer_kernel_size = 3

# assume 64bit fix point
bytesize = 8

# for each input in the test set
for i, img in enumerate(cnn_data['x_test']):

    # layer0: conv2d
    weights = cnn_data['l0weights']
    f_shape = weights[::, ::, 0].shape
    shape = f_shape + tuple(np.subtract(img.shape, f_shape) + 1)
    subs = np.lib.stride_tricks.as_strided(img, shape, img.strides * 2)
    layer0 = np.einsum('ijm,ijkl->klm', weights, subs)
    #print_max_err(i, layer0, 'test0l0')

    # layer1: activation function
    layer1 = np.tanh(0.75*layer0)
    #layer1 = pwl_activation(layer0)
    

    # layer2: max pooling 2d
    layer2 = layer1.reshape(int(layer1.shape[0] / 2), 2, int(layer1.shape[1] / 2), 2, -1).max(axis=(1, 3))
    #print_max_err(i, layer2, 'test0l2')
    


    if i == 0:
        print('saving data')
        np.save('exact_fm.npy', layer2)

    # layer3: conv2d
    weights = cnn_data['l3weights']
    shape = (first_layer_kernels, second_layer_kernel_size, second_layer_kernel_size, layer2.shape[0] - second_layer_kernel_size + 1,
             layer2.shape[1] - second_layer_kernel_size + 1)
    # stride values: bytesize = size of one data value
    stride = (bytesize, first_layer_kernels * layer2.shape[0] * bytesize, first_layer_kernels * bytesize, layer2.shape[1] * first_layer_kernels * bytesize,
              first_layer_kernels * bytesize)
    subs = np.lib.stride_tricks.as_strided(layer2, shape, stride)
    layer3 = np.einsum('ijhm,hijkl->klm', weights, subs) + [[cnn_data['l3biases']]]
    #print_max_err(i, layer3, 'test0l3')

    # layer4: activation function
    layer4 = np.maximum(layer3, 0)
    

    # layer5: conv2d
    weights = cnn_data['l5weights']
    shape = (second_layer_kernels, third_layer_kernel_size, third_layer_kernel_size, layer4.shape[0] - third_layer_kernel_size + 1,
             layer4.shape[1] - third_layer_kernel_size + 1)
    # stride values: bytesize = size of one data value
    stride = (bytesize, second_layer_kernels * layer4.shape[0] * bytesize, second_layer_kernels * bytesize, layer4.shape[1] * second_layer_kernels * bytesize,
              second_layer_kernels * bytesize)
    subs = np.lib.stride_tricks.as_strided(layer4, shape, stride)
    layer5 = np.einsum('ijhm,hijkl->klm', weights, subs) + [[cnn_data['l5biases']]]
    #print_max_err(i, layer5, 'test0l5')

    # layer6: activation function
    layer6 = np.maximum(layer5, 0)
    #print_max_err(i, layer6, 'test0l6')
    

    # layer7: max pooling with 'same' padding
    layer6_padded = np.zeros(tuple([sum(x) for x in zip(layer6.shape, (1, 1, 0))]))
    layer6_padded[:9, :9, :] = layer6
    layer6_padded[9, :9, :] = layer6[8, :, :]
    layer6_padded[:9, 9, :] = layer6[:, 8, :]
    layer7 = layer6_padded.reshape(int(layer6_padded.shape[0] / 2), 2, int(layer6_padded.shape[1] / 2), 2, -1).max(axis=(1, 3))
    #print_max_err(i, layer7, 'test0l7')
    
    # layer8: conv2d
    weights = cnn_data['l8weights']
    shape = (third_layer_kernels, fourth_layer_kernel_size, fourth_layer_kernel_size, layer7.shape[0] - fourth_layer_kernel_size + 1,
             layer7.shape[1] - fourth_layer_kernel_size + 1)
    # stride values: bytesize = size of one data value
    stride = (bytesize, third_layer_kernels * layer7.shape[0] * bytesize, third_layer_kernels * bytesize, layer7.shape[1] * third_layer_kernels * bytesize,
              third_layer_kernels * bytesize)
    subs = np.lib.stride_tricks.as_strided(layer7, shape, stride)
    layer8 = np.einsum('ijhm,hijkl->klm', weights, subs) + [[cnn_data['l8biases']]]
    #print_max_err(i, layer8, 'test0l8')

    # layer9: activation function
    layer9 = np.maximum(layer8, 0)
    

    # layer10: flatten
    layer10 = layer9.flatten()
    

    # layer11: dense
    layer11 = np.dot(layer10, cnn_data['l11weights']) + cnn_data['l11biases']
    #print_max_err(i, layer11, 'test0l11')
    

    # layer12: dense
    layer12 = np.dot(layer11, cnn_data['l12weights']) + cnn_data['l12biases']
    #print_max_err(i, layer12, 'test0l12')

    # layer13: softmax
    exp = np.exp(layer12)
    layer13 = exp / np.sum(exp)
    
    print("C1 shape is", layer13.shape)

    if (np.argmax(layer12) == np.argmax(cnn_data['y_test'][i])):
        correct_predictions += 1

    print(f'Test {i + 1} of 500. Accuracy: {correct_predictions / (i + 1):.3f}')

