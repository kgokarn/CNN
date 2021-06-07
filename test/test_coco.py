#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import time
from fxpmath import Fxp
from rig.type_casts import fix_to_float

pi_fxp = Fxp(None, signed=True, n_word=12, n_frac=8)    
pi_fxp.rounding = 'around'   


def fix_pt(val):                          
    return int(pi_fxp(val).bin())


def fix2float(val):
    f2f= fix_to_float(True,12, 10)
    return f2f(val)


cnn_data = np.load('cnn_1train_weights_aftest.npz')
new_x = np.zeros((5,5))
layer2 = np.zeros((13,13,20))


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
    iter=0
    count = 0

    while(iter<25):
        ii=0
        count2 = 0
        while(ii<25):
            new_x = img[ii:ii+5,iter:iter+5]
            # loop = asyncio.new_event_loop()
            # asyncio.set_event_loop(loop)
            # layer2 = loop.run_until_complete(tstbench.test_cnn_hw(new_x,weights,layer2,iter,ii,count,count2))
            



            
            
            
            @cocotb.test()
            async def test_cnn_hw(dut, ii=ii, iter=iter, count=count, count2=count2):
                clock = Clock(dut.clk, 10, units="ns")  # Create a 10ns period clock on port clk
                cocotb.fork(clock.start())  # Start the clock

                reset = 0
                
                print("iter before is",iter)
                print("ii before is",ii)


                await RisingEdge(dut.clk)
                dut.reset <= reset
                dut.i_activation_0_0 <= fix_pt(new_x[0, 0])
                dut.i_activation_0_1 <= fix_pt(new_x[0, 1])
                dut.i_activation_0_2 <= fix_pt(new_x[0, 2])
                dut.i_activation_0_3 <= fix_pt(new_x[0, 3])
                dut.i_activation_0_4 <= fix_pt(new_x[0, 4])
                dut.i_activation_1_0 <= fix_pt(new_x[1, 0])
                dut.i_activation_1_1 <= fix_pt(new_x[1, 1])
                dut.i_activation_1_2 <= fix_pt(new_x[1, 2])
                dut.i_activation_1_3 <= fix_pt(new_x[1, 3])
                dut.i_activation_1_4 <= fix_pt(new_x[1, 4])
                dut.i_activation_2_0 <= fix_pt(new_x[2, 0])
                dut.i_activation_2_1 <= fix_pt(new_x[2, 1])
                dut.i_activation_2_2 <= fix_pt(new_x[2, 2])
                dut.i_activation_2_3 <= fix_pt(new_x[2, 3])
                dut.i_activation_2_4 <= fix_pt(new_x[2, 4])
                dut.i_activation_3_0 <= fix_pt(new_x[3, 0])
                dut.i_activation_3_1 <= fix_pt(new_x[3, 1])
                dut.i_activation_3_2 <= fix_pt(new_x[3, 2])
                dut.i_activation_3_3 <= fix_pt(new_x[3, 3])
                dut.i_activation_3_4 <= fix_pt(new_x[3, 4])
                dut.i_activation_4_0 <= fix_pt(new_x[4, 0])
                dut.i_activation_4_1 <= fix_pt(new_x[4, 1])
                dut.i_activation_4_2 <= fix_pt(new_x[4, 2])
                dut.i_activation_4_3 <= fix_pt(new_x[4, 3])
                dut.i_activation_4_4 <= fix_pt(new_x[4, 4])

                dut.i_weight_0_0_0 <= fix_pt(weights[0, 0, 0])
                dut.i_weight_0_0_1 <= fix_pt(weights[0, 1, 0])
                dut.i_weight_0_0_2 <= fix_pt(weights[0, 2, 0])
                dut.i_weight_0_0_3 <= fix_pt(weights[0, 3, 0])
                dut.i_weight_0_1_0 <= fix_pt(weights[1, 0, 0])
                dut.i_weight_0_1_1 <= fix_pt(weights[1, 1, 0])
                dut.i_weight_0_1_2 <= fix_pt(weights[1, 2, 0])
                dut.i_weight_0_1_3 <= fix_pt(weights[1, 3, 0])
                dut.i_weight_0_2_0 <= fix_pt(weights[2, 0, 0])
                dut.i_weight_0_2_1 <= fix_pt(weights[2, 1, 0])
                dut.i_weight_0_2_2 <= fix_pt(weights[2, 2, 0])
                dut.i_weight_0_2_3 <= fix_pt(weights[2, 3, 0])
                dut.i_weight_0_3_0 <= fix_pt(weights[3, 0, 0])
                dut.i_weight_0_3_1 <= fix_pt(weights[3, 1, 0])
                dut.i_weight_0_3_2 <= fix_pt(weights[3, 2, 0])
                dut.i_weight_0_3_3 <= fix_pt(weights[3, 3, 0])

                dut.i_weight_1_0_0 <= fix_pt(weights[0, 0, 1])
                dut.i_weight_1_0_1 <= fix_pt(weights[0, 1, 1])
                dut.i_weight_1_0_2 <= fix_pt(weights[0, 2, 1])
                dut.i_weight_1_0_3 <= fix_pt(weights[0, 3, 1])
                dut.i_weight_1_1_0 <= fix_pt(weights[1, 0, 1])
                dut.i_weight_1_1_1 <= fix_pt(weights[1, 1, 1])
                dut.i_weight_1_1_2 <= fix_pt(weights[1, 2, 1])
                dut.i_weight_1_1_3 <= fix_pt(weights[1, 3, 1])
                dut.i_weight_1_2_0 <= fix_pt(weights[2, 0, 1])
                dut.i_weight_1_2_1 <= fix_pt(weights[2, 1, 1])
                dut.i_weight_1_2_2 <= fix_pt(weights[2, 2, 1])
                dut.i_weight_1_2_3 <= fix_pt(weights[2, 3, 1])
                dut.i_weight_1_3_0 <= fix_pt(weights[3, 0, 1])
                dut.i_weight_1_3_1 <= fix_pt(weights[3, 1, 1])
                dut.i_weight_1_3_2 <= fix_pt(weights[3, 2, 1])
                dut.i_weight_1_3_3 <= fix_pt(weights[3, 3, 1])

                dut.i_weight_2_0_0 <= fix_pt(weights[0, 0, 2])
                dut.i_weight_2_0_1 <= fix_pt(weights[0, 1, 2])
                dut.i_weight_2_0_2 <= fix_pt(weights[0, 2, 2])
                dut.i_weight_2_0_3 <= fix_pt(weights[0, 3, 2])
                dut.i_weight_2_1_0 <= fix_pt(weights[1, 0, 2])
                dut.i_weight_2_1_1 <= fix_pt(weights[1, 1, 2])
                dut.i_weight_2_1_2 <= fix_pt(weights[1, 2, 2])
                dut.i_weight_2_1_3 <= fix_pt(weights[1, 3, 2])
                dut.i_weight_2_2_0 <= fix_pt(weights[2, 0, 2])
                dut.i_weight_2_2_1 <= fix_pt(weights[2, 1, 2])
                dut.i_weight_2_2_2 <= fix_pt(weights[2, 2, 2])
                dut.i_weight_2_2_3 <= fix_pt(weights[2, 3, 2])
                dut.i_weight_2_3_0 <= fix_pt(weights[3, 0, 2])
                dut.i_weight_2_3_1 <= fix_pt(weights[3, 1, 2])
                dut.i_weight_2_3_2 <= fix_pt(weights[3, 2, 2])
                dut.i_weight_2_3_3 <= fix_pt(weights[3, 3, 2])

                dut.i_weight_3_0_0 <= fix_pt(weights[0, 0, 3])
                dut.i_weight_3_0_1 <= fix_pt(weights[0, 1, 3])
                dut.i_weight_3_0_2 <= fix_pt(weights[0, 2, 3])
                dut.i_weight_3_0_3 <= fix_pt(weights[0, 3, 3])
                dut.i_weight_3_1_0 <= fix_pt(weights[1, 0, 3])
                dut.i_weight_3_1_1 <= fix_pt(weights[1, 1, 3])
                dut.i_weight_3_1_2 <= fix_pt(weights[1, 2, 3])
                dut.i_weight_3_1_3 <= fix_pt(weights[1, 3, 3])
                dut.i_weight_3_2_0 <= fix_pt(weights[2, 0, 3])
                dut.i_weight_3_2_1 <= fix_pt(weights[2, 1, 3])
                dut.i_weight_3_2_2 <= fix_pt(weights[2, 2, 3])
                dut.i_weight_3_2_3 <= fix_pt(weights[2, 3, 3])
                dut.i_weight_3_3_0 <= fix_pt(weights[3, 0, 3])
                dut.i_weight_3_3_1 <= fix_pt(weights[3, 1, 3])
                dut.i_weight_3_3_2 <= fix_pt(weights[3, 2, 3])
                dut.i_weight_3_3_3 <= fix_pt(weights[3, 3, 3])

                dut.i_weight_4_0_0 <= fix_pt(weights[0, 0, 4])
                dut.i_weight_4_0_1 <= fix_pt(weights[0, 1, 4])
                dut.i_weight_4_0_2 <= fix_pt(weights[0, 2, 4])
                dut.i_weight_4_0_3 <= fix_pt(weights[0, 3, 4])
                dut.i_weight_4_1_0 <= fix_pt(weights[1, 0, 4])
                dut.i_weight_4_1_1 <= fix_pt(weights[1, 1, 4])
                dut.i_weight_4_1_2 <= fix_pt(weights[1, 2, 4])
                dut.i_weight_4_1_3 <= fix_pt(weights[1, 3, 4])
                dut.i_weight_4_2_0 <= fix_pt(weights[2, 0, 4])
                dut.i_weight_4_2_1 <= fix_pt(weights[2, 1, 4])
                dut.i_weight_4_2_2 <= fix_pt(weights[2, 2, 4])
                dut.i_weight_4_2_3 <= fix_pt(weights[2, 3, 4])
                dut.i_weight_4_3_0 <= fix_pt(weights[3, 0, 4])
                dut.i_weight_4_3_1 <= fix_pt(weights[3, 1, 4])
                dut.i_weight_4_3_2 <= fix_pt(weights[3, 2, 4])
                dut.i_weight_4_3_3 <= fix_pt(weights[3, 3, 4])

                dut.i_weight_5_0_0 <= fix_pt(weights[0, 0, 5])
                dut.i_weight_5_0_1 <= fix_pt(weights[0, 1, 5])
                dut.i_weight_5_0_2 <= fix_pt(weights[0, 2, 5])
                dut.i_weight_5_0_3 <= fix_pt(weights[0, 3, 5])
                dut.i_weight_5_1_0 <= fix_pt(weights[1, 0, 5])
                dut.i_weight_5_1_1 <= fix_pt(weights[1, 1, 5])
                dut.i_weight_5_1_2 <= fix_pt(weights[1, 2, 5])
                dut.i_weight_5_1_3 <= fix_pt(weights[1, 3, 5])
                dut.i_weight_5_2_0 <= fix_pt(weights[2, 0, 5])
                dut.i_weight_5_2_1 <= fix_pt(weights[2, 1, 5])
                dut.i_weight_5_2_2 <= fix_pt(weights[2, 2, 5])
                dut.i_weight_5_2_3 <= fix_pt(weights[2, 3, 5])
                dut.i_weight_5_3_0 <= fix_pt(weights[3, 0, 5])
                dut.i_weight_5_3_1 <= fix_pt(weights[3, 1, 5])
                dut.i_weight_5_3_2 <= fix_pt(weights[3, 2, 5])
                dut.i_weight_5_3_3 <= fix_pt(weights[3, 3, 5])

                dut.i_weight_6_0_0 <= fix_pt(weights[0, 0, 6])
                dut.i_weight_6_0_1 <= fix_pt(weights[0, 1, 6])
                dut.i_weight_6_0_2 <= fix_pt(weights[0, 2, 6])
                dut.i_weight_6_0_3 <= fix_pt(weights[0, 3, 6])
                dut.i_weight_6_1_0 <= fix_pt(weights[1, 0, 6])
                dut.i_weight_6_1_1 <= fix_pt(weights[1, 1, 6])
                dut.i_weight_6_1_2 <= fix_pt(weights[1, 2, 6])
                dut.i_weight_6_1_3 <= fix_pt(weights[1, 3, 6])
                dut.i_weight_6_2_0 <= fix_pt(weights[2, 0, 6])
                dut.i_weight_6_2_1 <= fix_pt(weights[2, 1, 6])
                dut.i_weight_6_2_2 <= fix_pt(weights[2, 2, 6])
                dut.i_weight_6_2_3 <= fix_pt(weights[2, 3, 6])
                dut.i_weight_6_3_0 <= fix_pt(weights[3, 0, 6])
                dut.i_weight_6_3_1 <= fix_pt(weights[3, 1, 6])
                dut.i_weight_6_3_2 <= fix_pt(weights[3, 2, 6])
                dut.i_weight_6_3_3 <= fix_pt(weights[3, 3, 6])

                dut.i_weight_7_0_0 <= fix_pt(weights[0, 0, 7])
                dut.i_weight_7_0_1 <= fix_pt(weights[0, 1, 7])
                dut.i_weight_7_0_2 <= fix_pt(weights[0, 2, 7])
                dut.i_weight_7_0_3 <= fix_pt(weights[0, 3, 7])
                dut.i_weight_7_1_0 <= fix_pt(weights[1, 0, 7])
                dut.i_weight_7_1_1 <= fix_pt(weights[1, 1, 7])
                dut.i_weight_7_1_2 <= fix_pt(weights[1, 2, 7])
                dut.i_weight_7_1_3 <= fix_pt(weights[1, 3, 7])
                dut.i_weight_7_2_0 <= fix_pt(weights[2, 0, 7])
                dut.i_weight_7_2_1 <= fix_pt(weights[2, 1, 7])
                dut.i_weight_7_2_2 <= fix_pt(weights[2, 2, 7])
                dut.i_weight_7_2_3 <= fix_pt(weights[2, 3, 7])
                dut.i_weight_7_3_0 <= fix_pt(weights[3, 0, 7])
                dut.i_weight_7_3_1 <= fix_pt(weights[3, 1, 7])
                dut.i_weight_7_3_2 <= fix_pt(weights[3, 2, 7])
                dut.i_weight_7_3_3 <= fix_pt(weights[3, 3, 7])

                dut.i_weight_8_0_0 <= fix_pt(weights[0, 0, 8])
                dut.i_weight_8_0_1 <= fix_pt(weights[0, 1, 8])
                dut.i_weight_8_0_2 <= fix_pt(weights[0, 2, 8])
                dut.i_weight_8_0_3 <= fix_pt(weights[0, 3, 8])
                dut.i_weight_8_1_0 <= fix_pt(weights[1, 0, 8])
                dut.i_weight_8_1_1 <= fix_pt(weights[1, 1, 8])
                dut.i_weight_8_1_2 <= fix_pt(weights[1, 2, 8])
                dut.i_weight_8_1_3 <= fix_pt(weights[1, 3, 8])
                dut.i_weight_8_2_0 <= fix_pt(weights[2, 0, 8])
                dut.i_weight_8_2_1 <= fix_pt(weights[2, 1, 8])
                dut.i_weight_8_2_2 <= fix_pt(weights[2, 2, 8])
                dut.i_weight_8_2_3 <= fix_pt(weights[2, 3, 8])
                dut.i_weight_8_3_0 <= fix_pt(weights[3, 0, 8])
                dut.i_weight_8_3_1 <= fix_pt(weights[3, 1, 8])
                dut.i_weight_8_3_2 <= fix_pt(weights[3, 2, 8])
                dut.i_weight_8_3_3 <= fix_pt(weights[3, 3, 8])

                dut.i_weight_9_0_0 <= fix_pt(weights[0, 0, 9])
                dut.i_weight_9_0_1 <= fix_pt(weights[0, 1, 9])
                dut.i_weight_9_0_2 <= fix_pt(weights[0, 2, 9])
                dut.i_weight_9_0_3 <= fix_pt(weights[0, 3, 9])
                dut.i_weight_9_1_0 <= fix_pt(weights[1, 0, 9])
                dut.i_weight_9_1_1 <= fix_pt(weights[1, 1, 9])
                dut.i_weight_9_1_2 <= fix_pt(weights[1, 2, 9])
                dut.i_weight_9_1_3 <= fix_pt(weights[1, 3, 9])
                dut.i_weight_9_2_0 <= fix_pt(weights[2, 0, 9])
                dut.i_weight_9_2_1 <= fix_pt(weights[2, 1, 9])
                dut.i_weight_9_2_2 <= fix_pt(weights[2, 2, 9])
                dut.i_weight_9_2_3 <= fix_pt(weights[2, 3, 9])
                dut.i_weight_9_3_0 <= fix_pt(weights[3, 0, 9])
                dut.i_weight_9_3_1 <= fix_pt(weights[3, 1, 9])
                dut.i_weight_9_3_2 <= fix_pt(weights[3, 2, 9])
                dut.i_weight_9_3_3 <= fix_pt(weights[3, 3, 9])

                dut.i_weight_10_0_0 <= fix_pt(weights[0, 0, 10])
                dut.i_weight_10_0_1 <= fix_pt(weights[0, 1, 10])
                dut.i_weight_10_0_2 <= fix_pt(weights[0, 2, 10])
                dut.i_weight_10_0_3 <= fix_pt(weights[0, 3, 10])
                dut.i_weight_10_1_0 <= fix_pt(weights[1, 0, 10])
                dut.i_weight_10_1_1 <= fix_pt(weights[1, 1, 10])
                dut.i_weight_10_1_2 <= fix_pt(weights[1, 2, 10])
                dut.i_weight_10_1_3 <= fix_pt(weights[1, 3, 10])
                dut.i_weight_10_2_0 <= fix_pt(weights[2, 0, 10])
                dut.i_weight_10_2_1 <= fix_pt(weights[2, 1, 10])
                dut.i_weight_10_2_2 <= fix_pt(weights[2, 2, 10])
                dut.i_weight_10_2_3 <= fix_pt(weights[2, 3, 10])
                dut.i_weight_10_3_0 <= fix_pt(weights[3, 0, 10])
                dut.i_weight_10_3_1 <= fix_pt(weights[3, 1, 10])
                dut.i_weight_10_3_2 <= fix_pt(weights[3, 2, 10])
                dut.i_weight_10_3_3 <= fix_pt(weights[3, 3, 10])

                dut.i_weight_11_0_0 <= fix_pt(weights[0, 0, 11])
                dut.i_weight_11_0_1 <= fix_pt(weights[0, 1, 11])
                dut.i_weight_11_0_2 <= fix_pt(weights[0, 2, 11])
                dut.i_weight_11_0_3 <= fix_pt(weights[0, 3, 11])
                dut.i_weight_11_1_0 <= fix_pt(weights[1, 0, 11])
                dut.i_weight_11_1_1 <= fix_pt(weights[1, 1, 11])
                dut.i_weight_11_1_2 <= fix_pt(weights[1, 2, 11])
                dut.i_weight_11_1_3 <= fix_pt(weights[1, 3, 11])
                dut.i_weight_11_2_0 <= fix_pt(weights[2, 0, 11])
                dut.i_weight_11_2_1 <= fix_pt(weights[2, 1, 11])
                dut.i_weight_11_2_2 <= fix_pt(weights[2, 2, 11])
                dut.i_weight_11_2_3 <= fix_pt(weights[2, 3, 11])
                dut.i_weight_11_3_0 <= fix_pt(weights[3, 0, 11])
                dut.i_weight_11_3_1 <= fix_pt(weights[3, 1, 11])
                dut.i_weight_11_3_2 <= fix_pt(weights[3, 2, 11])
                dut.i_weight_11_3_3 <= fix_pt(weights[3, 3, 11])

                dut.i_weight_12_0_0 <= fix_pt(weights[0, 0, 12])
                dut.i_weight_12_0_1 <= fix_pt(weights[0, 1, 12])
                dut.i_weight_12_0_2 <= fix_pt(weights[0, 2, 12])
                dut.i_weight_12_0_3 <= fix_pt(weights[0, 3, 12])
                dut.i_weight_12_1_0 <= fix_pt(weights[1, 0, 12])
                dut.i_weight_12_1_1 <= fix_pt(weights[1, 1, 12])
                dut.i_weight_12_1_2 <= fix_pt(weights[1, 2, 12])
                dut.i_weight_12_1_3 <= fix_pt(weights[1, 3, 12])
                dut.i_weight_12_2_0 <= fix_pt(weights[2, 0, 12])
                dut.i_weight_12_2_1 <= fix_pt(weights[2, 1, 12])
                dut.i_weight_12_2_2 <= fix_pt(weights[2, 2, 12])
                dut.i_weight_12_2_3 <= fix_pt(weights[2, 3, 12])
                dut.i_weight_12_3_0 <= fix_pt(weights[3, 0, 12])
                dut.i_weight_12_3_1 <= fix_pt(weights[3, 1, 12])
                dut.i_weight_12_3_2 <= fix_pt(weights[3, 2, 12])
                dut.i_weight_12_3_3 <= fix_pt(weights[3, 3, 12])

                dut.i_weight_13_0_0 <= fix_pt(weights[0, 0, 13])
                dut.i_weight_13_0_1 <= fix_pt(weights[0, 1, 13])
                dut.i_weight_13_0_2 <= fix_pt(weights[0, 2, 13])
                dut.i_weight_13_0_3 <= fix_pt(weights[0, 3, 13])
                dut.i_weight_13_1_0 <= fix_pt(weights[1, 0, 13])
                dut.i_weight_13_1_1 <= fix_pt(weights[1, 1, 13])
                dut.i_weight_13_1_2 <= fix_pt(weights[1, 2, 13])
                dut.i_weight_13_1_3 <= fix_pt(weights[1, 3, 13])
                dut.i_weight_13_2_0 <= fix_pt(weights[2, 0, 13])
                dut.i_weight_13_2_1 <= fix_pt(weights[2, 1, 13])
                dut.i_weight_13_2_2 <= fix_pt(weights[2, 2, 13])
                dut.i_weight_13_2_3 <= fix_pt(weights[2, 3, 13])
                dut.i_weight_13_3_0 <= fix_pt(weights[3, 0, 13])
                dut.i_weight_13_3_1 <= fix_pt(weights[3, 1, 13])
                dut.i_weight_13_3_2 <= fix_pt(weights[3, 2, 13])
                dut.i_weight_13_3_3 <= fix_pt(weights[3, 3, 13])

                dut.i_weight_14_0_0 <= fix_pt(weights[0, 0, 14])
                dut.i_weight_14_0_1 <= fix_pt(weights[0, 1, 14])
                dut.i_weight_14_0_2 <= fix_pt(weights[0, 2, 14])
                dut.i_weight_14_0_3 <= fix_pt(weights[0, 3, 14])
                dut.i_weight_14_1_0 <= fix_pt(weights[1, 0, 14])
                dut.i_weight_14_1_1 <= fix_pt(weights[1, 1, 14])
                dut.i_weight_14_1_2 <= fix_pt(weights[1, 2, 14])
                dut.i_weight_14_1_3 <= fix_pt(weights[1, 3, 14])
                dut.i_weight_14_2_0 <= fix_pt(weights[2, 0, 14])
                dut.i_weight_14_2_1 <= fix_pt(weights[2, 1, 14])
                dut.i_weight_14_2_2 <= fix_pt(weights[2, 2, 14])
                dut.i_weight_14_2_3 <= fix_pt(weights[2, 3, 14])
                dut.i_weight_14_3_0 <= fix_pt(weights[3, 0, 14])
                dut.i_weight_14_3_1 <= fix_pt(weights[3, 1, 14])
                dut.i_weight_14_3_2 <= fix_pt(weights[3, 2, 14])
                dut.i_weight_14_3_3 <= fix_pt(weights[3, 3, 14])

                dut.i_weight_15_0_0 <= fix_pt(weights[0, 0, 15])
                dut.i_weight_15_0_1 <= fix_pt(weights[0, 1, 15])
                dut.i_weight_15_0_2 <= fix_pt(weights[0, 2, 15])
                dut.i_weight_15_0_3 <= fix_pt(weights[0, 3, 15])
                dut.i_weight_15_1_0 <= fix_pt(weights[1, 0, 15])
                dut.i_weight_15_1_1 <= fix_pt(weights[1, 1, 15])
                dut.i_weight_15_1_2 <= fix_pt(weights[1, 2, 15])
                dut.i_weight_15_1_3 <= fix_pt(weights[1, 3, 15])
                dut.i_weight_15_2_0 <= fix_pt(weights[2, 0, 15])
                dut.i_weight_15_2_1 <= fix_pt(weights[2, 1, 15])
                dut.i_weight_15_2_2 <= fix_pt(weights[2, 2, 15])
                dut.i_weight_15_2_3 <= fix_pt(weights[2, 3, 15])
                dut.i_weight_15_3_0 <= fix_pt(weights[3, 0, 15])
                dut.i_weight_15_3_1 <= fix_pt(weights[3, 1, 15])
                dut.i_weight_15_3_2 <= fix_pt(weights[3, 2, 15])
                dut.i_weight_15_3_3 <= fix_pt(weights[3, 3, 15])

                dut.i_weight_16_0_0 <= fix_pt(weights[0, 0, 16])
                dut.i_weight_16_0_1 <= fix_pt(weights[0, 1, 16])
                dut.i_weight_16_0_2 <= fix_pt(weights[0, 2, 16])
                dut.i_weight_16_0_3 <= fix_pt(weights[0, 3, 16])
                dut.i_weight_16_1_0 <= fix_pt(weights[1, 0, 16])
                dut.i_weight_16_1_1 <= fix_pt(weights[1, 1, 16])
                dut.i_weight_16_1_2 <= fix_pt(weights[1, 2, 16])
                dut.i_weight_16_1_3 <= fix_pt(weights[1, 3, 16])
                dut.i_weight_16_2_0 <= fix_pt(weights[2, 0, 16])
                dut.i_weight_16_2_1 <= fix_pt(weights[2, 1, 16])
                dut.i_weight_16_2_2 <= fix_pt(weights[2, 2, 16])
                dut.i_weight_16_2_3 <= fix_pt(weights[2, 3, 16])
                dut.i_weight_16_3_0 <= fix_pt(weights[3, 0, 16])
                dut.i_weight_16_3_1 <= fix_pt(weights[3, 1, 16])
                dut.i_weight_16_3_2 <= fix_pt(weights[3, 2, 16])
                dut.i_weight_16_3_3 <= fix_pt(weights[3, 3, 16])

                dut.i_weight_17_0_0 <= fix_pt(weights[0, 0, 17])
                dut.i_weight_17_0_1 <= fix_pt(weights[0, 1, 17])
                dut.i_weight_17_0_2 <= fix_pt(weights[0, 2, 17])
                dut.i_weight_17_0_3 <= fix_pt(weights[0, 3, 17])
                dut.i_weight_17_1_0 <= fix_pt(weights[1, 0, 17])
                dut.i_weight_17_1_1 <= fix_pt(weights[1, 1, 17])
                dut.i_weight_17_1_2 <= fix_pt(weights[1, 2, 17])
                dut.i_weight_17_1_3 <= fix_pt(weights[1, 3, 17])
                dut.i_weight_17_2_0 <= fix_pt(weights[2, 0, 17])
                dut.i_weight_17_2_1 <= fix_pt(weights[2, 1, 17])
                dut.i_weight_17_2_2 <= fix_pt(weights[2, 2, 17])
                dut.i_weight_17_2_3 <= fix_pt(weights[2, 3, 17])
                dut.i_weight_17_3_0 <= fix_pt(weights[3, 0, 17])
                dut.i_weight_17_3_1 <= fix_pt(weights[3, 1, 17])
                dut.i_weight_17_3_2 <= fix_pt(weights[3, 2, 17])
                dut.i_weight_17_3_3 <= fix_pt(weights[3, 3, 17])

                dut.i_weight_18_0_0 <= fix_pt(weights[0, 0, 18])
                dut.i_weight_18_0_1 <= fix_pt(weights[0, 1, 18])
                dut.i_weight_18_0_2 <= fix_pt(weights[0, 2, 18])
                dut.i_weight_18_0_3 <= fix_pt(weights[0, 3, 18])
                dut.i_weight_18_1_0 <= fix_pt(weights[1, 0, 18])
                dut.i_weight_18_1_1 <= fix_pt(weights[1, 1, 18])
                dut.i_weight_18_1_2 <= fix_pt(weights[1, 2, 18])
                dut.i_weight_18_1_3 <= fix_pt(weights[1, 3, 18])
                dut.i_weight_18_2_0 <= fix_pt(weights[2, 0, 18])
                dut.i_weight_18_2_1 <= fix_pt(weights[2, 1, 18])
                dut.i_weight_18_2_2 <= fix_pt(weights[2, 2, 18])
                dut.i_weight_18_2_3 <= fix_pt(weights[2, 3, 18])
                dut.i_weight_18_3_0 <= fix_pt(weights[3, 0, 18])
                dut.i_weight_18_3_1 <= fix_pt(weights[3, 1, 18])
                dut.i_weight_18_3_2 <= fix_pt(weights[3, 2, 18])
                dut.i_weight_18_3_3 <= fix_pt(weights[3, 3, 18])

                dut.i_weight_19_0_0 <= fix_pt(weights[0, 0, 19])
                dut.i_weight_19_0_1 <= fix_pt(weights[0, 1, 19])
                dut.i_weight_19_0_2 <= fix_pt(weights[0, 2, 19])
                dut.i_weight_19_0_3 <= fix_pt(weights[0, 3, 19])
                dut.i_weight_19_1_0 <= fix_pt(weights[1, 0, 19])
                dut.i_weight_19_1_1 <= fix_pt(weights[1, 1, 19])
                dut.i_weight_19_1_2 <= fix_pt(weights[1, 2, 19])
                dut.i_weight_19_1_3 <= fix_pt(weights[1, 3, 19])
                dut.i_weight_19_2_0 <= fix_pt(weights[2, 0, 19])
                dut.i_weight_19_2_1 <= fix_pt(weights[2, 1, 19])
                dut.i_weight_19_2_2 <= fix_pt(weights[2, 2, 19])
                dut.i_weight_19_2_3 <= fix_pt(weights[2, 3, 19])
                dut.i_weight_19_3_0 <= fix_pt(weights[3, 0, 19])
                dut.i_weight_19_3_1 <= fix_pt(weights[3, 1, 19])
                dut.i_weight_19_3_2 <= fix_pt(weights[3, 2, 19])
                dut.i_weight_19_3_3 <= fix_pt(weights[3, 3, 19])

                time.sleep(10)
                
                
                #print("count is:",count)
                #print("count2 is:",count2)
                print("iter",iter)
                print("ii",ii)

                layer2[count, count2, 0] = fix2float(dut.o_output_0)
                layer2[count, count2, 1] = fix2float(dut.o_output_1)
                layer2[count, count2, 2] = fix2float(dut.o_output_2)
                layer2[count, count2, 3] = fix2float(dut.o_output_3)
                layer2[count, count2, 4] = fix2float(dut.o_output_4)
                layer2[count, count2, 5] = fix2float(dut.o_output_5)
                layer2[count, count2, 6] = fix2float(dut.o_output_6)
                layer2[count, count2, 7] = fix2float(dut.o_output_7)
                layer2[count, count2, 8]  = fix2float(dut.o_output_8)
                layer2[count, count2, 9] = fix2float(dut.o_output_9)
                layer2[count, count2, 10] = fix2float(dut.o_output_10)
                layer2[count, count2, 11] = fix2float(dut.o_output_11)
                layer2[count, count2, 12] = fix2float(dut.o_output_12)
                layer2[count, count2, 13] = fix2float(dut.o_output_13)
                layer2[count, count2, 14] = fix2float(dut.o_output_14)
                layer2[count, count2, 15] = fix2float(dut.o_output_15)
                layer2[count, count2, 16] = fix2float(dut.o_output_16)
                layer2[count, count2, 17] = fix2float(dut.o_output_17)
                layer2[count, count2, 18] = fix2float(dut.o_output_18)
                layer2[count, count2, 19] = fix2float(dut.o_output_19)
                
                


            
            ii = ii + 2
            count2 = count2+1
        iter = iter + 2
        
        count=count+1


    # if i == 0:
        # print('saving data')
        # np.save('exact_fm.npy', layer2)

    # # layer3: conv2d
    # weights = cnn_data['l3weights']
    # shape = (first_layer_kernels, second_layer_kernel_size, second_layer_kernel_size, layer2.shape[0] - second_layer_kernel_size + 1,
             # layer2.shape[1] - second_layer_kernel_size + 1)
    # # stride values: bytesize = size of one data value
    # stride = (bytesize, first_layer_kernels * layer2.shape[0] * bytesize, first_layer_kernels * bytesize, layer2.shape[1] * first_layer_kernels * bytesize,
              # first_layer_kernels * bytesize)
    # subs = np.lib.stride_tricks.as_strided(layer2, shape, stride)
    # layer3 = np.einsum('ijhm,hijkl->klm', weights, subs) + [[cnn_data['l3biases']]]
    # #print_max_err(i, layer3, 'test0l3')

    # # layer4: activation function
    # layer4 = np.maximum(layer3, 0)

    # # layer5: conv2d
    # weights = cnn_data['l5weights']
    # shape = (second_layer_kernels, third_layer_kernel_size, third_layer_kernel_size, layer4.shape[0] - third_layer_kernel_size + 1,
             # layer4.shape[1] - third_layer_kernel_size + 1)
    # # stride values: bytesize = size of one data value
    # stride = (bytesize, second_layer_kernels * layer4.shape[0] * bytesize, second_layer_kernels * bytesize, layer4.shape[1] * second_layer_kernels * bytesize,
              # second_layer_kernels * bytesize)
    # subs = np.lib.stride_tricks.as_strided(layer4, shape, stride)
    # layer5 = np.einsum('ijhm,hijkl->klm', weights, subs) + [[cnn_data['l5biases']]]
    # #print_max_err(i, layer5, 'test0l5')

    # # layer6: activation function
    # layer6 = np.maximum(layer5, 0)
    # #print_max_err(i, layer6, 'test0l6')

    # # layer7: max pooling with 'same' padding
    # layer6_padded = np.zeros(tuple([sum(x) for x in zip(layer6.shape, (1, 1, 0))]))
    # layer6_padded[:9, :9, :] = layer6
    # layer6_padded[9, :9, :] = layer6[8, :, :]
    # layer6_padded[:9, 9, :] = layer6[:, 8, :]
    # layer7 = layer6_padded.reshape(int(layer6_padded.shape[0] / 2), 2, int(layer6_padded.shape[1] / 2), 2, -1).max(axis=(1, 3))
    # #print_max_err(i, layer7, 'test0l7')

    # # layer8: conv2d
    # weights = cnn_data['l8weights']
    # shape = (third_layer_kernels, fourth_layer_kernel_size, fourth_layer_kernel_size, layer7.shape[0] - fourth_layer_kernel_size + 1,
             # layer7.shape[1] - fourth_layer_kernel_size + 1)
    # # stride values: bytesize = size of one data value
    # stride = (bytesize, third_layer_kernels * layer7.shape[0] * bytesize, third_layer_kernels * bytesize, layer7.shape[1] * third_layer_kernels * bytesize,
              # third_layer_kernels * bytesize)
    # subs = np.lib.stride_tricks.as_strided(layer7, shape, stride)
    # layer8 = np.einsum('ijhm,hijkl->klm', weights, subs) + [[cnn_data['l8biases']]]
    # #print_max_err(i, layer8, 'test0l8')

    # # layer9: activation function
    # layer9 = np.maximum(layer8, 0)

    # # layer10: flatten
    # layer10 = layer9.flatten()

    # # layer11: dense
    # layer11 = np.dot(layer10, cnn_data['l11weights']) + cnn_data['l11biases']
    # #print_max_err(i, layer11, 'test0l11')

    # # layer12: dense
    # layer12 = np.dot(layer11, cnn_data['l12weights']) + cnn_data['l12biases']
    # #print_max_err(i, layer12, 'test0l12')

    # # layer13: softmax
    # exp = np.exp(layer12)
    # layer13 = exp / np.sum(exp)

    # if (np.argmax(layer12) == np.argmax(cnn_data['y_test'][i])):
        # correct_predictions += 1

    # print(f'Test {i + 1} of 500. Accuracy: {correct_predictions / (i + 1):.3f}')

