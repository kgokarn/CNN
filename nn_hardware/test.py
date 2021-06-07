import numpy as np
import matplotlib.pyplot as plt
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import time

cnn_data = np.load('cnn_1train_weights_aftest.npz')
new_x = np.zeros((5,5))
layer2 = np.zeros((13,13,20))

for i, img in enumerate(cnn_data['x_test']):

    i=0
    count = 0

    while(i<25):
        ii=0
        count2 = 0
        while(ii<25):
            new_x = img[ii:ii+5,i:i+5]
            @cocotb.test()
            async def test_cnn_hw(dut):
                clock = Clock(dut.clk, 10, units="ns")  # Create a 10us period clock on port clk
                cocotb.fork(clock.start())  # Start the clock

                reset = 0


                await RisingEdge(dut.clk)
                dut.reset <= reset
                dut.i_activation_0_0 <= new_x[0, 0]
                dut.i_activation_0_1 <= new_x[0, 1]
                dut.i_activation_0_2 <= new_x[0, 2]
                dut.i_activation_0_3 <= new_x[0, 3]
                dut.i_activation_0_4 <= new_x[0, 4]
                dut.i_activation_1_0 <= new_x[1, 0]
                dut.i_activation_1_1 <= new_x[1, 1]
                dut.i_activation_1_2 <= new_x[1, 2]
                dut.i_activation_1_3 <= new_x[1, 3]
                dut.i_activation_1_4 <= new_x[1, 4]
                dut.i_activation_2_0 <= new_x[2, 0]
                dut.i_activation_2_1 <= new_x[2, 1]
                dut.i_activation_2_2 <= new_x[2, 2]
                dut.i_activation_2_3 <= new_x[2, 3]
                dut.i_activation_2_4 <= new_x[2, 4]
                dut.i_activation_3_0 <= new_x[3, 0]
                dut.i_activation_3_1 <= new_x[3, 1]
                dut.i_activation_3_2 <= new_x[3, 2]
                dut.i_activation_3_3 <= new_x[3, 3]
                dut.i_activation_3_4 <= new_x[3, 4]
                dut.i_activation_4_0 <= new_x[4, 0]
                dut.i_activation_4_1 <= new_x[4, 1]
                dut.i_activation_4_2 <= new_x[4, 2]
                dut.i_activation_4_3 <= new_x[4, 3]
                dut.i_activation_4_4 <= new_x[4, 4]

                dut.i_weight_0_0_0 <= weights[0, 0, 0]
                dut.i_weight_0_0_1 <= weights[0, 1, 0]
                dut.i_weight_0_0_2 <= weights[0, 2, 0]
                dut.i_weight_0_0_3 <= weights[0, 3, 0]
                dut.i_weight_0_1_0 <= weights[1, 0, 0]
                dut.i_weight_0_1_1 <= weights[1, 1, 0]
                dut.i_weight_0_1_2 <= weights[1, 2, 0]
                dut.i_weight_0_1_3 <= weights[1, 3, 0]
                dut.i_weight_0_2_0 <= weights[2, 0, 0]
                dut.i_weight_0_2_1 <= weights[2, 1, 0]
                dut.i_weight_0_2_2 <= weights[2, 2, 0]
                dut.i_weight_0_2_3 <= weights[2, 3, 0]
                dut.i_weight_0_3_0 <= weights[3, 0, 0]
                dut.i_weight_0_3_1 <= weights[3, 1, 0]
                dut.i_weight_0_3_2 <= weights[3, 2, 0]
                dut.i_weight_0_3_3 <= weights[3, 3, 0]

                dut.i_weight_1_0_0 <= weights[0, 0, 1]
                dut.i_weight_1_0_1 <= weights[0, 1, 1]
                dut.i_weight_1_0_2 <= weights[0, 2, 1]
                dut.i_weight_1_0_3 <= weights[0, 3, 1]
                dut.i_weight_1_1_0 <= weights[1, 0, 1]
                dut.i_weight_1_1_1 <= weights[1, 1, 1]
                dut.i_weight_1_1_2 <= weights[1, 2, 1]
                dut.i_weight_1_1_3 <= weights[1, 3, 1]
                dut.i_weight_1_2_0 <= weights[2, 0, 1]
                dut.i_weight_1_2_1 <= weights[2, 1, 1]
                dut.i_weight_1_2_2 <= weights[2, 2, 1]
                dut.i_weight_1_2_3 <= weights[2, 3, 1]
                dut.i_weight_1_3_0 <= weights[3, 0, 1]
                dut.i_weight_1_3_1 <= weights[3, 1, 1]
                dut.i_weight_1_3_2 <= weights[3, 2, 1]
                dut.i_weight_1_3_3 <= weights[3, 3, 1]

                dut.i_weight_2_0_0 <= weights[0, 0, 2]
                dut.i_weight_2_0_1 <= weights[0, 1, 2]
                dut.i_weight_2_0_2 <= weights[0, 2, 2]
                dut.i_weight_2_0_3 <= weights[0, 3, 2]
                dut.i_weight_2_1_0 <= weights[1, 0, 2]
                dut.i_weight_2_1_1 <= weights[1, 1, 2]
                dut.i_weight_2_1_2 <= weights[1, 2, 2]
                dut.i_weight_2_1_3 <= weights[1, 3, 2]
                dut.i_weight_2_2_0 <= weights[2, 0, 2]
                dut.i_weight_2_2_1 <= weights[2, 1, 2]
                dut.i_weight_2_2_2 <= weights[2, 2, 2]
                dut.i_weight_2_2_3 <= weights[2, 3, 2]
                dut.i_weight_2_3_0 <= weights[3, 0, 2]
                dut.i_weight_2_3_1 <= weights[3, 1, 2]
                dut.i_weight_2_3_2 <= weights[3, 2, 2]
                dut.i_weight_2_3_3 <= weights[3, 3, 2]

                dut.i_weight_3_0_0 <= weights[0, 0, 3]
                dut.i_weight_3_0_1 <= weights[0, 1, 3]
                dut.i_weight_3_0_2 <= weights[0, 2, 3]
                dut.i_weight_3_0_3 <= weights[0, 3, 3]
                dut.i_weight_3_1_0 <= weights[1, 0, 3]
                dut.i_weight_3_1_1 <= weights[1, 1, 3]
                dut.i_weight_3_1_2 <= weights[1, 2, 3]
                dut.i_weight_3_1_3 <= weights[1, 3, 3]
                dut.i_weight_3_2_0 <= weights[2, 0, 3]
                dut.i_weight_3_2_1 <= weights[2, 1, 3]
                dut.i_weight_3_2_2 <= weights[2, 2, 3]
                dut.i_weight_3_2_3 <= weights[2, 3, 3]
                dut.i_weight_3_3_0 <= weights[3, 0, 3]
                dut.i_weight_3_3_1 <= weights[3, 1, 3]
                dut.i_weight_3_3_2 <= weights[3, 2, 3]
                dut.i_weight_3_3_3 <= weights[3, 3, 3]

                dut.i_weight_4_0_0 <= weights[0, 0, 4]
                dut.i_weight_4_0_1 <= weights[0, 1, 4]
                dut.i_weight_4_0_2 <= weights[0, 2, 4]
                dut.i_weight_4_0_3 <= weights[0, 3, 4]
                dut.i_weight_4_1_0 <= weights[1, 0, 4]
                dut.i_weight_4_1_1 <= weights[1, 1, 4]
                dut.i_weight_4_1_2 <= weights[1, 2, 4]
                dut.i_weight_4_1_3 <= weights[1, 3, 4]
                dut.i_weight_4_2_0 <= weights[2, 0, 4]
                dut.i_weight_4_2_1 <= weights[2, 1, 4]
                dut.i_weight_4_2_2 <= weights[2, 2, 4]
                dut.i_weight_4_2_3 <= weights[2, 3, 4]
                dut.i_weight_4_3_0 <= weights[3, 0, 4]
                dut.i_weight_4_3_1 <= weights[3, 1, 4]
                dut.i_weight_4_3_2 <= weights[3, 2, 4]
                dut.i_weight_4_3_3 <= weights[3, 3, 4]

                dut.i_weight_5_0_0 <= weights[0, 0, 5]
                dut.i_weight_5_0_1 <= weights[0, 1, 5]
                dut.i_weight_5_0_2 <= weights[0, 2, 5]
                dut.i_weight_5_0_3 <= weights[0, 3, 5]
                dut.i_weight_5_1_0 <= weights[1, 0, 5]
                dut.i_weight_5_1_1 <= weights[1, 1, 5]
                dut.i_weight_5_1_2 <= weights[1, 2, 5]
                dut.i_weight_5_1_3 <= weights[1, 3, 5]
                dut.i_weight_5_2_0 <= weights[2, 0, 5]
                dut.i_weight_5_2_1 <= weights[2, 1, 5]
                dut.i_weight_5_2_2 <= weights[2, 2, 5]
                dut.i_weight_5_2_3 <= weights[2, 3, 5]
                dut.i_weight_5_3_0 <= weights[3, 0, 5]
                dut.i_weight_5_3_1 <= weights[3, 1, 5]
                dut.i_weight_5_3_2 <= weights[3, 2, 5]
                dut.i_weight_5_3_3 <= weights[3, 3, 5]

                dut.i_weight_6_0_0 <= weights[0, 0, 6]
                dut.i_weight_6_0_1 <= weights[0, 1, 6]
                dut.i_weight_6_0_2 <= weights[0, 2, 6]
                dut.i_weight_6_0_3 <= weights[0, 3, 6]
                dut.i_weight_6_1_0 <= weights[1, 0, 6]
                dut.i_weight_6_1_1 <= weights[1, 1, 6]
                dut.i_weight_6_1_2 <= weights[1, 2, 6]
                dut.i_weight_6_1_3 <= weights[1, 3, 6]
                dut.i_weight_6_2_0 <= weights[2, 0, 6]
                dut.i_weight_6_2_1 <= weights[2, 1, 6]
                dut.i_weight_6_2_2 <= weights[2, 2, 6]
                dut.i_weight_6_2_3 <= weights[2, 3, 6]
                dut.i_weight_6_3_0 <= weights[3, 0, 6]
                dut.i_weight_6_3_1 <= weights[3, 1, 6]
                dut.i_weight_6_3_2 <= weights[3, 2, 6]
                dut.i_weight_6_3_3 <= weights[3, 3, 6]

                dut.i_weight_7_0_0 <= weights[0, 0, 7]
                dut.i_weight_7_0_1 <= weights[0, 1, 7]
                dut.i_weight_7_0_2 <= weights[0, 2, 7]
                dut.i_weight_7_0_3 <= weights[0, 3, 7]
                dut.i_weight_7_1_0 <= weights[1, 0, 7]
                dut.i_weight_7_1_1 <= weights[1, 1, 7]
                dut.i_weight_7_1_2 <= weights[1, 2, 7]
                dut.i_weight_7_1_3 <= weights[1, 3, 7]
                dut.i_weight_7_2_0 <= weights[2, 0, 7]
                dut.i_weight_7_2_1 <= weights[2, 1, 7]
                dut.i_weight_7_2_2 <= weights[2, 2, 7]
                dut.i_weight_7_2_3 <= weights[2, 3, 7]
                dut.i_weight_7_3_0 <= weights[3, 0, 7]
                dut.i_weight_7_3_1 <= weights[3, 1, 7]
                dut.i_weight_7_3_2 <= weights[3, 2, 7]
                dut.i_weight_7_3_3 <= weights[3, 3, 7]

                dut.i_weight_8_0_0 <= weights[0, 0, 8]
                dut.i_weight_8_0_1 <= weights[0, 1, 8]
                dut.i_weight_8_0_2 <= weights[0, 2, 8]
                dut.i_weight_8_0_3 <= weights[0, 3, 8]
                dut.i_weight_8_1_0 <= weights[1, 0, 8]
                dut.i_weight_8_1_1 <= weights[1, 1, 8]
                dut.i_weight_8_1_2 <= weights[1, 2, 8]
                dut.i_weight_8_1_3 <= weights[1, 3, 8]
                dut.i_weight_8_2_0 <= weights[2, 0, 8]
                dut.i_weight_8_2_1 <= weights[2, 1, 8]
                dut.i_weight_8_2_2 <= weights[2, 2, 8]
                dut.i_weight_8_2_3 <= weights[2, 3, 8]
                dut.i_weight_8_3_0 <= weights[3, 0, 8]
                dut.i_weight_8_3_1 <= weights[3, 1, 8]
                dut.i_weight_8_3_2 <= weights[3, 2, 8]
                dut.i_weight_8_3_3 <= weights[3, 3, 8]

                dut.i_weight_9_0_0 <= weights[0, 0, 9]
                dut.i_weight_9_0_1 <= weights[0, 1, 9]
                dut.i_weight_9_0_2 <= weights[0, 2, 9]
                dut.i_weight_9_0_3 <= weights[0, 3, 9]
                dut.i_weight_9_1_0 <= weights[1, 0, 9]
                dut.i_weight_9_1_1 <= weights[1, 1, 9]
                dut.i_weight_9_1_2 <= weights[1, 2, 9]
                dut.i_weight_9_1_3 <= weights[1, 3, 9]
                dut.i_weight_9_2_0 <= weights[2, 0, 9]
                dut.i_weight_9_2_1 <= weights[2, 1, 9]
                dut.i_weight_9_2_2 <= weights[2, 2, 9]
                dut.i_weight_9_2_3 <= weights[2, 3, 9]
                dut.i_weight_9_3_0 <= weights[3, 0, 9]
                dut.i_weight_9_3_1 <= weights[3, 1, 9]
                dut.i_weight_9_3_2 <= weights[3, 2, 9]
                dut.i_weight_9_3_3 <= weights[3, 3, 9]

                dut.i_weight_10_0_0 <= weights[0, 0, 10]
                dut.i_weight_10_0_1 <= weights[0, 1, 10]
                dut.i_weight_10_0_2 <= weights[0, 2, 10]
                dut.i_weight_10_0_3 <= weights[0, 3, 10]
                dut.i_weight_10_1_0 <= weights[1, 0, 10]
                dut.i_weight_10_1_1 <= weights[1, 1, 10]
                dut.i_weight_10_1_2 <= weights[1, 2, 10]
                dut.i_weight_10_1_3 <= weights[1, 3, 10]
                dut.i_weight_10_2_0 <= weights[2, 0, 10]
                dut.i_weight_10_2_1 <= weights[2, 1, 10]
                dut.i_weight_10_2_2 <= weights[2, 2, 10]
                dut.i_weight_10_2_3 <= weights[2, 3, 10]
                dut.i_weight_10_3_0 <= weights[3, 0, 10]
                dut.i_weight_10_3_1 <= weights[3, 1, 10]
                dut.i_weight_10_3_2 <= weights[3, 2, 10]
                dut.i_weight_10_3_3 <= weights[3, 3, 10]

                dut.i_weight_11_0_0 <= weights[0, 0, 11]
                dut.i_weight_11_0_1 <= weights[0, 1, 11]
                dut.i_weight_11_0_2 <= weights[0, 2, 11]
                dut.i_weight_11_0_3 <= weights[0, 3, 11]
                dut.i_weight_11_1_0 <= weights[1, 0, 11]
                dut.i_weight_11_1_1 <= weights[1, 1, 11]
                dut.i_weight_11_1_2 <= weights[1, 2, 11]
                dut.i_weight_11_1_3 <= weights[1, 3, 11]
                dut.i_weight_11_2_0 <= weights[2, 0, 11]
                dut.i_weight_11_2_1 <= weights[2, 1, 11]
                dut.i_weight_11_2_2 <= weights[2, 2, 11]
                dut.i_weight_11_2_3 <= weights[2, 3, 11]
                dut.i_weight_11_3_0 <= weights[3, 0, 11]
                dut.i_weight_11_3_1 <= weights[3, 1, 11]
                dut.i_weight_11_3_2 <= weights[3, 2, 11]
                dut.i_weight_11_3_3 <= weights[3, 3, 11]

                dut.i_weight_12_0_0 <= weights[0, 0, 12]
                dut.i_weight_12_0_1 <= weights[0, 1, 12]
                dut.i_weight_12_0_2 <= weights[0, 2, 12]
                dut.i_weight_12_0_3 <= weights[0, 3, 12]
                dut.i_weight_12_1_0 <= weights[1, 0, 12]
                dut.i_weight_12_1_1 <= weights[1, 1, 12]
                dut.i_weight_12_1_2 <= weights[1, 2, 12]
                dut.i_weight_12_1_3 <= weights[1, 3, 12]
                dut.i_weight_12_2_0 <= weights[2, 0, 12]
                dut.i_weight_12_2_1 <= weights[2, 1, 12]
                dut.i_weight_12_2_2 <= weights[2, 2, 12]
                dut.i_weight_12_2_3 <= weights[2, 3, 12]
                dut.i_weight_12_3_0 <= weights[3, 0, 12]
                dut.i_weight_12_3_1 <= weights[3, 1, 12]
                dut.i_weight_12_3_2 <= weights[3, 2, 12]
                dut.i_weight_12_3_3 <= weights[3, 3, 12]

                dut.i_weight_13_0_0 <= weights[0, 0, 13]
                dut.i_weight_13_0_1 <= weights[0, 1, 13]
                dut.i_weight_13_0_2 <= weights[0, 2, 13]
                dut.i_weight_13_0_3 <= weights[0, 3, 13]
                dut.i_weight_13_1_0 <= weights[1, 0, 13]
                dut.i_weight_13_1_1 <= weights[1, 1, 13]
                dut.i_weight_13_1_2 <= weights[1, 2, 13]
                dut.i_weight_13_1_3 <= weights[1, 3, 13]
                dut.i_weight_13_2_0 <= weights[2, 0, 13]
                dut.i_weight_13_2_1 <= weights[2, 1, 13]
                dut.i_weight_13_2_2 <= weights[2, 2, 13]
                dut.i_weight_13_2_3 <= weights[2, 3, 13]
                dut.i_weight_13_3_0 <= weights[3, 0, 13]
                dut.i_weight_13_3_1 <= weights[3, 1, 13]
                dut.i_weight_13_3_2 <= weights[3, 2, 13]
                dut.i_weight_13_3_3 <= weights[3, 3, 13]

                dut.i_weight_14_0_0 <= weights[0, 0, 14]
                dut.i_weight_14_0_1 <= weights[0, 1, 14]
                dut.i_weight_14_0_2 <= weights[0, 2, 14]
                dut.i_weight_14_0_3 <= weights[0, 3, 14]
                dut.i_weight_14_1_0 <= weights[1, 0, 14]
                dut.i_weight_14_1_1 <= weights[1, 1, 14]
                dut.i_weight_14_1_2 <= weights[1, 2, 14]
                dut.i_weight_14_1_3 <= weights[1, 3, 14]
                dut.i_weight_14_2_0 <= weights[2, 0, 14]
                dut.i_weight_14_2_1 <= weights[2, 1, 14]
                dut.i_weight_14_2_2 <= weights[2, 2, 14]
                dut.i_weight_14_2_3 <= weights[2, 3, 14]
                dut.i_weight_14_3_0 <= weights[3, 0, 14]
                dut.i_weight_14_3_1 <= weights[3, 1, 14]
                dut.i_weight_14_3_2 <= weights[3, 2, 14]
                dut.i_weight_14_3_3 <= weights[3, 3, 14]

                dut.i_weight_15_0_0 <= weights[0, 0, 15]
                dut.i_weight_15_0_1 <= weights[0, 1, 15]
                dut.i_weight_15_0_2 <= weights[0, 2, 15]
                dut.i_weight_15_0_3 <= weights[0, 3, 15]
                dut.i_weight_15_1_0 <= weights[1, 0, 15]
                dut.i_weight_15_1_1 <= weights[1, 1, 15]
                dut.i_weight_15_1_2 <= weights[1, 2, 15]
                dut.i_weight_15_1_3 <= weights[1, 3, 15]
                dut.i_weight_15_2_0 <= weights[2, 0, 15]
                dut.i_weight_15_2_1 <= weights[2, 1, 15]
                dut.i_weight_15_2_2 <= weights[2, 2, 15]
                dut.i_weight_15_2_3 <= weights[2, 3, 15]
                dut.i_weight_15_3_0 <= weights[3, 0, 15]
                dut.i_weight_15_3_1 <= weights[3, 1, 15]
                dut.i_weight_15_3_2 <= weights[3, 2, 15]
                dut.i_weight_15_3_3 <= weights[3, 3, 15]

                dut.i_weight_16_0_0 <= weights[0, 0, 16]
                dut.i_weight_16_0_1 <= weights[0, 1, 16]
                dut.i_weight_16_0_2 <= weights[0, 2, 16]
                dut.i_weight_16_0_3 <= weights[0, 3, 16]
                dut.i_weight_16_1_0 <= weights[1, 0, 16]
                dut.i_weight_16_1_1 <= weights[1, 1, 16]
                dut.i_weight_16_1_2 <= weights[1, 2, 16]
                dut.i_weight_16_1_3 <= weights[1, 3, 16]
                dut.i_weight_16_2_0 <= weights[2, 0, 16]
                dut.i_weight_16_2_1 <= weights[2, 1, 16]
                dut.i_weight_16_2_2 <= weights[2, 2, 16]
                dut.i_weight_16_2_3 <= weights[2, 3, 16]
                dut.i_weight_16_3_0 <= weights[3, 0, 16]
                dut.i_weight_16_3_1 <= weights[3, 1, 16]
                dut.i_weight_16_3_2 <= weights[3, 2, 16]
                dut.i_weight_16_3_3 <= weights[3, 3, 16]

                dut.i_weight_17_0_0 <= weights[0, 0, 17]
                dut.i_weight_17_0_1 <= weights[0, 1, 17]
                dut.i_weight_17_0_2 <= weights[0, 2, 17]
                dut.i_weight_17_0_3 <= weights[0, 3, 17]
                dut.i_weight_17_1_0 <= weights[1, 0, 17]
                dut.i_weight_17_1_1 <= weights[1, 1, 17]
                dut.i_weight_17_1_2 <= weights[1, 2, 17]
                dut.i_weight_17_1_3 <= weights[1, 3, 17]
                dut.i_weight_17_2_0 <= weights[2, 0, 17]
                dut.i_weight_17_2_1 <= weights[2, 1, 17]
                dut.i_weight_17_2_2 <= weights[2, 2, 17]
                dut.i_weight_17_2_3 <= weights[2, 3, 17]
                dut.i_weight_17_3_0 <= weights[3, 0, 17]
                dut.i_weight_17_3_1 <= weights[3, 1, 17]
                dut.i_weight_17_3_2 <= weights[3, 2, 17]
                dut.i_weight_17_3_3 <= weights[3, 3, 17]

                dut.i_weight_18_0_0 <= weights[0, 0, 18]
                dut.i_weight_18_0_1 <= weights[0, 1, 18]
                dut.i_weight_18_0_2 <= weights[0, 2, 18]
                dut.i_weight_18_0_3 <= weights[0, 3, 18]
                dut.i_weight_18_1_0 <= weights[1, 0, 18]
                dut.i_weight_18_1_1 <= weights[1, 1, 18]
                dut.i_weight_18_1_2 <= weights[1, 2, 18]
                dut.i_weight_18_1_3 <= weights[1, 3, 18]
                dut.i_weight_18_2_0 <= weights[2, 0, 18]
                dut.i_weight_18_2_1 <= weights[2, 1, 18]
                dut.i_weight_18_2_2 <= weights[2, 2, 18]
                dut.i_weight_18_2_3 <= weights[2, 3, 18]
                dut.i_weight_18_3_0 <= weights[3, 0, 18]
                dut.i_weight_18_3_1 <= weights[3, 1, 18]
                dut.i_weight_18_3_2 <= weights[3, 2, 18]
                dut.i_weight_18_3_3 <= weights[3, 3, 18]

                dut.i_weight_19_0_0 <= weights[0, 0, 19]
                dut.i_weight_19_0_1 <= weights[0, 1, 19]
                dut.i_weight_19_0_2 <= weights[0, 2, 19]
                dut.i_weight_19_0_3 <= weights[0, 3, 19]
                dut.i_weight_19_1_0 <= weights[1, 0, 19]
                dut.i_weight_19_1_1 <= weights[1, 1, 19]
                dut.i_weight_19_1_2 <= weights[1, 2, 19]
                dut.i_weight_19_1_3 <= weights[1, 3, 19]
                dut.i_weight_19_2_0 <= weights[2, 0, 19]
                dut.i_weight_19_2_1 <= weights[2, 1, 19]
                dut.i_weight_19_2_2 <= weights[2, 2, 19]
                dut.i_weight_19_2_3 <= weights[2, 3, 19]
                dut.i_weight_19_3_0 <= weights[3, 0, 19]
                dut.i_weight_19_3_1 <= weights[3, 1, 19]
                dut.i_weight_19_3_2 <= weights[3, 2, 19]
                dut.i_weight_19_3_3 <= weights[3, 3, 19]

                time.sleep(10)

                out0 = dut.o_output_0
                out1 = dut.o_output_1
                out2 = dut.o_output_2
                out3 = dut.o_output_3
                out4 = dut.o_output_4
                out5 = dut.o_output_5
                out6 = dut.o_output_6
                out7 = dut.o_output_7
                out8 = dut.o_output_8
                out9 = dut.o_output_9
                out10 = dut.o_output_10
                out11 = dut.o_output_11
                out12 = dut.o_output_12
                out13 = dut.o_output_13
                out14 = dut.o_output_14
                out15 = dut.o_output_15
                out16 = dut.o_output_16
                out17 = dut.o_output_17
                out18 = dut.o_output_18
                out19 = dut.o_output_19

            layer2[count,count2,0] = out0
            layer2[count, count2, 1] = out1
            layer2[count, count2, 2] = out2
            layer2[count, count2, 3] = out3
            layer2[count, count2, 4] = out4
            layer2[count, count2, 5] = out5
            layer2[count, count2, 6] = out6
            layer2[count, count2, 7] = out7
            layer2[count, count2, 8] = out8
            layer2[count, count2, 9] = out9
            layer2[count, count2, 10] = out10
            layer2[count, count2, 11] = out11
            layer2[count, count2, 12] = out12
            layer2[count, count2, 13] = out13
            layer2[count, count2, 14] = out14
            layer2[count, count2, 15] = out15
            layer2[count, count2, 16] = out16
            layer2[count, count2, 17] = out17
            layer2[count, count2, 18] = out18
            layer2[count, count2, 19] = out19



            ii = ii + 2
            count2 = count2+1
        i = i + 2
        count=count+1


    # # layer0: conv2d
    # weights = cnn_data['l0weights']
    # f_shape = weights[::, ::, 0].shape
    # shape = f_shape + tuple(np.subtract(img.shape, f_shape) + 1)
    # subs = np.lib.stride_tricks.as_strided(img, shape, img.strides * 2)
    # layer0 = np.einsum('ijm,ijkl->klm', weights, subs)
    # #print_max_err(i, layer0, 'test0l0')


    # # layer1: activation function
    # layer1 = np.tanh(0.75 * layer0)
    # # layer1 = pwl_activation(layer0)

    # # layer2: max pooling 2d
    # layer2 = layer1.reshape(int(layer1.shape[0] / 2), 2, int(layer1.shape[1] / 2), 2, -1).max(axis=(1, 3))
    # # print_max_err(i, layer2, 'test0l2')

