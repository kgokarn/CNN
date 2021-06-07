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

@cocotb.test()
async def test_cnn_hw(dut, new_x=new_x, weights=weights,layer2=layer2,iter=iter,ii=ii,count=count,count2=count2):
    clock = Clock(dut.clk, 10, units="ns")  # Create a 10ns period clock on port clk
    cocotb.fork(clock.start())  # Start the clock

    reset = 0

    print("iter before is", iter)
    print("ii before is", ii)

    await RisingEdge(dut.clk)
    dut.reset <= reset

    for i2 in range(5):
        for i3 in range(5):
            getattr(dut, f"i_activation_{i2}_{i3}") <= fix_pt(new_x[i2, i3])


    for i4 in range(20):
        for i5 in range(4):
            for i6 in range(4):
                getattr(dut, f"i_weight_{i4}_{i5}_{i6}") <= fix_pt(weights[i4,i5,i6])




    time.sleep(10)


    layer2[count, count2, 0] = fix2float(dut.o_output_0)
    layer2[count, count2, 1] = fix2float(dut.o_output_1)
    layer2[count, count2, 2] = fix2float(dut.o_output_2)
    layer2[count, count2, 3] = fix2float(dut.o_output_3)
    layer2[count, count2, 4] = fix2float(dut.o_output_4)
    layer2[count, count2, 5] = fix2float(dut.o_output_5)
    layer2[count, count2, 6] = fix2float(dut.o_output_6)
    layer2[count, count2, 7] = fix2float(dut.o_output_7)
    layer2[count, count2, 8] = fix2float(dut.o_output_8)
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
    
    return layer2
