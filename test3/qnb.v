module full_adder(input1_i,input2_i,carry_i,sum_o,carry_o);
	input input1_i, input2_i;
	input carry_i;
	output sum_o;
	output carry_o;

	assign sum_o = input1_i ^ input2_i ^ carry_i;
	assign carry_o = (input1_i & input2_i) | (input1_i & carry_i) | (input2_i & carry_i);

endmodule


module small_adder(input1_i, input2_i, generate_term_i, propagate_term_i, sum_o, carry_o);
	parameter width = 7;
	input [width-1:0] input1_i;
	input [width-1:0] input2_i;
	input [width-1:0] generate_term_i, propagate_term_i;
	output [width-1:0] sum_o;
	output [width:0] carry_o;

	wire [width-1:0] sum_w;
	wire [width:0] carry_w;

	assign carry_w[0] = 1'b0;

	genvar i;

	generate
		for(i = 0; i < width; i = i+1)
		begin
			full_adder fa(.input1_i(input1_i[i]), .input2_i(input2_i[i]), .carry_i(carry_w[i]), .sum_o(sum_w[i]), .carry_o());
		end
	endgenerate

	genvar j;

	generate
		for(j=0; j<width; j = j+1)
		begin
			assign carry_w[j+1] = generate_term_i[j]| (propagate_term_i[j] & carry_w[j]);
		end

	endgenerate

	assign sum_o = sum_w;
	assign carry_o = carry_w;

endmodule


module small_adder_msb(input1_i, input2_i, generate_term_i, propagate_term_i, sum_o, carry_o);
	parameter width = 7;
	input [width-1:0] input1_i;
	input [width-1:0] input2_i;
	input [width-1:0] generate_term_i, propagate_term_i;
	output sum_o, carry_o;


	wire [width-1:0] sum_w;
	wire [width:0] carry_w;


	assign carry_w[0] = 1'b0;

	genvar i;

	generate
		for(i = 0; i < width; i = i+1)
		begin
			full_adder fa(.input1_i(input1_i[i]), .input2_i(input2_i[i]), .carry_i(carry_w[i]), .sum_o(sum_w[i]), .carry_o());
		end
	endgenerate

	genvar j;

	generate
		for(j=0; j<width; j = j+1)
		begin
			assign carry_w[j+1] = generate_term_i[j]| (propagate_term_i[j] & carry_w[j]);
		end

	endgenerate

	assign sum_o = sum_w[width-1];
	assign carry_o = carry_w[width];

endmodule


module almost_correct_adder(add1_i, add2_i, result_o);
parameter width = 28;
parameter logwidth = 7;

input [width-1:0] add1_i;
input [width-1:0] add2_i;
output [width-1:0] result_o;

wire [width-1:0] generate_w, propagate_w, sum_w;
wire [width:0] carry_w;

genvar i;

generate
	for(i=0; i<width-logwidth+1; i = i+1)
		begin
		if(i < 1)
			begin
			small_adder  sa(.input1_i(add1_i[logwidth-1+i:i]),.input2_i(add2_i[logwidth-1+i:i]),.generate_term_i(generate_w[logwidth-1+i:i]),.propagate_term_i(propagate_w[logwidth-1+i:i]),.sum_o(sum_w[logwidth-1+i:i]),.carry_o(carry_w[logwidth+i:i]));
			end
		else
			begin
			small_adder_msb samsb(.input1_i(add1_i[logwidth-1+i:i]),.input2_i(add2_i[logwidth-1+i:i]),.generate_term_i(generate_w[logwidth-1+i:i]),.propagate_term_i(propagate_w[logwidth-1+i:i]),.sum_o(sum_w[logwidth-1+i]),.carry_o(carry_w[logwidth+i]));
			end
		end

endgenerate



genvar j;

generate
	for(j=0; j<width; j = j+1)
		begin
		assign generate_w[j] = add1_i[j] & add2_i[j];
		assign propagate_w[j] = add1_i[j] | add2_i[j];
		end

endgenerate

assign result_o = sum_w;

endmodule


module macb(
    input wire clk,
    input wire reset,
    input wire next_mac_input,
    input wire signed [11:0] i_activation_0,
    input wire signed [11:0] i_activation_1,
    input wire signed [11:0] i_activation_2,
    input wire signed [11:0] i_activation_3,
    input wire signed [11:0] i_activation_4,
    input wire signed [11:0] i_activation_5,
    input wire signed [11:0] i_activation_6,
    input wire signed [11:0] i_activation_7,
    input wire signed [11:0] i_activation_8,
    input wire signed [11:0] i_activation_9,
    input wire signed [11:0] i_activation_10,
    input wire signed [11:0] i_activation_11,
    input wire signed [11:0] i_activation_12,
    input wire signed [11:0] i_activation_13,
    input wire signed [11:0] i_activation_14,
    input wire signed [11:0] i_activation_15,
    input wire signed [11:0] i_weight_0,
    input wire signed [11:0] i_weight_1,
    input wire signed [11:0] i_weight_2,
    input wire signed [11:0] i_weight_3,
    input wire signed [11:0] i_weight_4,
    input wire signed [11:0] i_weight_5,
    input wire signed [11:0] i_weight_6,
    input wire signed [11:0] i_weight_7,
    input wire signed [11:0] i_weight_8,
    input wire signed [11:0] i_weight_9,
    input wire signed [11:0] i_weight_10,
    input wire signed [11:0] i_weight_11,
    input wire signed [11:0] i_weight_12,
    input wire signed [11:0] i_weight_13,
    input wire signed [11:0] i_weight_14,
    input wire signed [11:0] i_weight_15,
    output wire signed [27:0] o_output,
    output wire ready_out,
    output wire running_out
    );
    
    reg [4:0] state;
    reg ready;
    reg running;
    reg signed [27:0] accumulator;
	reg signed [27:0] acti_weight;
	wire signed [27:0] int_sum;
	
	almost_correct_adder aca(.add1_i(accumulator), .add2_i(acti_weight), .result_o(int_sum));

    
   always @ (posedge clk) begin
        if (reset | next_mac_input) begin
            accumulator <= 0;
			acti_weight <=0;
            ready <= 0;
            state <= 0;
            running <= 0;
        end else begin
            case(state) 
                5'b00000: begin
                    running <= 1;
					accumulator <= int_sum;
                    acti_weight <= i_activation_0 * i_weight_0;
                    state <= state + 1;
                end
                5'b00001: begin
					accumulator <= int_sum;
                    acti_weight <= i_activation_1 * i_weight_1;
                    state <= state + 1;
                end
                5'b00010: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_2 * i_weight_2;
                    state <= state + 1;
                end
                5'b00011: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_3 * i_weight_3;
                    state <= state + 1;
                end
                5'b00100: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_4 * i_weight_4;
                    state <= state + 1;
                end
                5'b00101: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_5 * i_weight_5;
                    state <= state + 1;
                end
                5'b00110: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_6 * i_weight_6;
                    state <= state + 1;
                end
                5'b00111: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_7 * i_weight_7;
                    state <= state + 1;
                end
                5'b01000: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_8 * i_weight_8;
                    state <= state + 1;
                end
                5'b01001: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_9 * i_weight_9;
                    state <= state + 1;
                end
                5'b01010: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_10 * i_weight_10;
                    state <= state + 1;
                end
                5'b01011: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_11 * i_weight_11;
                    state <= state + 1;
                end
                5'b01100: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_12 * i_weight_12;
                    state <= state + 1;
                end
                5'b01101: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_13 * i_weight_13;
                    state <= state + 1;
                end
                5'b01110: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_14 * i_weight_14;
                    state <= state + 1;
                end
                5'b01111: begin
					accumulator <= int_sum;
					acti_weight <= i_activation_15 * i_weight_15;
                    ready <= 1;
                    running <= 0;
                    state <= state + 1;
                end
                5'b10000: begin
					accumulator <= int_sum;
                    state <= 5'b00000;
                    ready <= 0;
                end
            endcase
        end
    end
    
	assign o_output = accumulator;
    assign ready_out = ready;
    assign running_out = running;
    
endmodule

module afb(
    input wire signed [16:0] activation,
    output wire signed [11:0] acti_result
    );
    
    wire signed [16:0] temp_result;
    
    assign temp_result = (activation < $signed(17'b1_1111_0000_0000_0000)) ? 17'b1_1111_1000_0000_0000 :
                         ((activation >= $signed(17'b1_1111_0000_0000_0000)) & (activation < $signed(17'b1_1111_1010_1000_0000))) ? ((activation >>> 2) + $signed(17'b1_1111_1100_0000_0000)) :
                         ((activation >= $signed(17'b1_1111_1010_1000_0000)) & (activation < $signed(17'b0_0000_0101_1000_0000))) ? activation :
                         ((activation >= $signed(17'b0_0000_0101_1000_0000)) & (activation < $signed(17'b0_0001_0000_0000_0000))) ? ((activation >>> 2) + $signed(17'b0_0000_0100_0000_0000)) :
                         17'b0_0000_0111_1111_1111;
    
    assign acti_result = temp_result[11:0];
    
endmodule

module neuron_top(
    input wire clk,
    input wire reset,
    input wire next_neuron_input,
    input wire signed [11:0] i_activation_0,
    input wire signed [11:0] i_activation_1,
    input wire signed [11:0] i_activation_2,
    input wire signed [11:0] i_activation_3,
    input wire signed [11:0] i_activation_4,
    input wire signed [11:0] i_activation_5,
    input wire signed [11:0] i_activation_6,
    input wire signed [11:0] i_activation_7,
    input wire signed [11:0] i_activation_8,
    input wire signed [11:0] i_activation_9,
    input wire signed [11:0] i_activation_10,
    input wire signed [11:0] i_activation_11,
    input wire signed [11:0] i_activation_12,
    input wire signed [11:0] i_activation_13,
    input wire signed [11:0] i_activation_14,
    input wire signed [11:0] i_activation_15,
    input wire signed [11:0] i_activation_16,
    input wire signed [11:0] i_activation_17,
    input wire signed [11:0] i_activation_18,
    input wire signed [11:0] i_activation_19,
    input wire signed [11:0] i_activation_20,
    input wire signed [11:0] i_activation_21,
    input wire signed [11:0] i_activation_22,
    input wire signed [11:0] i_activation_23,
    input wire signed [11:0] i_activation_24,
    input wire signed [11:0] i_weight_0,
    input wire signed [11:0] i_weight_1,
    input wire signed [11:0] i_weight_2,
    input wire signed [11:0] i_weight_3,
    input wire signed [11:0] i_weight_4,
    input wire signed [11:0] i_weight_5,
    input wire signed [11:0] i_weight_6,
    input wire signed [11:0] i_weight_7,
    input wire signed [11:0] i_weight_8,
    input wire signed [11:0] i_weight_9,
    input wire signed [11:0] i_weight_10,
    input wire signed [11:0] i_weight_11,
    input wire signed [11:0] i_weight_12,
    input wire signed [11:0] i_weight_13,
    input wire signed [11:0] i_weight_14,
    input wire signed [11:0] i_weight_15,
    output wire signed [11:0] o_output,
    output wire ready_out,
    output wire running_out
    );
    
    reg [2:0] fsm_state;
    reg next_mac_input;
    wire mac_ready;
    wire mac_running;
    reg neuron_ready;
    reg neuron_running;
    wire signed [27:0] mac_out;
    reg signed [11:0] afb_out_0;
    reg signed [11:0] afb_out_1;
    reg signed [11:0] afb_out_2;
    reg signed [11:0] afb_out_3;
    reg signed [16:0] afb_in;
    wire signed [11:0] afb_out;
    reg signed [11:0] i_activation_mac_0, i_activation_mac_1, i_activation_mac_2, i_activation_mac_3, i_activation_mac_4, i_activation_mac_5,
                      i_activation_mac_6, i_activation_mac_7, i_activation_mac_8, i_activation_mac_9, i_activation_mac_10, i_activation_mac_11,
                      i_activation_mac_12, i_activation_mac_13, i_activation_mac_14, i_activation_mac_15;
                      
    macb macb_unit(.clk(clk), .reset(reset), .next_mac_input(next_mac_input), .i_activation_0(i_activation_mac_0), .i_activation_1(i_activation_mac_1), .i_activation_2(i_activation_mac_2),
                   .i_activation_3(i_activation_mac_3), .i_activation_4(i_activation_mac_4), .i_activation_5(i_activation_mac_5), .i_activation_6(i_activation_mac_6),
                   .i_activation_7(i_activation_mac_7), .i_activation_8(i_activation_mac_8), .i_activation_9(i_activation_mac_9), .i_activation_10(i_activation_mac_10),
                   .i_activation_11(i_activation_mac_11), .i_activation_12(i_activation_mac_12), .i_activation_13(i_activation_mac_13), .i_activation_14(i_activation_mac_14), .i_activation_15(i_activation_mac_15),
                   .i_weight_0(i_weight_0), .i_weight_1(i_weight_1), .i_weight_2(i_weight_2), .i_weight_3(i_weight_3), .i_weight_4(i_weight_4),
                   .i_weight_5(i_weight_5), .i_weight_6(i_weight_6), .i_weight_7(i_weight_7), .i_weight_8(i_weight_8), .i_weight_9(i_weight_9),
                   .i_weight_10(i_weight_10), .i_weight_11(i_weight_11), .i_weight_12(i_weight_12), .i_weight_13(i_weight_13), .i_weight_14(i_weight_14), .i_weight_15(i_weight_15),
                   .o_output(mac_out), .ready_out(mac_ready), .running_out(mac_running));
                   
    afb af_unit(.activation(afb_in), .acti_result(afb_out));
    
    always @ (posedge clk) begin
        if (reset | next_neuron_input) begin
            fsm_state <= 0;
            afb_out_0 <= 0;
            afb_out_1 <= 0;
            afb_out_2 <= 0;
            afb_out_3 <= 0;
            afb_in <= 0;
            neuron_ready <= 0;
            neuron_running <= 0;
            next_mac_input <= 1;
        end else begin
            case(fsm_state)
                3'b000: begin
                    neuron_running <= 1;
                    if (!mac_running & !mac_ready) begin
                        i_activation_mac_0 <= i_activation_0;
                        i_activation_mac_1 <= i_activation_1;
                        i_activation_mac_2 <= i_activation_2;
                        i_activation_mac_3 <= i_activation_3;
                        i_activation_mac_4 <= i_activation_5;
                        i_activation_mac_5 <= i_activation_6;
                        i_activation_mac_6 <= i_activation_7;
                        i_activation_mac_7 <= i_activation_8;
                        i_activation_mac_8 <= i_activation_10;
                        i_activation_mac_9 <= i_activation_11;
                        i_activation_mac_10 <= i_activation_12;
                        i_activation_mac_11 <= i_activation_13;
                        i_activation_mac_12 <= i_activation_15;
                        i_activation_mac_13 <= i_activation_16;
                        i_activation_mac_14 <= i_activation_17;
                        i_activation_mac_15 <= i_activation_18;
                        next_mac_input <= 0;
                    end else if (!mac_running & mac_ready) begin
                        afb_in <= mac_out[27:11];
                        fsm_state <= 3'b001;
                        next_mac_input <= 1;
                    end
                end
                3'b001: begin
                    if (!mac_running & !mac_ready) begin
                        afb_out_0 <= afb_out;
                        i_activation_mac_0 <= i_activation_1;
                        i_activation_mac_1 <= i_activation_2;
                        i_activation_mac_2 <= i_activation_3;
                        i_activation_mac_3 <= i_activation_4;
                        i_activation_mac_4 <= i_activation_6;
                        i_activation_mac_5 <= i_activation_7;
                        i_activation_mac_6 <= i_activation_8;
                        i_activation_mac_7 <= i_activation_9;
                        i_activation_mac_8 <= i_activation_11;
                        i_activation_mac_9 <= i_activation_12;
                        i_activation_mac_10 <= i_activation_13;
                        i_activation_mac_11 <= i_activation_14;
                        i_activation_mac_12 <= i_activation_16;
                        i_activation_mac_13 <= i_activation_17;
                        i_activation_mac_14 <= i_activation_18;
                        i_activation_mac_15 <= i_activation_19;
                        next_mac_input <= 0;
                    end else if (!mac_running & mac_ready) begin
                        afb_in <= mac_out[27:11];
                        fsm_state <= 3'b010;
                        next_mac_input <= 1;
                    end
                end
                3'b010: begin
                    if (!mac_running & !mac_ready) begin
                        afb_out_1 <= afb_out;
                        i_activation_mac_0 <= i_activation_5;
                        i_activation_mac_1 <= i_activation_6;
                        i_activation_mac_2 <= i_activation_7;
                        i_activation_mac_3 <= i_activation_8;
                        i_activation_mac_4 <= i_activation_10;
                        i_activation_mac_5 <= i_activation_11;
                        i_activation_mac_6 <= i_activation_12;
                        i_activation_mac_7 <= i_activation_13;
                        i_activation_mac_8 <= i_activation_15;
                        i_activation_mac_9 <= i_activation_16;
                        i_activation_mac_10 <= i_activation_17;
                        i_activation_mac_11 <= i_activation_18;
                        i_activation_mac_12 <= i_activation_20;
                        i_activation_mac_13 <= i_activation_21;
                        i_activation_mac_14 <= i_activation_22;
                        i_activation_mac_15 <= i_activation_23;
                        next_mac_input <= 0;
                    end else if (!mac_running & mac_ready) begin
                        afb_in <= mac_out[27:11];
                        fsm_state <= 3'b011;
                        next_mac_input <= 1;
                    end
                end
                3'b011: begin
                    if (!mac_running & !mac_ready) begin
                        afb_out_2 <= afb_out;
                        i_activation_mac_0 <= i_activation_6;
                        i_activation_mac_1 <= i_activation_7;
                        i_activation_mac_2 <= i_activation_8;
                        i_activation_mac_3 <= i_activation_9;
                        i_activation_mac_4 <= i_activation_11;
                        i_activation_mac_5 <= i_activation_12;
                        i_activation_mac_6 <= i_activation_13;
                        i_activation_mac_7 <= i_activation_14;
                        i_activation_mac_8 <= i_activation_16;
                        i_activation_mac_9 <= i_activation_17;
                        i_activation_mac_10 <= i_activation_18;
                        i_activation_mac_11 <= i_activation_19;
                        i_activation_mac_12 <= i_activation_21;
                        i_activation_mac_13 <= i_activation_22;
                        i_activation_mac_14 <= i_activation_23;
                        i_activation_mac_15 <= i_activation_24;
                        next_mac_input <= 0;
                    end else if (!mac_running & mac_ready) begin
                        afb_in <= mac_out[27:11];
                        fsm_state <= 3'b100;
                        next_mac_input <= 1;
                    end
                end
                3'b100: begin
                    afb_out_3 <= afb_out;
                    fsm_state <= 3'b101;
                    neuron_ready <= 1;
                    neuron_running <= 0;
                end
                3'b101: begin
                    neuron_ready <= 0;
                    fsm_state <= 3'b000;
                end
            endcase
        end
    end
    
    assign ready_out = neuron_ready;
    assign running_out = neuron_running;
    assign o_output = ((afb_out_0 >= afb_out_1) & (afb_out_0 >= afb_out_2) & (afb_out_0 >= afb_out_3)) ? afb_out_0 :
                 ((afb_out_1 > afb_out_0) & (afb_out_1 >= afb_out_2) & (afb_out_1 >= afb_out_3)) ? afb_out_1 :
                 ((afb_out_2 > afb_out_0) & (afb_out_2 > afb_out_1) & (afb_out_2 >= afb_out_3)) ? afb_out_2 :
                 afb_out_3;
    
endmodule

module top(
    input wire clk,
    input wire reset,
    input wire signed [11:0] i_activation_0_0,
    input wire signed [11:0] i_activation_0_1,
    input wire signed [11:0] i_activation_0_2,
    input wire signed [11:0] i_activation_0_3,
    input wire signed [11:0] i_activation_0_4,
    input wire signed [11:0] i_activation_1_0,
    input wire signed [11:0] i_activation_1_1,
    input wire signed [11:0] i_activation_1_2,
    input wire signed [11:0] i_activation_1_3,
    input wire signed [11:0] i_activation_1_4,
    input wire signed [11:0] i_activation_2_0,
    input wire signed [11:0] i_activation_2_1,
    input wire signed [11:0] i_activation_2_2,
    input wire signed [11:0] i_activation_2_3,
    input wire signed [11:0] i_activation_2_4,
    input wire signed [11:0] i_activation_3_0,
    input wire signed [11:0] i_activation_3_1,
    input wire signed [11:0] i_activation_3_2,
    input wire signed [11:0] i_activation_3_3,
    input wire signed [11:0] i_activation_3_4,
    input wire signed [11:0] i_activation_4_0,
    input wire signed [11:0] i_activation_4_1,
    input wire signed [11:0] i_activation_4_2,
    input wire signed [11:0] i_activation_4_3,
    input wire signed [11:0] i_activation_4_4,
    input wire signed [11:0] i_weight_0_0_0,
    input wire signed [11:0] i_weight_0_0_1,
    input wire signed [11:0] i_weight_0_0_2,
    input wire signed [11:0] i_weight_0_0_3,
    input wire signed [11:0] i_weight_0_1_0,
    input wire signed [11:0] i_weight_0_1_1,
    input wire signed [11:0] i_weight_0_1_2,
    input wire signed [11:0] i_weight_0_1_3,
    input wire signed [11:0] i_weight_0_2_0,
    input wire signed [11:0] i_weight_0_2_1,
    input wire signed [11:0] i_weight_0_2_2,
    input wire signed [11:0] i_weight_0_2_3,
    input wire signed [11:0] i_weight_0_3_0,
    input wire signed [11:0] i_weight_0_3_1,
    input wire signed [11:0] i_weight_0_3_2,
    input wire signed [11:0] i_weight_0_3_3,
    input wire signed [11:0] i_weight_1_0_0,
    input wire signed [11:0] i_weight_1_0_1,
    input wire signed [11:0] i_weight_1_0_2,
    input wire signed [11:0] i_weight_1_0_3,
    input wire signed [11:0] i_weight_1_1_0,
    input wire signed [11:0] i_weight_1_1_1,
    input wire signed [11:0] i_weight_1_1_2,
    input wire signed [11:0] i_weight_1_1_3,
    input wire signed [11:0] i_weight_1_2_0,
    input wire signed [11:0] i_weight_1_2_1,
    input wire signed [11:0] i_weight_1_2_2,
    input wire signed [11:0] i_weight_1_2_3,
    input wire signed [11:0] i_weight_1_3_0,
    input wire signed [11:0] i_weight_1_3_1,
    input wire signed [11:0] i_weight_1_3_2,
    input wire signed [11:0] i_weight_1_3_3,
    input wire signed [11:0] i_weight_2_0_0,
    input wire signed [11:0] i_weight_2_0_1,
    input wire signed [11:0] i_weight_2_0_2,
    input wire signed [11:0] i_weight_2_0_3,
    input wire signed [11:0] i_weight_2_1_0,
    input wire signed [11:0] i_weight_2_1_1,
    input wire signed [11:0] i_weight_2_1_2,
    input wire signed [11:0] i_weight_2_1_3,
    input wire signed [11:0] i_weight_2_2_0,
    input wire signed [11:0] i_weight_2_2_1,
    input wire signed [11:0] i_weight_2_2_2,
    input wire signed [11:0] i_weight_2_2_3,
    input wire signed [11:0] i_weight_2_3_0,
    input wire signed [11:0] i_weight_2_3_1,
    input wire signed [11:0] i_weight_2_3_2,
    input wire signed [11:0] i_weight_2_3_3,
    input wire signed [11:0] i_weight_3_0_0,
    input wire signed [11:0] i_weight_3_0_1,
    input wire signed [11:0] i_weight_3_0_2,
    input wire signed [11:0] i_weight_3_0_3,
    input wire signed [11:0] i_weight_3_1_0,
    input wire signed [11:0] i_weight_3_1_1,
    input wire signed [11:0] i_weight_3_1_2,
    input wire signed [11:0] i_weight_3_1_3,
    input wire signed [11:0] i_weight_3_2_0,
    input wire signed [11:0] i_weight_3_2_1,
    input wire signed [11:0] i_weight_3_2_2,
    input wire signed [11:0] i_weight_3_2_3,
    input wire signed [11:0] i_weight_3_3_0,
    input wire signed [11:0] i_weight_3_3_1,
    input wire signed [11:0] i_weight_3_3_2,
    input wire signed [11:0] i_weight_3_3_3,
    input wire signed [11:0] i_weight_4_0_0,
    input wire signed [11:0] i_weight_4_0_1,
    input wire signed [11:0] i_weight_4_0_2,
    input wire signed [11:0] i_weight_4_0_3,
    input wire signed [11:0] i_weight_4_1_0,
    input wire signed [11:0] i_weight_4_1_1,
    input wire signed [11:0] i_weight_4_1_2,
    input wire signed [11:0] i_weight_4_1_3,
    input wire signed [11:0] i_weight_4_2_0,
    input wire signed [11:0] i_weight_4_2_1,
    input wire signed [11:0] i_weight_4_2_2,
    input wire signed [11:0] i_weight_4_2_3,
    input wire signed [11:0] i_weight_4_3_0,
    input wire signed [11:0] i_weight_4_3_1,
    input wire signed [11:0] i_weight_4_3_2,
    input wire signed [11:0] i_weight_4_3_3,
    input wire signed [11:0] i_weight_5_0_0,
    input wire signed [11:0] i_weight_5_0_1,
    input wire signed [11:0] i_weight_5_0_2,
    input wire signed [11:0] i_weight_5_0_3,
    input wire signed [11:0] i_weight_5_1_0,
    input wire signed [11:0] i_weight_5_1_1,
    input wire signed [11:0] i_weight_5_1_2,
    input wire signed [11:0] i_weight_5_1_3,
    input wire signed [11:0] i_weight_5_2_0,
    input wire signed [11:0] i_weight_5_2_1,
    input wire signed [11:0] i_weight_5_2_2,
    input wire signed [11:0] i_weight_5_2_3,
    input wire signed [11:0] i_weight_5_3_0,
    input wire signed [11:0] i_weight_5_3_1,
    input wire signed [11:0] i_weight_5_3_2,
    input wire signed [11:0] i_weight_5_3_3,
    input wire signed [11:0] i_weight_6_0_0,
    input wire signed [11:0] i_weight_6_0_1,
    input wire signed [11:0] i_weight_6_0_2,
    input wire signed [11:0] i_weight_6_0_3,
    input wire signed [11:0] i_weight_6_1_0,
    input wire signed [11:0] i_weight_6_1_1,
    input wire signed [11:0] i_weight_6_1_2,
    input wire signed [11:0] i_weight_6_1_3,
    input wire signed [11:0] i_weight_6_2_0,
    input wire signed [11:0] i_weight_6_2_1,
    input wire signed [11:0] i_weight_6_2_2,
    input wire signed [11:0] i_weight_6_2_3,
    input wire signed [11:0] i_weight_6_3_0,
    input wire signed [11:0] i_weight_6_3_1,
    input wire signed [11:0] i_weight_6_3_2,
    input wire signed [11:0] i_weight_6_3_3,
    input wire signed [11:0] i_weight_7_0_0,
    input wire signed [11:0] i_weight_7_0_1,
    input wire signed [11:0] i_weight_7_0_2,
    input wire signed [11:0] i_weight_7_0_3,
    input wire signed [11:0] i_weight_7_1_0,
    input wire signed [11:0] i_weight_7_1_1,
    input wire signed [11:0] i_weight_7_1_2,
    input wire signed [11:0] i_weight_7_1_3,
    input wire signed [11:0] i_weight_7_2_0,
    input wire signed [11:0] i_weight_7_2_1,
    input wire signed [11:0] i_weight_7_2_2,
    input wire signed [11:0] i_weight_7_2_3,
    input wire signed [11:0] i_weight_7_3_0,
    input wire signed [11:0] i_weight_7_3_1,
    input wire signed [11:0] i_weight_7_3_2,
    input wire signed [11:0] i_weight_7_3_3,
    input wire signed [11:0] i_weight_8_0_0,
    input wire signed [11:0] i_weight_8_0_1,
    input wire signed [11:0] i_weight_8_0_2,
    input wire signed [11:0] i_weight_8_0_3,
    input wire signed [11:0] i_weight_8_1_0,
    input wire signed [11:0] i_weight_8_1_1,
    input wire signed [11:0] i_weight_8_1_2,
    input wire signed [11:0] i_weight_8_1_3,
    input wire signed [11:0] i_weight_8_2_0,
    input wire signed [11:0] i_weight_8_2_1,
    input wire signed [11:0] i_weight_8_2_2,
    input wire signed [11:0] i_weight_8_2_3,
    input wire signed [11:0] i_weight_8_3_0,
    input wire signed [11:0] i_weight_8_3_1,
    input wire signed [11:0] i_weight_8_3_2,
    input wire signed [11:0] i_weight_8_3_3,
    input wire signed [11:0] i_weight_9_0_0,
    input wire signed [11:0] i_weight_9_0_1,
    input wire signed [11:0] i_weight_9_0_2,
    input wire signed [11:0] i_weight_9_0_3,
    input wire signed [11:0] i_weight_9_1_0,
    input wire signed [11:0] i_weight_9_1_1,
    input wire signed [11:0] i_weight_9_1_2,
    input wire signed [11:0] i_weight_9_1_3,
    input wire signed [11:0] i_weight_9_2_0,
    input wire signed [11:0] i_weight_9_2_1,
    input wire signed [11:0] i_weight_9_2_2,
    input wire signed [11:0] i_weight_9_2_3,
    input wire signed [11:0] i_weight_9_3_0,
    input wire signed [11:0] i_weight_9_3_1,
    input wire signed [11:0] i_weight_9_3_2,
    input wire signed [11:0] i_weight_9_3_3,
    input wire signed [11:0] i_weight_10_0_0,
    input wire signed [11:0] i_weight_10_0_1,
    input wire signed [11:0] i_weight_10_0_2,
    input wire signed [11:0] i_weight_10_0_3,
    input wire signed [11:0] i_weight_10_1_0,
    input wire signed [11:0] i_weight_10_1_1,
    input wire signed [11:0] i_weight_10_1_2,
    input wire signed [11:0] i_weight_10_1_3,
    input wire signed [11:0] i_weight_10_2_0,
    input wire signed [11:0] i_weight_10_2_1,
    input wire signed [11:0] i_weight_10_2_2,
    input wire signed [11:0] i_weight_10_2_3,
    input wire signed [11:0] i_weight_10_3_0,
    input wire signed [11:0] i_weight_10_3_1,
    input wire signed [11:0] i_weight_10_3_2,
    input wire signed [11:0] i_weight_10_3_3,
    input wire signed [11:0] i_weight_11_0_0,
    input wire signed [11:0] i_weight_11_0_1,
    input wire signed [11:0] i_weight_11_0_2,
    input wire signed [11:0] i_weight_11_0_3,
    input wire signed [11:0] i_weight_11_1_0,
    input wire signed [11:0] i_weight_11_1_1,
    input wire signed [11:0] i_weight_11_1_2,
    input wire signed [11:0] i_weight_11_1_3,
    input wire signed [11:0] i_weight_11_2_0,
    input wire signed [11:0] i_weight_11_2_1,
    input wire signed [11:0] i_weight_11_2_2,
    input wire signed [11:0] i_weight_11_2_3,
    input wire signed [11:0] i_weight_11_3_0,
    input wire signed [11:0] i_weight_11_3_1,
    input wire signed [11:0] i_weight_11_3_2,
    input wire signed [11:0] i_weight_11_3_3,
    input wire signed [11:0] i_weight_12_0_0,
    input wire signed [11:0] i_weight_12_0_1,
    input wire signed [11:0] i_weight_12_0_2,
    input wire signed [11:0] i_weight_12_0_3,
    input wire signed [11:0] i_weight_12_1_0,
    input wire signed [11:0] i_weight_12_1_1,
    input wire signed [11:0] i_weight_12_1_2,
    input wire signed [11:0] i_weight_12_1_3,
    input wire signed [11:0] i_weight_12_2_0,
    input wire signed [11:0] i_weight_12_2_1,
    input wire signed [11:0] i_weight_12_2_2,
    input wire signed [11:0] i_weight_12_2_3,
    input wire signed [11:0] i_weight_12_3_0,
    input wire signed [11:0] i_weight_12_3_1,
    input wire signed [11:0] i_weight_12_3_2,
    input wire signed [11:0] i_weight_12_3_3,
    input wire signed [11:0] i_weight_13_0_0,
    input wire signed [11:0] i_weight_13_0_1,
    input wire signed [11:0] i_weight_13_0_2,
    input wire signed [11:0] i_weight_13_0_3,
    input wire signed [11:0] i_weight_13_1_0,
    input wire signed [11:0] i_weight_13_1_1,
    input wire signed [11:0] i_weight_13_1_2,
    input wire signed [11:0] i_weight_13_1_3,
    input wire signed [11:0] i_weight_13_2_0,
    input wire signed [11:0] i_weight_13_2_1,
    input wire signed [11:0] i_weight_13_2_2,
    input wire signed [11:0] i_weight_13_2_3,
    input wire signed [11:0] i_weight_13_3_0,
    input wire signed [11:0] i_weight_13_3_1,
    input wire signed [11:0] i_weight_13_3_2,
    input wire signed [11:0] i_weight_13_3_3,
    input wire signed [11:0] i_weight_14_0_0,
    input wire signed [11:0] i_weight_14_0_1,
    input wire signed [11:0] i_weight_14_0_2,
    input wire signed [11:0] i_weight_14_0_3,
    input wire signed [11:0] i_weight_14_1_0,
    input wire signed [11:0] i_weight_14_1_1,
    input wire signed [11:0] i_weight_14_1_2,
    input wire signed [11:0] i_weight_14_1_3,
    input wire signed [11:0] i_weight_14_2_0,
    input wire signed [11:0] i_weight_14_2_1,
    input wire signed [11:0] i_weight_14_2_2,
    input wire signed [11:0] i_weight_14_2_3,
    input wire signed [11:0] i_weight_14_3_0,
    input wire signed [11:0] i_weight_14_3_1,
    input wire signed [11:0] i_weight_14_3_2,
    input wire signed [11:0] i_weight_14_3_3,
    input wire signed [11:0] i_weight_15_0_0,
    input wire signed [11:0] i_weight_15_0_1,
    input wire signed [11:0] i_weight_15_0_2,
    input wire signed [11:0] i_weight_15_0_3,
    input wire signed [11:0] i_weight_15_1_0,
    input wire signed [11:0] i_weight_15_1_1,
    input wire signed [11:0] i_weight_15_1_2,
    input wire signed [11:0] i_weight_15_1_3,
    input wire signed [11:0] i_weight_15_2_0,
    input wire signed [11:0] i_weight_15_2_1,
    input wire signed [11:0] i_weight_15_2_2,
    input wire signed [11:0] i_weight_15_2_3,
    input wire signed [11:0] i_weight_15_3_0,
    input wire signed [11:0] i_weight_15_3_1,
    input wire signed [11:0] i_weight_15_3_2,
    input wire signed [11:0] i_weight_15_3_3,
    input wire signed [11:0] i_weight_16_0_0,
    input wire signed [11:0] i_weight_16_0_1,
    input wire signed [11:0] i_weight_16_0_2,
    input wire signed [11:0] i_weight_16_0_3,
    input wire signed [11:0] i_weight_16_1_0,
    input wire signed [11:0] i_weight_16_1_1,
    input wire signed [11:0] i_weight_16_1_2,
    input wire signed [11:0] i_weight_16_1_3,
    input wire signed [11:0] i_weight_16_2_0,
    input wire signed [11:0] i_weight_16_2_1,
    input wire signed [11:0] i_weight_16_2_2,
    input wire signed [11:0] i_weight_16_2_3,
    input wire signed [11:0] i_weight_16_3_0,
    input wire signed [11:0] i_weight_16_3_1,
    input wire signed [11:0] i_weight_16_3_2,
    input wire signed [11:0] i_weight_16_3_3,
    input wire signed [11:0] i_weight_17_0_0,
    input wire signed [11:0] i_weight_17_0_1,
    input wire signed [11:0] i_weight_17_0_2,
    input wire signed [11:0] i_weight_17_0_3,
    input wire signed [11:0] i_weight_17_1_0,
    input wire signed [11:0] i_weight_17_1_1,
    input wire signed [11:0] i_weight_17_1_2,
    input wire signed [11:0] i_weight_17_1_3,
    input wire signed [11:0] i_weight_17_2_0,
    input wire signed [11:0] i_weight_17_2_1,
    input wire signed [11:0] i_weight_17_2_2,
    input wire signed [11:0] i_weight_17_2_3,
    input wire signed [11:0] i_weight_17_3_0,
    input wire signed [11:0] i_weight_17_3_1,
    input wire signed [11:0] i_weight_17_3_2,
    input wire signed [11:0] i_weight_17_3_3,
    input wire signed [11:0] i_weight_18_0_0,
    input wire signed [11:0] i_weight_18_0_1,
    input wire signed [11:0] i_weight_18_0_2,
    input wire signed [11:0] i_weight_18_0_3,
    input wire signed [11:0] i_weight_18_1_0,
    input wire signed [11:0] i_weight_18_1_1,
    input wire signed [11:0] i_weight_18_1_2,
    input wire signed [11:0] i_weight_18_1_3,
    input wire signed [11:0] i_weight_18_2_0,
    input wire signed [11:0] i_weight_18_2_1,
    input wire signed [11:0] i_weight_18_2_2,
    input wire signed [11:0] i_weight_18_2_3,
    input wire signed [11:0] i_weight_18_3_0,
    input wire signed [11:0] i_weight_18_3_1,
    input wire signed [11:0] i_weight_18_3_2,
    input wire signed [11:0] i_weight_18_3_3,
    input wire signed [11:0] i_weight_19_0_0,
    input wire signed [11:0] i_weight_19_0_1,
    input wire signed [11:0] i_weight_19_0_2,
    input wire signed [11:0] i_weight_19_0_3,
    input wire signed [11:0] i_weight_19_1_0,
    input wire signed [11:0] i_weight_19_1_1,
    input wire signed [11:0] i_weight_19_1_2,
    input wire signed [11:0] i_weight_19_1_3,
    input wire signed [11:0] i_weight_19_2_0,
    input wire signed [11:0] i_weight_19_2_1,
    input wire signed [11:0] i_weight_19_2_2,
    input wire signed [11:0] i_weight_19_2_3,
    input wire signed [11:0] i_weight_19_3_0,
    input wire signed [11:0] i_weight_19_3_1,
    input wire signed [11:0] i_weight_19_3_2,
    input wire signed [11:0] i_weight_19_3_3,
    output reg signed [11:0] o_output_0,
    output reg signed [11:0] o_output_1,
    output reg signed [11:0] o_output_2,
    output reg signed [11:0] o_output_3,
    output reg signed [11:0] o_output_4,
    output reg signed [11:0] o_output_5,
    output reg signed [11:0] o_output_6,
    output reg signed [11:0] o_output_7,
    output reg signed [11:0] o_output_8,
    output reg signed [11:0] o_output_9,
    output reg signed [11:0] o_output_10,
    output reg signed [11:0] o_output_11,
    output reg signed [11:0] o_output_12,
    output reg signed [11:0] o_output_13,
    output reg signed [11:0] o_output_14,
    output reg signed [11:0] o_output_15,
    output reg signed [11:0] o_output_16,
    output reg signed [11:0] o_output_17,
    output reg signed [11:0] o_output_18,
    output reg signed [11:0] o_output_19,
	output reg hardware_sig
    );
    
    
    reg [4:0] fsm_state;
    reg next_neuron_input;
    wire neuron_ready;
    wire neuron_running;
    wire signed [11:0] neuron_out;
    
    reg signed [11:0] i_neuron_weight_0_0, i_neuron_weight_0_1, i_neuron_weight_0_2, i_neuron_weight_0_3,
                      i_neuron_weight_1_0, i_neuron_weight_1_1, i_neuron_weight_1_2, i_neuron_weight_1_3, 
                      i_neuron_weight_2_0, i_neuron_weight_2_1, i_neuron_weight_2_2, i_neuron_weight_2_3,
                      i_neuron_weight_3_0, i_neuron_weight_3_1, i_neuron_weight_3_2, i_neuron_weight_3_3;
                      
    neuron_top neuron(.clk(clk), .reset(reset), .next_neuron_input(next_neuron_input), .i_activation_0(i_activation_0_0), 
                      .i_activation_1(i_activation_0_1), .i_activation_2(i_activation_0_2), .i_activation_3(i_activation_0_3), .i_activation_4(i_activation_0_4), 
                      .i_activation_5(i_activation_1_0), .i_activation_6(i_activation_1_1), .i_activation_7(i_activation_1_2), .i_activation_8(i_activation_1_3), 
                      .i_activation_9(i_activation_1_4), .i_activation_10(i_activation_2_0), .i_activation_11(i_activation_2_1), .i_activation_12(i_activation_2_2), 
                      .i_activation_13(i_activation_2_3), .i_activation_14(i_activation_2_4), .i_activation_15(i_activation_3_0), .i_activation_16(i_activation_3_1), 
                      .i_activation_17(i_activation_3_2), .i_activation_18(i_activation_3_3), .i_activation_19(i_activation_3_4), .i_activation_20(i_activation_4_0), 
                      .i_activation_21(i_activation_4_1), .i_activation_22(i_activation_4_2), .i_activation_23(i_activation_4_3), .i_activation_24(i_activation_4_4),
                      .i_weight_0(i_neuron_weight_0_0), .i_weight_1(i_neuron_weight_0_1), .i_weight_2(i_neuron_weight_0_2), .i_weight_3(i_neuron_weight_0_3),
                      .i_weight_4(i_neuron_weight_1_0), .i_weight_5(i_neuron_weight_1_1), .i_weight_6(i_neuron_weight_1_2), .i_weight_7(i_neuron_weight_1_3),
                      .i_weight_8(i_neuron_weight_2_0), .i_weight_9(i_neuron_weight_2_1), .i_weight_10(i_neuron_weight_2_2), .i_weight_11(i_neuron_weight_2_3),
                      .i_weight_12(i_neuron_weight_3_0), .i_weight_13(i_neuron_weight_3_1), .i_weight_14(i_neuron_weight_3_2), .i_weight_15(i_neuron_weight_3_3),
                      .o_output(neuron_out), .ready_out(neuron_ready), .running_out(neuron_running));
    
    always @ (posedge clk) begin
        if (reset) begin
            fsm_state <= 0;
            next_neuron_input <= 1;
			o_output_0 <= 0;
            o_output_1 <= 0; 
            o_output_2 <= 0; 
            o_output_3 <= 0; 
            o_output_4 <= 0; 
            o_output_5 <= 0; 
            o_output_6 <= 0; 
            o_output_7 <= 0; 
            o_output_8 <= 0; 
            o_output_9 <= 0; 
            o_output_10 <= 0; 
            o_output_11 <= 0; 
            o_output_12 <= 0; 
            o_output_13 <= 0; 
            o_output_14 <= 0; 
            o_output_15 <= 0; 
            o_output_16 <= 0; 
            o_output_17 <= 0; 
            o_output_18 <= 0; 
            o_output_19 <= 0;
			hardware_sig <= 0;

			
        end else begin
            case(fsm_state)
                5'b00000: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_0_0_0;
                        i_neuron_weight_0_1 <= i_weight_0_0_1;
                        i_neuron_weight_0_2 <= i_weight_0_0_2;
                        i_neuron_weight_0_3 <= i_weight_0_0_3;
                        i_neuron_weight_1_0 <= i_weight_0_1_0;
                        i_neuron_weight_1_1 <= i_weight_0_1_1;
                        i_neuron_weight_1_2 <= i_weight_0_1_2;
                        i_neuron_weight_1_3 <= i_weight_0_1_3;
                        i_neuron_weight_2_0 <= i_weight_0_2_0;
                        i_neuron_weight_2_1 <= i_weight_0_2_1;
                        i_neuron_weight_2_2 <= i_weight_0_2_2;
                        i_neuron_weight_2_3 <= i_weight_0_2_3;
                        i_neuron_weight_3_0 <= i_weight_0_3_0;
                        i_neuron_weight_3_1 <= i_weight_0_3_1;
                        i_neuron_weight_3_2 <= i_weight_0_3_2;
                        i_neuron_weight_3_3 <= i_weight_0_3_3;
                        next_neuron_input <= 0;
					end else if (!neuron_running & neuron_ready) begin
                        o_output_0 <= neuron_out;
                        fsm_state <= 5'b00001;
                        next_neuron_input <= 1;
                    end
                end
                5'b00001: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_1_0_0;
                        i_neuron_weight_0_1 <= i_weight_1_0_1;
                        i_neuron_weight_0_2 <= i_weight_1_0_2;
                        i_neuron_weight_0_3 <= i_weight_1_0_3;
                        i_neuron_weight_1_0 <= i_weight_1_1_0;
                        i_neuron_weight_1_1 <= i_weight_1_1_1;
                        i_neuron_weight_1_2 <= i_weight_1_1_2;
                        i_neuron_weight_1_3 <= i_weight_1_1_3;
                        i_neuron_weight_2_0 <= i_weight_1_2_0;
                        i_neuron_weight_2_1 <= i_weight_1_2_1;
                        i_neuron_weight_2_2 <= i_weight_1_2_2;
                        i_neuron_weight_2_3 <= i_weight_1_2_3;
                        i_neuron_weight_3_0 <= i_weight_1_3_0;
                        i_neuron_weight_3_1 <= i_weight_1_3_1;
                        i_neuron_weight_3_2 <= i_weight_1_3_2;
                        i_neuron_weight_3_3 <= i_weight_1_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_1 <= neuron_out;
                        fsm_state <= 5'b00010;
                        next_neuron_input <= 1;
                    end
                end         
                5'b00010: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_2_0_0;
                        i_neuron_weight_0_1 <= i_weight_2_0_1;
                        i_neuron_weight_0_2 <= i_weight_2_0_2;
                        i_neuron_weight_0_3 <= i_weight_2_0_3;
                        i_neuron_weight_1_0 <= i_weight_2_1_0;
                        i_neuron_weight_1_1 <= i_weight_2_1_1;
                        i_neuron_weight_1_2 <= i_weight_2_1_2;
                        i_neuron_weight_1_3 <= i_weight_2_1_3;
                        i_neuron_weight_2_0 <= i_weight_2_2_0;
                        i_neuron_weight_2_1 <= i_weight_2_2_1;
                        i_neuron_weight_2_2 <= i_weight_2_2_2;
                        i_neuron_weight_2_3 <= i_weight_2_2_3;
                        i_neuron_weight_3_0 <= i_weight_2_3_0;
                        i_neuron_weight_3_1 <= i_weight_2_3_1;
                        i_neuron_weight_3_2 <= i_weight_2_3_2;
                        i_neuron_weight_3_3 <= i_weight_2_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_2 <= neuron_out;
                        fsm_state <= 5'b00011;
                        next_neuron_input <= 1;
                    end
                end
                5'b00011: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_3_0_0;
                        i_neuron_weight_0_1 <= i_weight_3_0_1;
                        i_neuron_weight_0_2 <= i_weight_3_0_2;
                        i_neuron_weight_0_3 <= i_weight_3_0_3;
                        i_neuron_weight_1_0 <= i_weight_3_1_0;
                        i_neuron_weight_1_1 <= i_weight_3_1_1;
                        i_neuron_weight_1_2 <= i_weight_3_1_2;
                        i_neuron_weight_1_3 <= i_weight_3_1_3;
                        i_neuron_weight_2_0 <= i_weight_3_2_0;
                        i_neuron_weight_2_1 <= i_weight_3_2_1;
                        i_neuron_weight_2_2 <= i_weight_3_2_2;
                        i_neuron_weight_2_3 <= i_weight_3_2_3;
                        i_neuron_weight_3_0 <= i_weight_3_3_0;
                        i_neuron_weight_3_1 <= i_weight_3_3_1;
                        i_neuron_weight_3_2 <= i_weight_3_3_2;
                        i_neuron_weight_3_3 <= i_weight_3_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_3 <= neuron_out;
                        fsm_state <= 5'b00100;
                        next_neuron_input <= 1;
                    end
                end
                5'b00100: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_4_0_0;
                        i_neuron_weight_0_1 <= i_weight_4_0_1;
                        i_neuron_weight_0_2 <= i_weight_4_0_2;
                        i_neuron_weight_0_3 <= i_weight_4_0_3;
                        i_neuron_weight_1_0 <= i_weight_4_1_0;
                        i_neuron_weight_1_1 <= i_weight_4_1_1;
                        i_neuron_weight_1_2 <= i_weight_4_1_2;
                        i_neuron_weight_1_3 <= i_weight_4_1_3;
                        i_neuron_weight_2_0 <= i_weight_4_2_0;
                        i_neuron_weight_2_1 <= i_weight_4_2_1;
                        i_neuron_weight_2_2 <= i_weight_4_2_2;
                        i_neuron_weight_2_3 <= i_weight_4_2_3;
                        i_neuron_weight_3_0 <= i_weight_4_3_0;
                        i_neuron_weight_3_1 <= i_weight_4_3_1;
                        i_neuron_weight_3_2 <= i_weight_4_3_2;
                        i_neuron_weight_3_3 <= i_weight_4_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_4 <= neuron_out;
                        fsm_state <= 5'b00101;
                        next_neuron_input <= 1;
                    end
                end
                5'b00101: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_5_0_0;
                        i_neuron_weight_0_1 <= i_weight_5_0_1;
                        i_neuron_weight_0_2 <= i_weight_5_0_2;
                        i_neuron_weight_0_3 <= i_weight_5_0_3;
                        i_neuron_weight_1_0 <= i_weight_5_1_0;
                        i_neuron_weight_1_1 <= i_weight_5_1_1;
                        i_neuron_weight_1_2 <= i_weight_5_1_2;
                        i_neuron_weight_1_3 <= i_weight_5_1_3;
                        i_neuron_weight_2_0 <= i_weight_5_2_0;
                        i_neuron_weight_2_1 <= i_weight_5_2_1;
                        i_neuron_weight_2_2 <= i_weight_5_2_2;
                        i_neuron_weight_2_3 <= i_weight_5_2_3;
                        i_neuron_weight_3_0 <= i_weight_5_3_0;
                        i_neuron_weight_3_1 <= i_weight_5_3_1;
                        i_neuron_weight_3_2 <= i_weight_5_3_2;
                        i_neuron_weight_3_3 <= i_weight_5_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_5 <= neuron_out;
                        fsm_state <= 5'b00110;
                        next_neuron_input <= 1;
                    end
                end
                5'b00110: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_6_0_0;
                        i_neuron_weight_0_1 <= i_weight_6_0_1;
                        i_neuron_weight_0_2 <= i_weight_6_0_2;
                        i_neuron_weight_0_3 <= i_weight_6_0_3;
                        i_neuron_weight_1_0 <= i_weight_6_1_0;
                        i_neuron_weight_1_1 <= i_weight_6_1_1;
                        i_neuron_weight_1_2 <= i_weight_6_1_2;
                        i_neuron_weight_1_3 <= i_weight_6_1_3;
                        i_neuron_weight_2_0 <= i_weight_6_2_0;
                        i_neuron_weight_2_1 <= i_weight_6_2_1;
                        i_neuron_weight_2_2 <= i_weight_6_2_2;
                        i_neuron_weight_2_3 <= i_weight_6_2_3;
                        i_neuron_weight_3_0 <= i_weight_6_3_0;
                        i_neuron_weight_3_1 <= i_weight_6_3_1;
                        i_neuron_weight_3_2 <= i_weight_6_3_2;
                        i_neuron_weight_3_3 <= i_weight_6_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_6 <= neuron_out;
                        fsm_state <= 5'b00111;
                        next_neuron_input <= 1;
                    end
                end
                5'b00111: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_7_0_0;
                        i_neuron_weight_0_1 <= i_weight_7_0_1;
                        i_neuron_weight_0_2 <= i_weight_7_0_2;
                        i_neuron_weight_0_3 <= i_weight_7_0_3;
                        i_neuron_weight_1_0 <= i_weight_7_1_0;
                        i_neuron_weight_1_1 <= i_weight_7_1_1;
                        i_neuron_weight_1_2 <= i_weight_7_1_2;
                        i_neuron_weight_1_3 <= i_weight_7_1_3;
                        i_neuron_weight_2_0 <= i_weight_7_2_0;
                        i_neuron_weight_2_1 <= i_weight_7_2_1;
                        i_neuron_weight_2_2 <= i_weight_7_2_2;
                        i_neuron_weight_2_3 <= i_weight_7_2_3;
                        i_neuron_weight_3_0 <= i_weight_7_3_0;
                        i_neuron_weight_3_1 <= i_weight_7_3_1;
                        i_neuron_weight_3_2 <= i_weight_7_3_2;
                        i_neuron_weight_3_3 <= i_weight_7_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_7 <= neuron_out;
                        fsm_state <= 5'b01000;
                        next_neuron_input <= 1;
                    end
                end
                5'b01000: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_8_0_0;
                        i_neuron_weight_0_1 <= i_weight_8_0_1;
                        i_neuron_weight_0_2 <= i_weight_8_0_2;
                        i_neuron_weight_0_3 <= i_weight_8_0_3;
                        i_neuron_weight_1_0 <= i_weight_8_1_0;
                        i_neuron_weight_1_1 <= i_weight_8_1_1;
                        i_neuron_weight_1_2 <= i_weight_8_1_2;
                        i_neuron_weight_1_3 <= i_weight_8_1_3;
                        i_neuron_weight_2_0 <= i_weight_8_2_0;
                        i_neuron_weight_2_1 <= i_weight_8_2_1;
                        i_neuron_weight_2_2 <= i_weight_8_2_2;
                        i_neuron_weight_2_3 <= i_weight_8_2_3;
                        i_neuron_weight_3_0 <= i_weight_8_3_0;
                        i_neuron_weight_3_1 <= i_weight_8_3_1;
                        i_neuron_weight_3_2 <= i_weight_8_3_2;
                        i_neuron_weight_3_3 <= i_weight_8_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_8 <= neuron_out;
                        fsm_state <= 5'b01001;
                        next_neuron_input <= 1;
                    end
                end
                5'b01001: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_9_0_0;
                        i_neuron_weight_0_1 <= i_weight_9_0_1;
                        i_neuron_weight_0_2 <= i_weight_9_0_2;
                        i_neuron_weight_0_3 <= i_weight_9_0_3;
                        i_neuron_weight_1_0 <= i_weight_9_1_0;
                        i_neuron_weight_1_1 <= i_weight_9_1_1;
                        i_neuron_weight_1_2 <= i_weight_9_1_2;
                        i_neuron_weight_1_3 <= i_weight_9_1_3;
                        i_neuron_weight_2_0 <= i_weight_9_2_0;
                        i_neuron_weight_2_1 <= i_weight_9_2_1;
                        i_neuron_weight_2_2 <= i_weight_9_2_2;
                        i_neuron_weight_2_3 <= i_weight_9_2_3;
                        i_neuron_weight_3_0 <= i_weight_9_3_0;
                        i_neuron_weight_3_1 <= i_weight_9_3_1;
                        i_neuron_weight_3_2 <= i_weight_9_3_2;
                        i_neuron_weight_3_3 <= i_weight_9_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_9 <= neuron_out;
                        fsm_state <= 5'b01010;
                        next_neuron_input <= 1;
                    end
                end
                5'b01010: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_10_0_0;
                        i_neuron_weight_0_1 <= i_weight_10_0_1;
                        i_neuron_weight_0_2 <= i_weight_10_0_2;
                        i_neuron_weight_0_3 <= i_weight_10_0_3;
                        i_neuron_weight_1_0 <= i_weight_10_1_0;
                        i_neuron_weight_1_1 <= i_weight_10_1_1;
                        i_neuron_weight_1_2 <= i_weight_10_1_2;
                        i_neuron_weight_1_3 <= i_weight_10_1_3;
                        i_neuron_weight_2_0 <= i_weight_10_2_0;
                        i_neuron_weight_2_1 <= i_weight_10_2_1;
                        i_neuron_weight_2_2 <= i_weight_10_2_2;
                        i_neuron_weight_2_3 <= i_weight_10_2_3;
                        i_neuron_weight_3_0 <= i_weight_10_3_0;
                        i_neuron_weight_3_1 <= i_weight_10_3_1;
                        i_neuron_weight_3_2 <= i_weight_10_3_2;
                        i_neuron_weight_3_3 <= i_weight_10_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_10 <= neuron_out;
                        fsm_state <= 5'b01011;
                        next_neuron_input <= 1;
                    end
                end
                5'b01011: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_11_0_0;
                        i_neuron_weight_0_1 <= i_weight_11_0_1;
                        i_neuron_weight_0_2 <= i_weight_11_0_2;
                        i_neuron_weight_0_3 <= i_weight_11_0_3;
                        i_neuron_weight_1_0 <= i_weight_11_1_0;
                        i_neuron_weight_1_1 <= i_weight_11_1_1;
                        i_neuron_weight_1_2 <= i_weight_11_1_2;
                        i_neuron_weight_1_3 <= i_weight_11_1_3;
                        i_neuron_weight_2_0 <= i_weight_11_2_0;
                        i_neuron_weight_2_1 <= i_weight_11_2_1;
                        i_neuron_weight_2_2 <= i_weight_11_2_2;
                        i_neuron_weight_2_3 <= i_weight_11_2_3;
                        i_neuron_weight_3_0 <= i_weight_11_3_0;
                        i_neuron_weight_3_1 <= i_weight_11_3_1;
                        i_neuron_weight_3_2 <= i_weight_11_3_2;
                        i_neuron_weight_3_3 <= i_weight_11_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_11 <= neuron_out;
                        fsm_state <= 5'b01100;
                        next_neuron_input <= 1;
                    end
                end
                5'b01100: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_12_0_0;
                        i_neuron_weight_0_1 <= i_weight_12_0_1;
                        i_neuron_weight_0_2 <= i_weight_12_0_2;
                        i_neuron_weight_0_3 <= i_weight_12_0_3;
                        i_neuron_weight_1_0 <= i_weight_12_1_0;
                        i_neuron_weight_1_1 <= i_weight_12_1_1;
                        i_neuron_weight_1_2 <= i_weight_12_1_2;
                        i_neuron_weight_1_3 <= i_weight_12_1_3;
                        i_neuron_weight_2_0 <= i_weight_12_2_0;
                        i_neuron_weight_2_1 <= i_weight_12_2_1;
                        i_neuron_weight_2_2 <= i_weight_12_2_2;
                        i_neuron_weight_2_3 <= i_weight_12_2_3;
                        i_neuron_weight_3_0 <= i_weight_12_3_0;
                        i_neuron_weight_3_1 <= i_weight_12_3_1;
                        i_neuron_weight_3_2 <= i_weight_12_3_2;
                        i_neuron_weight_3_3 <= i_weight_12_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_12 <= neuron_out;
                        fsm_state <= 5'b01101;
                        next_neuron_input <= 1;
                    end
                end
                5'b01101: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_13_0_0;
                        i_neuron_weight_0_1 <= i_weight_13_0_1;
                        i_neuron_weight_0_2 <= i_weight_13_0_2;
                        i_neuron_weight_0_3 <= i_weight_13_0_3;
                        i_neuron_weight_1_0 <= i_weight_13_1_0;
                        i_neuron_weight_1_1 <= i_weight_13_1_1;
                        i_neuron_weight_1_2 <= i_weight_13_1_2;
                        i_neuron_weight_1_3 <= i_weight_13_1_3;
                        i_neuron_weight_2_0 <= i_weight_13_2_0;
                        i_neuron_weight_2_1 <= i_weight_13_2_1;
                        i_neuron_weight_2_2 <= i_weight_13_2_2;
                        i_neuron_weight_2_3 <= i_weight_13_2_3;
                        i_neuron_weight_3_0 <= i_weight_13_3_0;
                        i_neuron_weight_3_1 <= i_weight_13_3_1;
                        i_neuron_weight_3_2 <= i_weight_13_3_2;
                        i_neuron_weight_3_3 <= i_weight_13_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_13 <= neuron_out;
                        fsm_state <= 5'b01110;
                        next_neuron_input <= 1;
                    end
                end
                5'b01110: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_14_0_0;
                        i_neuron_weight_0_1 <= i_weight_14_0_1;
                        i_neuron_weight_0_2 <= i_weight_14_0_2;
                        i_neuron_weight_0_3 <= i_weight_14_0_3;
                        i_neuron_weight_1_0 <= i_weight_14_1_0;
                        i_neuron_weight_1_1 <= i_weight_14_1_1;
                        i_neuron_weight_1_2 <= i_weight_14_1_2;
                        i_neuron_weight_1_3 <= i_weight_14_1_3;
                        i_neuron_weight_2_0 <= i_weight_14_2_0;
                        i_neuron_weight_2_1 <= i_weight_14_2_1;
                        i_neuron_weight_2_2 <= i_weight_14_2_2;
                        i_neuron_weight_2_3 <= i_weight_14_2_3;
                        i_neuron_weight_3_0 <= i_weight_14_3_0;
                        i_neuron_weight_3_1 <= i_weight_14_3_1;
                        i_neuron_weight_3_2 <= i_weight_14_3_2;
                        i_neuron_weight_3_3 <= i_weight_14_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_14 <= neuron_out;
                        fsm_state <= 5'b01111;
                        next_neuron_input <= 1;
                    end
                end
                5'b01111: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_15_0_0;
                        i_neuron_weight_0_1 <= i_weight_15_0_1;
                        i_neuron_weight_0_2 <= i_weight_15_0_2;
                        i_neuron_weight_0_3 <= i_weight_15_0_3;
                        i_neuron_weight_1_0 <= i_weight_15_1_0;
                        i_neuron_weight_1_1 <= i_weight_15_1_1;
                        i_neuron_weight_1_2 <= i_weight_15_1_2;
                        i_neuron_weight_1_3 <= i_weight_15_1_3;
                        i_neuron_weight_2_0 <= i_weight_15_2_0;
                        i_neuron_weight_2_1 <= i_weight_15_2_1;
                        i_neuron_weight_2_2 <= i_weight_15_2_2;
                        i_neuron_weight_2_3 <= i_weight_15_2_3;
                        i_neuron_weight_3_0 <= i_weight_15_3_0;
                        i_neuron_weight_3_1 <= i_weight_15_3_1;
                        i_neuron_weight_3_2 <= i_weight_15_3_2;
                        i_neuron_weight_3_3 <= i_weight_15_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_15 <= neuron_out;
                        fsm_state <= 5'b10000;
                        next_neuron_input <= 1;
                    end
                end
                5'b10000: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_16_0_0;
                        i_neuron_weight_0_1 <= i_weight_16_0_1;
                        i_neuron_weight_0_2 <= i_weight_16_0_2;
                        i_neuron_weight_0_3 <= i_weight_16_0_3;
                        i_neuron_weight_1_0 <= i_weight_16_1_0;
                        i_neuron_weight_1_1 <= i_weight_16_1_1;
                        i_neuron_weight_1_2 <= i_weight_16_1_2;
                        i_neuron_weight_1_3 <= i_weight_16_1_3;
                        i_neuron_weight_2_0 <= i_weight_16_2_0;
                        i_neuron_weight_2_1 <= i_weight_16_2_1;
                        i_neuron_weight_2_2 <= i_weight_16_2_2;
                        i_neuron_weight_2_3 <= i_weight_16_2_3;
                        i_neuron_weight_3_0 <= i_weight_16_3_0;
                        i_neuron_weight_3_1 <= i_weight_16_3_1;
                        i_neuron_weight_3_2 <= i_weight_16_3_2;
                        i_neuron_weight_3_3 <= i_weight_16_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_16 <= neuron_out;
                        fsm_state <= 5'b10001;
                        next_neuron_input <= 1;
                    end
                end
                5'b10001: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_17_0_0;
                        i_neuron_weight_0_1 <= i_weight_17_0_1;
                        i_neuron_weight_0_2 <= i_weight_17_0_2;
                        i_neuron_weight_0_3 <= i_weight_17_0_3;
                        i_neuron_weight_1_0 <= i_weight_17_1_0;
                        i_neuron_weight_1_1 <= i_weight_17_1_1;
                        i_neuron_weight_1_2 <= i_weight_17_1_2;
                        i_neuron_weight_1_3 <= i_weight_17_1_3;
                        i_neuron_weight_2_0 <= i_weight_17_2_0;
                        i_neuron_weight_2_1 <= i_weight_17_2_1;
                        i_neuron_weight_2_2 <= i_weight_17_2_2;
                        i_neuron_weight_2_3 <= i_weight_17_2_3;
                        i_neuron_weight_3_0 <= i_weight_17_3_0;
                        i_neuron_weight_3_1 <= i_weight_17_3_1;
                        i_neuron_weight_3_2 <= i_weight_17_3_2;
                        i_neuron_weight_3_3 <= i_weight_17_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_17 <= neuron_out;
                        fsm_state <= 5'b10010;
                        next_neuron_input <= 1;
                    end
                end
                5'b10010: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_18_0_0;
                        i_neuron_weight_0_1 <= i_weight_18_0_1;
                        i_neuron_weight_0_2 <= i_weight_18_0_2;
                        i_neuron_weight_0_3 <= i_weight_18_0_3;
                        i_neuron_weight_1_0 <= i_weight_18_1_0;
                        i_neuron_weight_1_1 <= i_weight_18_1_1;
                        i_neuron_weight_1_2 <= i_weight_18_1_2;
                        i_neuron_weight_1_3 <= i_weight_18_1_3;
                        i_neuron_weight_2_0 <= i_weight_18_2_0;
                        i_neuron_weight_2_1 <= i_weight_18_2_1;
                        i_neuron_weight_2_2 <= i_weight_18_2_2;
                        i_neuron_weight_2_3 <= i_weight_18_2_3;
                        i_neuron_weight_3_0 <= i_weight_18_3_0;
                        i_neuron_weight_3_1 <= i_weight_18_3_1;
                        i_neuron_weight_3_2 <= i_weight_18_3_2;
                        i_neuron_weight_3_3 <= i_weight_18_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_18 <= neuron_out;
                        fsm_state <= 5'b10011;
                        next_neuron_input <= 1;
                    end
                end
                5'b10011: begin
                    if (!neuron_running & !neuron_ready) begin
                        i_neuron_weight_0_0 <= i_weight_19_0_0;
                        i_neuron_weight_0_1 <= i_weight_19_0_1;
                        i_neuron_weight_0_2 <= i_weight_19_0_2;
                        i_neuron_weight_0_3 <= i_weight_19_0_3;
                        i_neuron_weight_1_0 <= i_weight_19_1_0;
                        i_neuron_weight_1_1 <= i_weight_19_1_1;
                        i_neuron_weight_1_2 <= i_weight_19_1_2;
                        i_neuron_weight_1_3 <= i_weight_19_1_3;
                        i_neuron_weight_2_0 <= i_weight_19_2_0;
                        i_neuron_weight_2_1 <= i_weight_19_2_1;
                        i_neuron_weight_2_2 <= i_weight_19_2_2;
                        i_neuron_weight_2_3 <= i_weight_19_2_3;
                        i_neuron_weight_3_0 <= i_weight_19_3_0;
                        i_neuron_weight_3_1 <= i_weight_19_3_1;
                        i_neuron_weight_3_2 <= i_weight_19_3_2;
                        i_neuron_weight_3_3 <= i_weight_19_3_3;
                        next_neuron_input <= 0;
                    end else if (!neuron_running & neuron_ready) begin
                        o_output_19 <= neuron_out;
                        fsm_state <= 5'b10100;
						
                        
						
                    end
                end
                5'b10100: begin
                    fsm_state <= 5'b00000;
					hardware_sig <= 1;
					
                end
            endcase
        end
    end
    
endmodule
