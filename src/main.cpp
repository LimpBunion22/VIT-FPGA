#include <iostream>
#include "defines.h"
#include "fpgaDefines.h"
#include "netFPGA.h"

int main(){

    fpga::fpga_handler the_handler;
    the_handler.activate_handler();

    size_t n_ins = 5;
    std::vector<size_t> n_p_l = {1,2};
    std::vector<int> activation_type = {net::RELU2};
    
    fpga::net_fpga my_net(n_ins, n_p_l, activation_type, the_handler);

    std::vector<float> inputs = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto outputs = my_net.launch_forward(inputs);

    for(int i=0; i<outputs.size(); i++)
        std::cout << outputs[i] << "\n";

    return 0;
}
