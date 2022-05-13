#include <iostream>
#include "defines.h"
#include "fpgaDefines.h"
#include "netFPGA.h"
#include <chrono>
#include <thread>
#include <iomanip>
#include <ctime>  


using namespace std;
using namespace chrono;


int main(){

    fpga::fpga_handler the_handler;
    the_handler.activate_handler();

    for(size_t n=1;n<11;n++){
    cout << "N: " <<n<<"\n";
    int n_nets = 4;

    size_t n_ins = 16;
    size_t base = 160;
    std::vector<size_t> n_p_l = {n*base,n*base,n*base,n*base,n*base};
    std::vector<int> activation_type = {net::RELU2,net::RELU2,net::RELU2,net::RELU2,net::RELU2};
    
    std::vector<fpga::net_fpga> nets;
    nets.reserve(n_nets);
    for(int i=0;i<n_nets;i++)
        nets.emplace_back(n_ins, n_p_l, activation_type, the_handler);

    fpga::net_fpga my_net(n_ins, n_p_l, activation_type, the_handler);

    std::vector<float> inputs = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f,1.0f, 0.0f, 0.0f, 0.0f, 0.0f,1.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f};
    // auto outputs = my_net.launch_forward(inputs);

    int iterations = 500;
    std::cout << "PRECARGA \n";
    for(int i=0;i<n_nets; i++)
        nets[i].enqueue_net(inputs,true,false,false);
    // my_net.enqueue_net(inputs,true,false,false);

    my_net.solve_pack();

    for(int i=0;i<n_nets; i++)
        nets[i].read_net();
    // my_net.read_net();
    std::cout << "-----------------------------------------------------------------------------------\n\n\n";

    
    auto start = high_resolution_clock::now();
    for(int it=0; it<iterations; it++){
        // std::cout << "ITERATION: "<<it<<"\n";
        for(int i=0;i<n_nets; i++)
            nets[i].enqueue_net(inputs,false,false,false);
        // my_net.enqueue_net(inputs,false,false,false);

        my_net.solve_pack();

        for(int i=0;i<n_nets; i++)
            nets[i].read_net();
        // my_net.read_net();
        // std::cout << "-----------------------------------------------------------------------------------\n";
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Mean exe time: " << duration.count()/n_nets/iterations << "\n";
    // for(int i=0;i<n_nets;i++)
    //     nets[i];
    // for(int i=0; i<outputs.size(); i++)
    //     std::cout << outputs[i] << "\n";
    }
    return 0;
}
