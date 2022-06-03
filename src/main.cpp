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

    for(int n=1;n<11;n++){
    cout << "N: " <<n<<"\n";
    int n_nets = 8;

    int n_ins = 16;
    int base = 160;
    // std::vector<int> n_p_l = {n*base,n*base,n*base,n*base,n*base};
    std::vector<int> n_p_l = {10*base,10*base,10*base,10*base,10*base};
    std::vector<int> activation_type = {net::RELU2,net::RELU2,net::RELU2,net::RELU2,net::RELU2};
    
    std::vector<fpga::net_fpga> nets;
    nets.reserve(n_nets);
    for(int i=0;i<n_nets;i++){
        nets.emplace_back(the_handler);
        nets[i].build_net_from_data(n_ins, n_p_l, activation_type);
    }

    fpga::net_fpga my_net(the_handler);
    my_net.build_net_from_data(n_ins, n_p_l, activation_type);

    std::vector<float> inputs = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f,1.0f, 0.0f, 0.0f, 0.0f, 0.0f,1.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f};
    // auto outputs = my_net.launch_forward(inputs);

    int iterations = 10;
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