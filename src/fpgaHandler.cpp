#include <fpgaHandler.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"
#include <thread>
#include <chrono>
#include <iomanip>
#include <ctime>    

using namespace std;
using namespace fpga;
using namespace aocl_utils;
using namespace chrono;

bool fpga_handler::there_is_a_handler = false;
fpga_handler* cl_ptr = nullptr;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_program program = NULL;

int inout_net_free_mem;
cl_mem in_out_dev = NULL;
int params_net_free_mem;
cl_mem params_dev = NULL;
int bias_net_free_mem;
cl_mem bias_dev = NULL;   
// cl_mem configuration_dev = NULL;  

void set_args_callback(cl_event event, cl_int event_command_exec_status, void *user_data);

fpga_handler::fpga_handler() : net_list(MAX_SZ_ENQUEUE)
{

    if (there_is_a_handler)
        fpga_error("This FPGA device already has a handler");
    else
    {
        im_the_handler = true;
        there_is_a_handler = true;
        cl_ptr = this;
    }
}

fpga_handler::~fpga_handler()
{
    _cleanup();
}

void fpga_handler::activate_handler()
{
    if (im_the_handler)
        _init_program();
}

int fpga_handler::enqueue_net(fpga_data &in_net, std::vector<long int> &inputs, bool reload, bool big_nets)
{
    int cnt = 0;
    if(nets_enqueued>=MAX_SZ_ENQUEUE){
        fpga_error("Enqueues overflow");
    }else{
        nets_enqueued++;
        bool aux_bool = true;

        while(aux_bool){
            aux_bool = !net_list[cnt].free_slot;
            if(!aux_bool){
                net_list[cnt].free_slot = false;
                net_list[cnt].net = in_net;
                net_list[cnt].inputs = inputs;
                net_list[cnt].reload = reload;
                net_list[cnt].enqueued = true;
                net_list[cnt].big_net = big_nets || in_net.n_ins*sizeof(long int)>INOUT_SIZE/2 || in_net.n_params*sizeof(long int)>PARAMS_SIZE || in_net.n_neurons>BIAS_SIZE;
            }
            cnt++;
            if(cnt==MAX_SZ_ENQUEUE && aux_bool)
                fpga_error("Enqueues counter overflow");
        }
    }
    return cnt;
}

void fpga_handler::solve_nets()
{ 
    vector<int> enq_nets(N_CORES,0);

    bool solv_bool = true;
    int enq_nets_cnt = 0, solve_nets_cnt = 0;
#if fpga_performance == 1
    int enq_time = 0;
    int solve_time = 0;
    int read_time = 0;
    int enq_layer_time = 0;
#endif

    while(solv_bool){
#if fpga_performance == 1
        auto start = high_resolution_clock::now();
#endif
        if(enq_nets_cnt<nets_enqueued){
            for(int c=0; c<N_CORES; c++){
                if(cores[c].kernel_busy == false){
                    enq_nets[c] = enq_nets_cnt;

                    cores[c].enq_inputs(net_list[enq_nets_cnt].inputs, context);
                    net_list[enq_nets_cnt].loaded = true;
                    net_list[enq_nets_cnt].outs = vector<long int>(net_list[enq_nets_cnt].net.n_p_l[net_list[enq_nets_cnt].net.n_layers-1],0);
                    enq_nets_cnt++;
                    if(enq_nets_cnt>=nets_enqueued)
                        goto PROCESS;                
                }
            }
        }

        PROCESS:
#if fpga_performance == 1
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        enq_time += duration.count();
        start = high_resolution_clock::now();
#endif

        for(int c=0; c<N_CORES; c++){
            if(cores[c].kernel_busy == true){
                int ind = enq_nets[c];
                #if fpga_performance == 1
                    auto start2 = high_resolution_clock::now();
                #endif
                if(net_list[ind].layer == net_list[ind].net.n_layers){
                    cores[c].enq_read(net_list[ind].outs);
                    #if fpga_performance == 1
                            auto end2 = high_resolution_clock::now();
                            auto duration2 = duration_cast<microseconds>(end2 - start2);
                            read_time += duration2.count();
                    #endif
                    cores[c].kernel_busy = false;
                    net_list[ind].solved = true;
                    if(net_list[ind].big_net)
                        cores[c].switch_mem_mode(MEM_MODE_BY_LOTS);
                    else
                        cores[c].switch_mem_mode(MEM_MODE_FULL);
                    solve_nets_cnt++;
                }else{
                    cores[c].enq_layer(net_list[ind].net,net_list[ind].layer,net_list[ind].reload);
                    #if fpga_performance == 1
                            auto end2 = high_resolution_clock::now();
                            auto duration2 = duration_cast<microseconds>(end2 - start2);
                            enq_layer_time += duration2.count();
                    #endif
                    net_list[ind].layer++;
                }
            }
        }
#if fpga_performance == 1
        end = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end - start);
        solve_time += duration.count();
#endif

        if(solve_nets_cnt >= nets_enqueued)
            solv_bool = false;
    }
#if fpga_performance == 1
    fpga_info("enq ins performance "<<enq_time<<"us");
    fpga_info("solve total performance "<<solve_time<<"us");
    fpga_info("solve read performance "<<read_time<<"us");
    fpga_info("solve layer performance "<<enq_layer_time<<"us");
#endif
}

std::vector<long int> fpga_handler::read_net(int identifier)
{
    int id = identifier-1;
    net_list[id].free_slot = true;
    net_list[id].enqueued = false;
    net_list[id].loaded = false;
    net_list[id].solved = false;
    net_list[id].layer = 0;
    nets_enqueued--;

    return net_list[id].outs;
}

void fpga_handler::_cleanup()
{
    cout2(YELLOW, "   IN CLEANUP", "");
    if (im_the_handler)
    {
        cout2(YELLOW, "   CLEANING", "");
        if (program){
            clReleaseProgram(program);
            program = nullptr;
        }
        if (context){
            clReleaseContext(context);
            context = nullptr;
        }
        for (int i = 0; i < N_CORES; i++)
            cores[i].kernel_cleanup();
    }
}

void fpga_handler::_init_program()
{    
    // _cleanup();

    cl_int err = 0;
    cout2(YELLOW, "   INIT_PROGRAM", "");
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "Failed to get platforms");

    cout2(YELLOW, "   PLATFORMS", "");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);
    checkError(err, "Failed to find device");

    cout2(YELLOW, "   DEVICES", "");
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "Failed to create context");

    cout2(YELLOW, "   PROGRAM", "");
    std::string binary_file = getBoardBinaryFile(PROGRAM_NAME, device); // Coge el aocx
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
    err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(err, "Failed to create program");

    cores.reserve(N_CORES);
    for(int i=1; i<(N_CORES+1); i++){
        string aux_str(KERNEL_BASE_NAME);
        aux_str.append(to_string(i));
        cores.emplace_back(i, aux_str, program, context, device, INOUT_SIZE, PARAMS_SIZE, BIAS_SIZE);
    }
}

void cleanup()
{
    cl_ptr->_cleanup();
}