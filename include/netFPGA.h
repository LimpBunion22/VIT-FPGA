#ifndef NETFPGA_H
#define NETFPGA_H

#include <netAbstract.h>
// #include <mathStructsCPU.h>
#include <chrono>
#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"


// void cleanup();

namespace fpga
{
    #define NET_KERNEL 0
    #define IMAGE_KERNEL 1

    #define IMAGE_HEIGHT 1080
    #define IMAGE_WIDTH 1920

    #define BATCH_SIZE 24

    class net_fpga : public net::net_abstract
    {    

    public:

        //Net variables
        int n_ins;
        int n_layers;
        int* n_p_l;
        int n_neurons;
        int n_params;

        DATA_TYPE* params;
        int activations;
        DATA_TYPE* bias;

        int n_sets;
        bool gradient_init;

        int64_t gradient_performance;
        int64_t forward_performance;

        //OpenCL & FPGA variables
        static int net_fpga_counter;
        static bool program_init;
        static bool forward_kernel_init;
        static bool reload_params;

        // OpenCL runtime configuration
        static cl_platform_id platform;
        static cl_device_id device;
        static cl_context context;
        static cl_command_queue queue;
        static cl_kernel kernel;
        static cl_program program;
        static cl_int err;

        static cl_mem inputs_dev;
        static cl_mem params_dev;
        static cl_mem bias_dev;
        static cl_mem outs_dev;
        static cl_mem npl_dev;

        static int n_ins_buff;
        static int n_layers_buff;
        static int* n_p_l_buff;

        static DATA_TYPE* params_buff;
        static DATA_TYPE* bias_buff;
        static DATA_TYPE* inputs_buff;
        static DATA_TYPE* oputputs_buff;

        static cl_event init_event;
        static cl_event finish_event;

        static cl_event im_init_event[BATCH_SIZE];
        static cl_event im_finish_event[BATCH_SIZE];
        static cl_event im_read_event[BATCH_SIZE];

        static unsigned char in_images[BATCH_SIZE][IMAGE_HEIGHT*IMAGE_WIDTH];
        static unsigned char out_images[BATCH_SIZE][IMAGE_HEIGHT*IMAGE_WIDTH];
        static int wr_batch_cnt;
        static int rd_batch_cnt;
        static int free_batch;


    private:
        net_fpga() = delete;
        void _init_program(int prg = NET_KERNEL);
        void _init_kernel(const char* kernel_name);
        void _init_kernel(const char* kernel_name, const net::image_set &set);
        void _load_params();
        // void _load_params(const net::image_set &set);

    public:
        ~net_fpga();
        net_fpga(const net::net_data &data, bool derivate, bool random); //* net::net_data como copia para mantener operaciones move
        net_fpga(net_fpga &&rh);
        net_fpga &operator=(net_fpga &&rh);
        net_fpga &operator=(const net_fpga &rh);

        net::net_data get_net_data() override;
        std::vector<DATA_TYPE> launch_forward(const std::vector<DATA_TYPE> &inputs) override;
        void init_gradient(const net::net_sets &sets) override;
        std::vector<DATA_TYPE> launch_gradient(size_t iterations, DATA_TYPE error_threshold, DATA_TYPE multiplier) override;
        void print_inner_vals() override;
        signed long get_gradient_performance() override;
        signed long get_forward_performance() override;
        void filter_image(const net::image_set &set) override;
        net::image_set get_filtered_image() override;

    // public:
    //     friend void cleanup();
    };
}

#endif