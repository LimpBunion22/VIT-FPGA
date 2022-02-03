#ifndef NETFPGA_H
#define NETFPGA_H

#include <netAbstract.h>
#include <chrono>
#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"

namespace fpga
{
#define NET_KERNEL 0
#define IMAGE_KERNEL 1

#define IMAGE_HEIGHT 1080
#define IMAGE_WIDTH 1920    

    class net_fpga : public net::net_abstract
    {
        
    public:
        //Net variables
        int n_ins;
        int n_layers;
        int *n_p_l;
        int n_neurons;
        int n_params;

        DATA_TYPE *params;
        int activations;
        DATA_TYPE *bias;

        int n_sets;
        bool gradient_init;

        int64_t gradient_performance;
        int64_t forward_performance;

        
        // void * (*clMapHostPipeIntelFPGA) (cl_mem, cl_map_flags, size_t, size_t *, cl_int *);
        // cl_int (*clUnmapHostPipeIntelFPGA) (cl_mem, void *, size_t, size_t *);
    private:
        net_fpga() = delete;
        void _init_program(int prg = NET_KERNEL);
        void _init_kernel(const char *kernel_name);
        void _init_kernel(const char *kernel_name, const net::image_set &set);
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