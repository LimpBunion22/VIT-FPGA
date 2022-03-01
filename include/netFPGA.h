#ifndef NETFPGA_H
#define NETFPGA_H

#include <netAbstract.h>
#include <chrono>
#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"

//Modos de programa de la FPGA
#define NN 0
#define IMG 1
#define CNN 2

//Programas de la FPGA
#define NOT_LOADED "FPGA_SIN_PROGRAMA"
#define NN_DNN1 "PRG_DNN_V1"
#define IMG_1920x1080 "PRG_IMG_1920x1080"
#define IMG_1000x1000 "PRG_IMG_1000x1000"

//Kernels de los programas
#define NN_DNN1_KERNEL "network_v1"
#define IMG_IN_KERNEL "image_process"
#define IMG_OUT_BORDERS_KERNEL "image_borders"

namespace fpga
{
    class net_fpga : public net::net_abstract
    {
        
    public:

        std::string net_ident;

        //Net variables
        int n_ins;
        int n_layers;
        int *n_p_l;
        int n_neurons;
        int n_params;

        float *params;
        int activations;
        float *bias;

        int n_sets;
        bool gradient_init;

        int64_t gradient_performance;
        int64_t forward_performance;

        
        // void * (*clMapHostPipeIntelFPGA) (cl_mem, cl_map_flags, size_t, size_t *, cl_int *);
        // cl_int (*clUnmapHostPipeIntelFPGA) (cl_mem, void *, size_t, size_t *);
    private:
        // net_fpga() = delete;
        void _init_program(std::string prg_name, int net_kind, int pxl_count = 0);
        void _init_nn_kernels();
        void _init_img_kernels(int pxl_count);
        void _load_params();
        // void _load_params(const net::image_set &set);

    public:
        ~net_fpga();
        net_fpga();
        net_fpga(const net::net_data &data, bool random); //* net::net_data como copia para mantener operaciones move
        net_fpga(net_fpga &&rh);
        net_fpga &operator=(net_fpga &&rh);
        net_fpga &operator=(const net_fpga &rh);

        net::net_data get_net_data() override;
        std::vector<float> launch_forward(const std::vector<float> &inputs) override;
        void init_gradient(const net::net_sets &sets) override;
        std::vector<float> launch_gradient(size_t iterations, float error_threshold, float multiplier) override;
        void print_inner_vals() override;
        signed long get_gradient_performance() override;
        signed long get_forward_performance() override;
        
        //1920x1080
        void process_img_1920_1080(unsigned char* red_image, unsigned char* green_image,unsigned char* blue_image);
        net::image_set get_img_1920_1080();

        //1000x1000
        void process_img_1000_1000(unsigned char* red_image, unsigned char* green_image,unsigned char* blue_image);
        std::vector<float> get_img_1000_1000();

        // public:
        //     friend void cleanup();
    };
}

#endif