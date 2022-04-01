#ifndef NETFPGA_H
#define NETFPGA_H

#include <netAbstract.h>
#include <fpgaHandler.h>
#include <chrono>

//Modos de programa de la FPGA
#define NN 0
#define IMG 1
#define CNN 2

//Programas de la FPGA
#define NOT_LOADED "FPGA_SIN_PROGRAMA"
#define NN_DNN1 "PRG_DNN_V1"
#define IMG_1920x1080 "PRG_IMG_1920x1080"
#define IMG_1000x1000 "PRG_IMG_1000x1000"
#define IMG_1000x1000TO100x100 "PRG_IMG_1000x1000TO100x100"

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
        int n_outs;
        int n_layers;
        int *n_p_l;
        int n_neurons;
        int n_params;

        int *params;
        int activations;
        int *bias;

        int n_sets;
        bool gradient_init;

        int64_t gradient_performance;
        int64_t forward_performance;

    private:

        fpga::fpga_handler* master = nullptr;
        int identifier = 0;

    public:

        ~net_fpga();
        net_fpga();
        net_fpga(const net::net_data &data, bool random); //* net::net_data como copia para mantener operaciones move
        net_fpga(net_fpga &&rh);
        net_fpga &operator=(net_fpga &&rh);
        net_fpga &operator=(const net_fpga &rh);

        net::net_data get_net_data() override;

        std::vector<float> launch_forward(const std::vector<float> &inputs) override;
        void enqueue_net(const std::vector<float> &inputs);
        void solve_pack();
        std::vector<float>  read_net();
        
        std::vector<float> launch_gradient(size_t iterations, float error_threshold, float multiplier) override;
        void print_inner_vals() override;
        signed long get_gradient_performance() override;
        signed long get_forward_performance() override;
    };
}

#endif