#ifndef NETFPGA_H
#define NETFPGA_H

#include <netBuilder.h>
#include <fpgaHandler.h>
#include <chrono>

// Modos de programa de la FPGA
#define NN 0
#define IMG 1
#define CNN 2

// Programas de la FPGA
#define NOT_LOADED "FPGA_SIN_PROGRAMA"
#define NN_DNN1 "PRG_DNN_V1"
#define IMG_1920x1080 "PRG_IMG_1920x1080"
#define IMG_1000x1000 "PRG_IMG_1000x1000"
#define IMG_1000x1000TO100x100 "PRG_IMG_1000x1000TO100x100"

// Kernels de los programas
#define NN_DNN1_KERNEL "network_v1"
#define IMG_IN_KERNEL "image_process"
#define IMG_OUT_BORDERS_KERNEL "image_borders"

namespace fpga
{

    class net_fpga : public net::builder
    {

    public:
        std::string net_ident;

        // Net variables
        int n_ins;
        int n_outs;
        int n_layers;
        std::vector<int> n_p_l;
        int n_neurons;
        int n_params;

        std::vector<FPGA_DATA_TYPE> params;
        int activations;
        std::vector<FPGA_DATA_TYPE> bias;

        int n_sets;
        bool gradient_init;

        int64_t gradient_performance;
        int64_t forward_performance;

        fpga_data my_data;

    private:
        fpga_handler &master;
        fpga_data get_fpga_data();
        int identifier = 0;

    public:
        ~net_fpga();
        net_fpga() = delete;
        net_fpga(fpga_handler &handler);

        void set_input_size(int input_size) override;
        void build_fully_layer(int layer_size, int activation = net::RELU2) override;
        void build_net() override;
        void build_net_from_file(const net::layout &layout) override;
        void build_net_from_data(int input_size, const std::vector<int> &n_p_l, const std::vector<int> &activations) override;

        void set_fpga_handler(fpga_handler &handler);

        net_fpga(net_fpga &&rh);
        net_fpga &operator=(net_fpga &&rh);
        net_fpga &operator=(const net_fpga &rh);

        net::layout get_net_data()const override;

        std::vector<float> run_forward(const std::vector<float> &inputs) override;
        void enqueue_net(const std::vector<float> &inputs, bool reload = true, bool same_in = false, bool big_nets = false);
        void solve_pack();
        std::vector<float> read_net();

        std::vector<float> run_gradient(const net::set &set) override;

        net::builder &attr(int attr, float value) override;
        net::builder &attr(int attr, int value = 0) override;
        signed long get_gradient_performance()const override;
        signed long get_forward_performance()const override;
    };
}

#endif