#ifndef NETCPU_H
#define NETCPU_H

#include <netAbstract.h>
// #include <mathStructsCPU.h>
#include <chrono>

namespace fpga
{
    class net_fpga : public net::net_abstract
    {

    private:
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

    private:
        net_fpga() = delete;

    public:
        net_fpga(const net::net_data &data, bool derivate, bool random); //* net::net_data como copia para mantener operaciones move
        net_fpga(net_fpga &&rh);
        net_fpga &operator=(net_fpga &&rh);
        net_fpga &operator=(const net_fpga &rh);

        net::net_data get_net_data() override;
        std::vector<DATA_TYPE> launch_forward(const std::vector<DATA_TYPE> &inputs) override;
        void init_gradient(const net::net_sets &sets) override;
        std::vector<DATA_TYPE> launch_gradient(size_t iterations) override;
        void print_inner_vals() override;
        signed long get_gradient_performance() override;
        signed long get_forward_performance() override;
    };
}

#endif