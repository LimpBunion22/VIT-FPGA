#ifndef NETABSTRACT_H
#define NETABSTRACT_H

#include <defines.h>

namespace net
{
    class net_abstract
    {
    public:
        virtual ~net_abstract() {}
        virtual net_data get_net_data() = 0;
        virtual std::vector<float> launch_forward(const std::vector<float> &inputs) = 0; //* returns result
        virtual std::vector<float> launch_gradient(const net_sets &sets, size_t iterations, size_t batch_size,
                                                   float alpha, float alpha_decay, float lambda, float error_threshold, int norm) = 0; //* returns iterations errors
        virtual void print_inner_vals() = 0;
        virtual signed long get_gradient_performance() = 0;
        virtual signed long get_forward_performance() = 0;
    };
}
#endif