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
        virtual std::vector<DATA_TYPE> launch_forward(const std::vector<DATA_TYPE> &inputs) = 0; //* returns result
        virtual void init_gradient(const net_sets &sets) = 0;
        virtual std::vector<DATA_TYPE> launch_gradient(size_t iterations) = 0; //* returns iterations errors
        virtual void print_inner_vals() = 0;
        virtual signed long get_gradient_performance() = 0;
        virtual signed long get_forward_performance() = 0;
    };
}
#endif