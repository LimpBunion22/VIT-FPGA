#ifndef NET_FPGA_HANDLER_H
#define NET_FPGA_HANDLER_H

#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"
#include "defines.h"
#include "fpgaDefines.h"
#include "kernelCore.h"

using namespace aocl_utils;

namespace fpga{

class fpga_handler{

    public:

        // static fpga_handler* ptr_handler;
        static bool there_is_a_handler;
        bool im_the_handler = false;

    private:        

        bool shared_in = false;

        int nets_enqueued = 0;
        std::vector<net_register> net_list; 

        std::vector<kernel_core> cores;

    public:
        
        ~fpga_handler();
        fpga_handler();

        void activate_handler();

        int enqueue_net(fpga_data & in_net, std::vector<long int> &inputs, bool reload = true, bool big_nets = false); //Gestion de eventos para hacerlo no bloqueante
        void solve_nets();
        std::vector<long int> read_net(int identifier);//Bloqueante

        void _cleanup();

    private:
    
        void _init_program();
};
}

void cleanup();

#endif