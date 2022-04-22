#ifndef NET_FPGA_HANDLER_H
#define NET_FPGA_HANDLER_H

#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"
#include "defines.h"
#include "fpgaDefines.h"

using namespace aocl_utils;

namespace fpga{

class fpga_handler{

    public:

        // static fpga_handler* ptr_handler;
        static bool there_is_a_handler;
        bool im_the_handler = false;

    private:        

        bool shared_in = false;

        std::vector<char*> img_out_dirs;
        // std::vector<int*> net_outs_dirs;

        std::vector<unsigned char> out_img_buff;
        std::vector<long int> out_buff;

        int nets_enqueued = 0;
        std::vector<net_register> net_list;   

        int wr_ev_ind = 0;
        std::vector<cl_event> wr_events;
        int exe_ev_ind = 0;
        std::vector<cl_event> exe_events;
        int rd_ev_ind = 0;
        std::vector<cl_event> rd_events;

    public:
        
        ~fpga_handler();
        fpga_handler();

        void activate_handler();

        void enqueue_image(std::string prg_name, std::vector<unsigned char> &in_image);//No bloqueante
        int enqueue_net(fpga_data & in_net, std::vector<long int> &inputs, bool reload = true, bool same_in = false, bool big_nets = false); //Gestion de eventos para hacerlo no bloqueante

        bool check_img_ready();

        void solve_nets();

        void read_image(std::vector<unsigned char> out_image);//Bloqueante
        std::vector<long int> read_net(int identifier, int nouts = 0, bool all = false);//Bloqueante

        void _cleanup();

    private:

        void _show_events();
        void _init_program(std::string prg_name, int prg_kind);
};
}

void cleanup();

#endif