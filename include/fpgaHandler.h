#ifndef NET_FPGA_HANDLER_H
#define NET_FPGA_HANDLER_H

#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"
#include "defines.h"

#define fpga_verbose 0
#if fpga_verbose == 0
    #define cout1(color, text, index)
    #define cout2(color, text, index)
    #define cout3(color, text, index)
#elif fpga_verbose == 1
    #define cout1(color, text, index) cout << color << text << index << RESET << "\n";
    #define cout2(color, text, index)
    #define cout3(color, text, index)
#elif fpga_verbose == 2
    #define cout1(color, text, index) cout << color << text << index << RESET << "\n";
    #define cout2(color, text, index) cout << color << text << index << RESET << "\n";
    #define cout3(color, text, index)
#elif fpga_verbose == 3
    #define cout1(color, text, index) cout << color << text << index << RESET << "\n";
    #define cout2(color, text, index) cout << color << text << index << RESET << "\n";
    #define cout3(color, text, index) cout << color << text << index << RESET << "\n";
#endif

// Configuration: [n_ins, in_dir, out_dir, params_dir, bias_dir]
#define N_INS 16
#define N_NEURONS 16
#define DECIMAL_FACTOR 1024
#define MAX_SZ_ENQUEUE 256

//Modos de programa de la FPGA
#define NN 0
#define IMG 1
#define CNN 2
using namespace aocl_utils;

namespace fpga{

typedef struct
{
    int n_ins;
    int n_layers;
    int n_params;
    int n_neurons;
    int* n_p_l;
    long int* params;
    long int* bias;
    // std::vector<int> activation_type; //* valor numérico que indica qué función usar por capa
} fpga_data;

typedef struct 
{
    fpga_data net;    
    bool big_net = false;
    
    bool enqueued = false;
    bool loaded = false;
    bool solved = false;
    bool readed = false;

    int wr_event = 0;
    int exe_event = 0;
    int rd_event = 0;

    int inout_mem_res = 0;
    int params_mem_res = 0;
    int bias_mem_res = 0;

    int in_out_base = 0;
    int params_base = 0;
    int bias_base = 0;

    // int outs_rel = 0;
    int params_enq_dir = 0;
    int bias_enq_dir = 0;

    int params_exe_dir = 0;    
    int bias_exe_dir = 0;

    int in_host_dir = 0;
    int params_host_dir = 0;
    int bias_host_dir = 0;

    int layer = 0;
    unsigned char layer_parity = 1;
}net_register;


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

        // // OpenCL runtime configuration
        // cl_platform_id platform = NULL;
        // cl_device_id device = NULL;
        // cl_context context = NULL;
        // cl_command_queue wr_queue = nullptr;
        // cl_command_queue rd_queue = nullptr;
        // cl_command_queue exe_queue = nullptr;
        // cl_program program = NULL;
        // cl_kernel kernel = NULL;
        // cl_int err = 0;

        // int in_img_free_mem;
        // cl_mem in_red_img_dev = NULL;
        // cl_mem in_green_img_dev = NULL;
        // cl_mem in_blue_img_dev = NULL;
        // int out_img_free_mem;
        // cl_mem out_img_dev = NULL;

        // int inout_net_free_mem;
        // cl_mem in_out_dev = NULL;
        // int params_net_free_mem;
        // cl_mem params_dev = NULL;
        // int bias_net_free_mem;
        // cl_mem bias_dev = NULL;   
        // cl_mem configuration_dev = NULL;     

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