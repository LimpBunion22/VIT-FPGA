#ifndef FPGA_DEFINES_H
#define FPGA_DEFINES_H

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

//Modos de memoria de los buffers
#define MEM_MODE_BY_LOTS 0
#define MEM_MODE_FULL 1

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
}

#endif