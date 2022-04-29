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

namespace fpga{

    #define fpga_info(text) cout << BOLDWHITE << "INFO: " << text << "  [file: " << __FILE__ << ", line: " << __LINE__ << "]" << RESET << "\n"
    #define fpga_warning(text) cout << BOLDYELLOW << "WARNING: " << text << "  [file: " << __FILE__ << ", line: " << __LINE__ << "]" << RESET << "\n"
    #define fpga_error(text) cout << BOLDRED << "ERROR: " << text << "  [file: " << __FILE__ << ", line: " << __LINE__ << "]" << RESET << "\n"

    // Configuration
    #define N_CORES 8
    #define KERNEL_BASE_NAME "MustangGT1965_"
    #define PROGRAM_NAME "engine_kernel"
    
    #define N_INS 16
    #define N_NEURONS 16
    #define DECIMAL_FACTOR 1024
    #define MAX_EVENTS 64
    #define MAX_SZ_ENQUEUE 256

    //Buffers size
    #define INOUT_SIZE (2 * 16 * 1024 * 8 / N_CORES)
    #define PARAMS_SIZE (16UL * 1024 * 16 * 1024 * 8 / N_CORES)
    #define BIAS_SIZE (2 * 16 * 1024 * 8 / N_CORES)

    //Modos de memoria de los buffers
    #define MEM_MODE_BY_LOTS 0
    #define MEM_MODE_FULL 1

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

        bool free_slot = true;

        bool reload = true;
        bool big_net = false;
        
        bool enqueued = false;
        bool loaded = false;
        bool solved = false;

        int layer = 0;

        std::vector<long int> inputs;
        std::vector<long int> outs;

    }net_register;
}

#endif