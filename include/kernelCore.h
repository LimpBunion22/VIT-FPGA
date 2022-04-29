#ifndef P_KERNEL_CORE_H
#define P_KERNEL_CORE_H

#include <string>
#include "AOCLUtils/aocl_utils.h"
#include "fpgaDefines.h"

namespace fpga{
    class kernel_core{

        public:
            
            u_int kernel_id = -1;
            std::string kernel_name = "UNAMED KERNEL";
            cl_kernel kernel = nullptr;

            bool kernel_busy = false;

            int params_h_dir, params_d_dir, bias_h_dir, bias_d_dir = 0;

            std::vector<int> configuration;

            int uce_ind, lib_uce_index, used_uce_index = 0;
            std::vector<cl_event> user_callback_events;

        private:  

            u_char memory_mode = MEM_MODE_BY_LOTS;
            
            cl_command_queue wr_queue, exe_queue = nullptr;

            int in_out_bytes_sz, params_bytes_sz, bias_bytes_sz = 0;
            cl_mem in_out_dev, params_dev, bias_dev = nullptr;
            int le_ind, lib_le_index, used_le_index = 0;
            std::vector<cl_event> line_events;

            int inout_side_sel = 0;

        public:

            ~kernel_core();

            kernel_core(int id, std::string name, cl_program program, cl_context context, cl_device_id device, int bytes_inout, int bytes_params, int bytes_bias);

            bool switch_mem_mode(u_char new_mem_mode);

            void enq_inputs(std::vector<long int> &inputs, cl_context context);
            void enq_layer(fpga_data &fpga_data2enq, int layer, bool load_params = true);
            void enq_read(std::vector<long int> & outs);

            void kernel_cleanup();
            int _uce_event();

        private:

            void _reset_events(cl_context context);
            int _le_event();
            int _pipe_le_event();

            void _show_events();

    };

    void enq_callback_func(cl_event event, cl_int event_command_exec_status, void *user_data);
}

#endif