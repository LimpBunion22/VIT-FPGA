#ifndef P_KERNEL_CORE_H
#define P_KERNEL_CORE_H

#include <string>
#include "AOCLUtils/aocl_utils.h"
#include "fpgaDefines.h"
#include <iostream>

namespace fpga
{
    class kernel_core
    {

    public:
        u_int kernel_id = -1;
        std::string kernel_name = "UNAMED KERNEL";
        cl_kernel kernel = nullptr;

        bool kernel_busy = false;

        int params_h_dir = 0, params_d_dir = 0, bias_h_dir = 0, bias_d_dir = 0;

        std::vector<int> configuration;

        int uce_ind = 0, lib_uce_index = 0, used_uce_index = 0;
        std::vector<cl_event> user_callback_events;

    // private:
        u_char memory_mode = MEM_MODE_BY_LOTS;

        cl_command_queue wr_queue = nullptr, exe_queue = nullptr;

        int in_out_bytes_sz = 0, params_bytes_sz = 0, bias_bytes_sz = 0;
        cl_mem in_out_dev = nullptr, params_dev = nullptr, bias_dev = nullptr;
        int le_ind = 0, lib_le_index = 0, used_le_index = 0;
        std::vector<cl_event> line_events;

        int inout_side_sel = 0;
        cl_context mycontext = nullptr;

    public:
        ~kernel_core();
        kernel_core() = delete;

        kernel_core(int id, std::string name, cl_program program, cl_context &context, cl_device_id device, size_t bytes_inout, size_t bytes_params, size_t bytes_bias);

        bool switch_mem_mode(u_char new_mem_mode);

        void enq_inputs(std::vector<long int> &inputs, cl_context &context);
        void enq_layer(fpga_data &fpga_data2enq, int layer, bool load_params = true);
        void enq_read(std::vector<long int> &outs);

        void kernel_cleanup();
        int _uce_event();
        int _uce_event_ind();

    // private:
        void _reset_events(cl_context &context);
        int _le_event();
        int _pipe_le_event();

        void _show_event_info(cl_event &event);
        void _show_events(int ind = -1);
    };

    void enq_callback_func(cl_event event, cl_int event_command_exec_status, void *user_data);
}

#endif