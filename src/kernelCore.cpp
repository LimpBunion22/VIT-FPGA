
#include "kernelCore.h"
#include "defines.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <ctime>  


using namespace std;
using namespace aocl_utils;
using namespace chrono;

namespace fpga
{
    kernel_core::~kernel_core()
    {
        kernel_cleanup();
    }

    kernel_core::kernel_core(int id, std::string name, cl_program program, cl_context &context, cl_device_id device, size_t bytes_inout, size_t bytes_params, size_t bytes_bias) : kernel_id(id), kernel_name(name), in_out_bytes_sz(bytes_inout / 2), params_bytes_sz(bytes_params), bias_bytes_sz(bytes_bias), line_events(MAX_EVENTS), user_callback_events(MAX_EVENTS), configuration(6, 0)
    {
        cl_int err;
        mycontext = context;
        mydevice = device;
        wr_queue = clCreateCommandQueue(context, device, 0, &err);
        checkError(err, "Failed to create write queue");
        exe_queue = clCreateCommandQueue(context, device, 0, &err);
        checkError(err, "Failed to create exe queue");

        kernel = clCreateKernel(program, kernel_name.c_str(), &err);
        checkError(err, "Failed to create kernel");

        in_out_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_inout, NULL, &err); // CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        checkError(err, "Failed to create buffer inouts");
        params_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_params, NULL, &err);
        checkError(err, "Failed to create buffer params");
        bias_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_bias, NULL, &err);
        checkError(err, "Failed to create buffer bias");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_out_dev);
        checkError(err, "Failed to set arg inouts");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_dev);
        checkError(err, "Failed to set arg params");
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bias_dev);
        checkError(err, "Failed to set arg bias");

        err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&configuration[0]);
        checkError(err, "set n_ins");
        err = clSetKernelArg(kernel, 4, sizeof(int), (void *)&configuration[1]);
        checkError(err, "set in_dir");
        err = clSetKernelArg(kernel, 5, sizeof(int), (void *)&configuration[2]);
        checkError(err, "set n_outs");
        err = clSetKernelArg(kernel, 6, sizeof(int), (void *)&configuration[3]);
        checkError(err, "set out_dir");
        err = clSetKernelArg(kernel, 7, sizeof(int), (void *)&configuration[4]);
        checkError(err, "set params_dir");
        err = clSetKernelArg(kernel, 8, sizeof(int), (void *)&configuration[5]);
        checkError(err, "set bias_dir");

        for (int i = 0; i < MAX_EVENTS; i++)
        {
            user_callback_events[i] = clCreateUserEvent(context, &err);
            checkError(err, "Failed event");
        }
    }

    bool kernel_core::switch_mem_mode(u_char new_mem_mode)
    {

        if (new_mem_mode == MEM_MODE_BY_LOTS || new_mem_mode == MEM_MODE_FULL)
        {
            memory_mode = new_mem_mode;
            return true;
        }
        else
        {
            return false;
        }
    }

    void kernel_core::enq_inputs(vector<FPGA_DATA_TYPE> &inputs, cl_context &context)
    {
        cl_int err;
        _reset_events(context);
        clReleaseCommandQueue(wr_queue);
        clReleaseCommandQueue(exe_queue);
        wr_queue = clCreateCommandQueue(context, mydevice, 0, &err);
        checkError(err, "Failed to create write queue");
        exe_queue = clCreateCommandQueue(context, mydevice, 0, &err);
        checkError(err, "Failed to create exe queue");

#if fpga_performance == 1
        auto start = high_resolution_clock::now();
#endif

        err = clEnqueueWriteBuffer(wr_queue, in_out_dev, CL_FALSE, inout_side_sel * in_out_bytes_sz, inputs.size() * sizeof(FPGA_DATA_TYPE), inputs.data(), 0, NULL, &line_events[_le_event()]);
        checkError(err, "Failed to enqueue inputs");
        err = clFlush(wr_queue);
        checkError(err, "Failed to flush writes");
#if fpga_performance == 1
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        write_ins_perf += duration.count();
#endif

        params_h_dir = 0;
        params_d_dir = 0;

        bias_h_dir = 0;
        bias_d_dir = 0;

        kernel_busy = true;
    }

    void kernel_core::enq_layer(fpga_data &fpga_data2enq, int layer, bool load_params)
    {

        cl_int err;
        
        // Configuration = {n_ins/N_INS, ins_dir(/SZ_PCK/sizeof(FPGA_DATA_TYPE)), n_outs/N_NEURONS, outs_dir(/SZ_PCK/sizeof(FPGA_DATA_TYPE)), params_dir(/SZ_PCK/sizeof(FPGA_DATA_TYPE)), bias_dir(/SZ_PCK/sizeof(FPGA_DATA_TYPE))}
        configuration[0] = fpga_data2enq.n_ins >> 4;
        configuration[1] = (inout_side_sel * in_out_bytes_sz) >> (4+sizeof(FPGA_DATA_TYPE));
        configuration[2] = fpga_data2enq.n_p_l[layer] >> 4;
        configuration[3] = ((inout_side_sel ^ 1) * in_out_bytes_sz) >> (4+sizeof(FPGA_DATA_TYPE));
        configuration[4] = params_d_dir >> (4+sizeof(FPGA_DATA_TYPE));
        configuration[5] = bias_d_dir >> (4+sizeof(FPGA_DATA_TYPE));

        int params2enq = (layer == 0 ? fpga_data2enq.n_ins * fpga_data2enq.n_p_l[layer] : fpga_data2enq.n_p_l[layer - 1] * fpga_data2enq.n_p_l[layer]) * sizeof(FPGA_DATA_TYPE);
        int bias2enq = fpga_data2enq.n_p_l[layer] * sizeof(FPGA_DATA_TYPE);           

        if (memory_mode == MEM_MODE_FULL)
        {
            cout1(BOLDWHITE,"MEM_MODE ","MEM_MODE_FULL")
            #if fpga_performance == 1
                    auto start = high_resolution_clock::now();
            #endif 
            if (load_params == true)
            {
                cout1(BOLDWHITE,"LOAD_PARAMS ","true")
                err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_d_dir, params2enq, &fpga_data2enq.params[params_h_dir], 0, NULL, NULL);
                checkError(err, "Failed to enqueue params");

                err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, bias_d_dir, bias2enq, &fpga_data2enq.bias[bias_h_dir], 0, NULL, &line_events[_le_event()]);
                checkError(err, "Failed to enqueue bias");
                err = clFlush(wr_queue);
                checkError(err, "Failed to flush writes");

                params_h_dir += params2enq/sizeof(FPGA_DATA_TYPE);
                bias_h_dir += bias2enq/sizeof(FPGA_DATA_TYPE); 
            }

            clSetEventCallback(line_events[le_ind], CL_COMPLETE, &enq_callback_func, (void *)this);
            err = clEnqueueTask(exe_queue, kernel, 1, &user_callback_events[_uce_event_ind()], &line_events[_le_event()]);
            checkError(err, "Failed to enqueue task");
            #if fpga_performance == 1
                auto end = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                write_bp_perf += duration.count();
                start = high_resolution_clock::now();
            #endif
            err = clFlush(exe_queue);
            checkError(err, "Failed to flush task");
            #if fpga_performance == 1
                end = high_resolution_clock::now();
                duration = duration_cast<microseconds>(end - start);
                enq_task_perf += duration.count();
            #endif

            params_d_dir += params2enq;
            bias_d_dir += bias2enq;
        }
        else
        {    
            #if fpga_performance == 1
                    auto start = high_resolution_clock::now();
            #endif      
            if (layer > 1)
            {
                err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_d_dir, params2enq, &fpga_data2enq.params[params_h_dir], 1, &line_events[ _pipe_le_event()], NULL);

            }else{
                err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_d_dir, params2enq, &fpga_data2enq.params[params_h_dir], 0, NULL, NULL);
            }
            err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, bias_d_dir, bias2enq, &fpga_data2enq.bias[bias_h_dir], 0, NULL, &line_events[_le_event()]);
            checkError(err, "Failed to enqueue bias");
            err = clFlush(wr_queue);
            checkError(err, "Failed to flush writes");
            #if fpga_performance == 1
                auto end = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                write_bp_perf += duration.count();
                start = high_resolution_clock::now();
            #endif

            params_h_dir += params2enq/sizeof(FPGA_DATA_TYPE);
            bias_h_dir += bias2enq/sizeof(FPGA_DATA_TYPE);            

            clSetEventCallback(line_events[le_ind], CL_COMPLETE, &enq_callback_func, (void *)this);            
            err = clEnqueueTask(exe_queue, kernel, 1, &user_callback_events[_uce_event_ind()], &line_events[_le_event()]);//
            checkError(err, "Failed to enqueue task");
            err = clFlush(exe_queue);
            checkError(err, "Failed to flush task");
            // _show_events(aux);
            #if fpga_performance == 1
                end = high_resolution_clock::now();
                duration = duration_cast<microseconds>(end - start);
                enq_task_perf += duration.count();
            #endif

            params_d_dir = (inout_side_sel^1) * params_bytes_sz / 2;
            bias_d_dir = (inout_side_sel^1) * bias_bytes_sz / 2;
        }

        inout_side_sel ^= 1;
    }

    void kernel_core::enq_read(std::vector<FPGA_DATA_TYPE> &outs)
    {
        cl_int err;
        err = clEnqueueReadBuffer(exe_queue, in_out_dev, CL_TRUE, inout_side_sel * in_out_bytes_sz, outs.size() * sizeof(FPGA_DATA_TYPE), outs.data(), 0, NULL, NULL);
        // _show_events();
        checkError(err, "Failed to enqueue read");
        #if fpga_performance == 1
            fpga_info("             CORE INFO PERFORMANCE");
            fpga_info("                 wr ins "<<write_ins_perf<<"us");
            fpga_info("                 wr bias and params "<<write_bp_perf<<"us");
            fpga_info("                 enq task "<<enq_task_perf<<"us");
            write_ins_perf=0;
            write_bp_perf=0;
            enq_task_perf=0;
        #endif

        kernel_busy = false;
    }

    void kernel_core::kernel_cleanup()
    {

        if (kernel){
            clReleaseKernel(kernel);
            kernel = nullptr;
        }
        if (wr_queue){
            clReleaseCommandQueue(wr_queue);
            wr_queue = nullptr;
        }
        if (exe_queue){
            clReleaseCommandQueue(exe_queue);
            exe_queue = nullptr;
        }

        for (int i = 0; i < MAX_EVENTS; i++)
        {
            if(user_callback_events[i]){
                clReleaseEvent(user_callback_events[i]);
                user_callback_events[i] = nullptr;
            }
            if(line_events[i]){
                clReleaseEvent(line_events[i]);
                line_events[i] = nullptr;
            }
        }
    }

    void kernel_core::_reset_events(cl_context &context)
    {

        cl_int err;
        uce_ind = 0;
        lib_uce_index = 0;
        le_ind = 0;
        lib_le_index = 0;

        for (int i = 0; i < uce_ind; i++)
        {
            clReleaseEvent(user_callback_events[i]);
            // clReleaseEvent(line_events[i]);
            user_callback_events[i] = clCreateUserEvent(context, &err);
            checkError(err, "Failed event");
            // clSetUserEventStatus(user_callback_events[i], CL_COMPLETE);
            // checkError(err, "Failed event");
        }
        // line_events = vector<cl_event>(MAX_EVENTS);
    }

    int kernel_core::_uce_event()
    {
        lib_uce_index++;
        if (lib_uce_index >= MAX_EVENTS)
            cout << RED << "UCE EVENT OVERFLOW" << RESET << "\n";
        // fpga_info("task running " << lib_uce_index-1);
        return lib_uce_index-1;
    }

    int kernel_core::_uce_event_ind()
    {
        uce_ind ++;
        if (uce_ind >= MAX_EVENTS)
            cout << RED << "UCE EVENT OVERFLOW" << RESET << "\n";
        // fpga_info("task enqueued " << uce_ind-1);
        // _show_events(uce_ind-1);

        return uce_ind-1;
    }

    int kernel_core::_le_event()
    {
        le_ind = lib_le_index;
        lib_le_index++;
        if (lib_le_index >= MAX_EVENTS)
            cout << RED << "LE EVENT OVERFLOW" << RESET << "\n";

        return le_ind;
    }

    int kernel_core::_pipe_le_event()
    {

        if ((lib_le_index - 3) % 2 != 0)
            cout << YELLOW << "PIPE LINE EVENTS UNEXPECTED" << RESET << "\n";

        return lib_le_index - 3;
    }

    void kernel_core::_show_event_info(cl_event &event){
    }

    void kernel_core::_show_events(int ind)
    {
        fpga_info("lib_uce_index " << lib_uce_index);
        fpga_info("lib_le_index " << lib_le_index);
        int max = ind==-1? MAX_EVENTS:ind+1;
        int min = ind==-1? 0:ind;
        for (int i = min; i < max; i++)
        {
            cl_int status;
            size_t sz_st = sizeof(cl_int);
            clGetEventInfo(user_callback_events[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sz_st, &status, &sz_st);
            switch (status)
            {
            case CL_QUEUED:
                cout << "   " << i << ": USER_EVENT " << BOLDWHITE << "QUEUED" << RESET;
                break;
            case CL_SUBMITTED:
                cout << "   " << i << ": USER_EVENT " << BLUE << "SUBMITED" << RESET;
                break;
            case CL_RUNNING:
                cout << "   " << i << ": USER_EVENT " << YELLOW << "RUNNING" << RESET;
                break;
            case CL_COMPLETE:
                cout << "   " << i << ": USER_EVENT " << GREEN << "COMPLETED" << RESET;
                break;
            default:
                cout << "   " << i << ": USER_EVENT "
                     << "UNKNOWN" << RESET;
            }
            clGetEventInfo(line_events[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sz_st, &status, &sz_st);
            switch (status)
            {
            case CL_QUEUED:
                cout << "   LINE_EVENT " << BOLDWHITE << ": QUEUED" << RESET << "\n";
                break;
            case CL_SUBMITTED:
                cout << "   LINE_EVENT " << BLUE << ": SUBMITED" << RESET << "\n";
                break;
            case CL_RUNNING:
                cout << "   LINE_EVENT " << YELLOW << ": RUNNING" << RESET << "\n";
                break;
            case CL_COMPLETE:
                cout << "   LINE_EVENT " << GREEN << ": COMPLETED" << RESET << "\n";
                break;
            default:
                cout << "   LINE_EVENT "
                     << ": UNKNOWN" << RESET << "\n";
            }
        }
    }

    void enq_callback_func(cl_event event, cl_int event_command_exec_status, void *user_data)
    {

        kernel_core *father = (kernel_core *)user_data;

        cl_int err;
        int n_ins = father->configuration[0];
        int in_dir = father->configuration[1];
        int n_outs = father->configuration[2];
        int out_dir = father->configuration[3];
        int params_dir = father->configuration[4];
        int bias_dir = father->configuration[5];

        err = clSetKernelArg(father->kernel, 3, sizeof(int), (void *)&n_ins);
        checkError(err, "set n_ins");
        err = clSetKernelArg(father->kernel, 4, sizeof(int), (void *)&in_dir);
        checkError(err, "set in_dir");
        err = clSetKernelArg(father->kernel, 5, sizeof(int), (void *)&n_outs);
        checkError(err, "set n_outs");
        err = clSetKernelArg(father->kernel, 6, sizeof(int), (void *)&out_dir);
        checkError(err, "set out_dir");
        err = clSetKernelArg(father->kernel, 7, sizeof(int), (void *)&params_dir);
        checkError(err, "set params_dir");
        err = clSetKernelArg(father->kernel, 8, sizeof(int), (void *)&bias_dir);
        checkError(err, "set bias_dir");

        clSetUserEventStatus(father->user_callback_events[father->_uce_event()], CL_COMPLETE);
        // err = clEnqueueTask(father->exe_queue, father->kernel, 1, &father->user_callback_events[father->uce_ind], &father->line_events[father->_le_event()]);
        // checkError(err, "set bias_dir");
    }
}