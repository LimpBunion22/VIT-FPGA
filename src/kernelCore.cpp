
#include "kernelCore.h"
#include "defines.h"
#include <iostream>

using namespace std;
using namespace fpga;
using namespace aocl_utils;

kernel_core::~kernel_core()
{
    kernel_cleanup();
}

kernel_core::kernel_core(int id, std::string name, cl_program program, cl_context context, cl_device_id device, int bytes_inout, int bytes_params, int bytes_bias) : kernel_id(id), kernel_name(name), in_out_bytes_sz(bytes_inout / 2), params_bytes_sz(bytes_params), bias_bytes_sz(bytes_bias), line_events(MAX_EVENTS), configuration(6, 0)
{
    cl_int err;
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
    bias_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_bias * sizeof(long int), NULL, &err);
    checkError(err, "Failed to create buffer bias");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_out_dev);
    checkError(err, "Failed to set arg inouts");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_dev);
    checkError(err, "Failed to set arg params");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bias_dev);
    checkError(err, "Failed to set arg bias");

    lib_uce_index = MAX_EVENTS - 1;
    user_callback_events.reserve(MAX_EVENTS);
    for (int i = 0; i < MAX_EVENTS; i++)
        user_callback_events.emplace_back(clCreateUserEvent(context, &err));

    lib_le_index = MAX_EVENTS - 1;
}

bool kernel_core::switch_mem_mode(u_char new_mem_mode){
    
    if(new_mem_mode == MEM_MODE_BY_LOTS || new_mem_mode == MEM_MODE_FULL){
        memory_mode = new_mem_mode;
        return true;
    }
    else{
        return false;
    }
}

void kernel_core::enq_inputs(vector<long int> &inputs, cl_context context)
{
    cl_int err;    
    _reset_events(context);

    err = clEnqueueWriteBuffer(wr_queue, in_out_dev, CL_FALSE, inout_side_sel * in_out_bytes_sz, inputs.size() * sizeof(long int), inputs.data(), 0, NULL, &line_events[_le_event()]);
    checkError(err, "Failed to enqueue inputs");

    params_h_dir = 0;
    params_d_dir = 0;

    bias_h_dir = 0;
    bias_d_dir = 0;

    kernel_busy = true;
}

void kernel_core::enq_layer(fpga_data &fpga_data2enq, int layer, bool load_params)
{

    cl_int err;

    // Configuration = {n_ins/N_NEURONS, ins_dir(/N_NEURONS/sizeof(long int)), n_outs/N_NEURONS, outs_dir/N_NEURONS/sizeof(long int), params_dir/N_NEURONS, bias_dir/N_NEURONS}
    configuration[0] = fpga_data2enq.n_ins >> 4;
    configuration[1] = inout_side_sel * in_out_bytes_sz >> 7;
    configuration[2] = fpga_data2enq.n_p_l[layer] >> 4;
    configuration[3] = (inout_side_sel ^ 1) * in_out_bytes_sz >> 7;
    configuration[4] = params_d_dir >> 7;
    configuration[5] = bias_d_dir >> 7;

    int params2enq = (layer == 0 ? fpga_data2enq.n_ins * fpga_data2enq.n_p_l[layer] : fpga_data2enq.n_p_l[layer - 1] * fpga_data2enq.n_p_l[layer])* sizeof(long int);
    int bias2enq = fpga_data2enq.n_p_l[layer]* sizeof(long int);

    if (memory_mode == MEM_MODE_FULL)
    {
        if (load_params == true)
        {
            err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_d_dir, params2enq, &fpga_data2enq.params[params_h_dir], 0, NULL, NULL);
            checkError(err, "Failed to enqueue inputs");
            err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, bias_d_dir, bias2enq, &fpga_data2enq.bias[bias_h_dir], 0, NULL, &line_events[_le_event()]);
            checkError(err, "Failed to enqueue inputs");

            params_h_dir += params2enq;
            bias_h_dir += bias2enq;
        }

        clSetEventCallback(line_events[le_ind], CL_COMPLETE, &enq_callback_func, (void *)this);
        err = clEnqueueTask(exe_queue, kernel, 1, &user_callback_events[uce_ind], &line_events[_le_event()]);
        checkError(err, "Failed to enqueue task");

        params_d_dir += params2enq;
        bias_d_dir += bias2enq;
    }
    else
    {
        if(layer>1)//Revisar el pipeline y actualizacion de direcciones en device. Implementar funciones de eventos
            err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_d_dir, params2enq, &fpga_data2enq.params[params_h_dir], 1, &line_events[_pipe_le_event()], NULL);
        else
            err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_d_dir, params2enq, &fpga_data2enq.params[params_h_dir], 0, NULL, NULL);
        checkError(err, "Failed to enqueue inputs");
        err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, bias_d_dir, bias2enq, &fpga_data2enq.bias[bias_h_dir], 0, NULL, &line_events[_le_event()]);
        checkError(err, "Failed to enqueue inputs");

        params_h_dir += params2enq;
        bias_h_dir += bias2enq;

        clSetEventCallback(line_events[le_ind], CL_COMPLETE, &enq_callback_func, (void *)this);
        err = clEnqueueTask(exe_queue, kernel, 1, &user_callback_events[uce_ind], &line_events[_le_event()]);
        checkError(err, "Failed to enqueue task");

        params_d_dir = inout_side_sel * params_bytes_sz/2;
        bias_d_dir = inout_side_sel * in_out_bytes_sz/2;
    }

    inout_side_sel ^= 1;
}

void kernel_core::enq_read(std::vector<long int> & outs)
{
    cl_int err;
    err = clEnqueueReadBuffer(exe_queue, in_out_dev, CL_TRUE, inout_side_sel*in_out_bytes_sz, outs.size() * sizeof(long int), outs.data(), 1, &line_events[_pipe_le_event()], NULL);
    checkError(err, "Failed to enqueue read");   

    kernel_busy = false; 
}

void kernel_core::kernel_cleanup()
{

    if (kernel)
        clReleaseKernel(kernel);
    if (wr_queue)
        clReleaseCommandQueue(wr_queue);
    if (exe_queue)
        clReleaseCommandQueue(exe_queue);

    for (int i = 0; i < MAX_EVENTS; i++)
    {
        clReleaseEvent(user_callback_events[i]);
        clReleaseEvent(line_events[i]);
    }
}

void kernel_core::_reset_events(cl_context context){

    cl_int err;

    for (int i = 0; i < MAX_EVENTS; i++)
    {
        clReleaseEvent(user_callback_events[i]);
        clReleaseEvent(line_events[i]);
        user_callback_events[i] = clCreateUserEvent(context, &err);
        checkError(err, "Failed event");
        line_events[i] = cl_event();
    }

}

int kernel_core::_uce_event(){

    uce_ind = lib_uce_index == 0 ? 0 : lib_uce_index-1;
    lib_uce_index ++;
    if(lib_uce_index>= MAX_EVENTS)
        cout << RED << "UCE EVENT OVERFLOW" << RESET <<"\n";

    return lib_uce_index-1;
}

int kernel_core::_le_event(){

    le_ind = lib_le_index == 0 ? 0 : lib_le_index-1;
    lib_le_index ++;
    if(lib_le_index>= MAX_EVENTS)
        cout << RED << "LE EVENT OVERFLOW" << RESET <<"\n";

    return lib_le_index-1;
}

int kernel_core::_pipe_le_event(){

    if((lib_le_index-3)%2 !=0)
        cout << YELLOW << "PIPE LINE EVENTS UNEXPECTED" << RESET <<"\n";

    return lib_le_index-3;
}

void kernel_core::_show_events()
{
    cout << "lib_uce_index " << lib_uce_index << "\n";
    cout << "lib_le_index " << lib_le_index << "\n";
    for (int i = 0; i <= MAX_EVENTS; i++)
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
            cout << "   " << i << ": USER_EVENT " << "UNKNOWN" << RESET;
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
            cout << "   LINE_EVENT " << ": UNKNOWN" << RESET << "\n";
        }
    }
}

void enq_callback_func(cl_event event, cl_int event_command_exec_status, void *user_data){

    kernel_core * father = (kernel_core *)user_data;

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
}