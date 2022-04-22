
#include "kernelCore.h"

using namespace std;
using namespace fpga;
using namespace aocl_utils;

#define MAX_EVENTS 64

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

void kernel_core::enq_inputs(vector<long int> &inputs)
{

    cl_int err;
    _reset_events();

    err = clEnqueueWriteBuffer(wr_queue, in_out_dev, CL_FALSE, inout_side_sel * in_out_bytes_sz, inputs.size() * sizeof(long int), inputs.data(), 0, NULL, &line_events[_le_event()]);
    checkError(err, "Failed to enqueue inputs");

    params_h_dir = 0;
    params_d_dir = 0;

    bias_h_dir = 0;
    bias_d_dir = 0;
}

void kernel_core::enq_layer(fpga_data &fpga_data2enq, int layer, bool load_params)
{

    cl_int err;

    // Configuration = {n_ins/N_NEURONS, ins_dir(/N_NEURONS/sizeof(long int)), n_outs/N_NEURONS, outs_dir/N_NEURONS/sizeof(long int), params_dir/N_NEURONS, bias_dir/N_NEURONS}
    configuration[0] = fpga_data2enq.n_ins >> 4;
    configuration[1] = inout_side_sel * in_out_bytes_sz >> 7;
    configuration[2] = fpga_data2enq.n_p_l[layer] >> 4;
    configuration[3] = (inout_side_sel ^ 1) * in_out_bytes_sz >> 7;
    configuration[4] = params_d_dir >> 4;
    configuration[5] = bias_d_dir >> 4;

    int params2enq = layer == 0 ? fpga_data2enq.n_ins * fpga_data2enq.n_p_l[layer] : fpga_data2enq.n_p_l[layer - 1] * fpga_data2enq.n_p_l[layer];
    int bias2enq = fpga_data2enq.n_p_l[layer];

    if (memory_mode == MEM_MODE_FULL)
    {

        if (load_params == true)
        {
            err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_d_dir * sizeof(long int), params2enq * sizeof(long int), &fpga_data2enq.params[params_h_dir], 0, NULL, NULL);
            checkError(err, "Failed to enqueue inputs");
            err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, bias_d_dir * sizeof(long int), bias2enq * sizeof(long int), &fpga_data2enq.bias[bias_h_dir], 0, NULL, &line_events[_le_event()]);
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
            err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_d_dir * sizeof(long int), params2enq * sizeof(long int), &fpga_data2enq.params[params_h_dir], 1, &line_events[_pipe_le_event()], NULL);
        else
            err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_d_dir * sizeof(long int), params2enq * sizeof(long int), &fpga_data2enq.params[params_h_dir], 0, NULL, NULL);
        checkError(err, "Failed to enqueue inputs");
        err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, bias_d_dir * sizeof(long int), bias2enq * sizeof(long int), &fpga_data2enq.bias[bias_h_dir], 0, NULL, &line_events[_le_event()]);
        checkError(err, "Failed to enqueue inputs");

        params_h_dir += params2enq;
        bias_h_dir += bias2enq;

        clSetEventCallback(line_events[le_ind], CL_COMPLETE, &enq_callback_func, (void *)this);
        err = clEnqueueTask(exe_queue, kernel, 1, &user_callback_events[uce_ind], &line_events[_le_event()]);
        checkError(err, "Failed to enqueue task");

        params_d_dir += params2enq;
        bias_d_dir += bias2enq;
    }

    inout_side_sel ^= 1;
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