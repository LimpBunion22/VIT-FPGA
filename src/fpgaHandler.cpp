#include "fpgaHandler.h"
#include <netFPGA.h>
// #include <math.h>
#include <algorithm>
#include <iostream>
// #include <assert.h>
#include <stdio.h>
#include <stdlib.h>
// #include <cstring>
// #include <chrono>
#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"

#define NET_KERNEL_NAME "ME289WindsorV8"
#define IMG_KERNEL_NAME ""

#define IMG_BUF_SIZE 256 * 1024 * 1024
#define NET_INOUT_BUF_SIZE 2 * 16 * 1024 * 4
#define NET_PARAMS_BUF_SIZE 16 * 1024 * 16 * 1024 * 4
#define NET_BIAS_BUF_SIZE 2 * 16 * 1024 * 4

using namespace std;
using namespace fpga;
using namespace aocl_utils;
using namespace chrono;

bool fpga_handler::there_is_a_handler = false;

fpga_handler::fpga_handler() : net_list(MAX_SZ_ENQUEUE), wr_events(MAX_SZ_ENQUEUE), exe_events(MAX_SZ_ENQUEUE), rd_events(MAX_SZ_ENQUEUE)
{

    if (there_is_a_handler)
        cout << "This FPGA device already has a handler\n";
    else
    {
        im_the_handler = true;
        there_is_a_handler = true;
    }
}

fpga_handler::~fpga_handler()
{

    _cleanup();
}

void fpga_handler::enqueue_image(std::string prg_name, std::vector<unsigned char> &in_image)
{
    cout << "Images not implemented yet\n";
}

bool fpga_handler::check_img_ready()
{
    cout << "Images not implemented yet\n";
}

void fpga_handler::read_image(std::vector<unsigned char> out_image)
{
    cout << "Images not implemented yet\n";
}

int fpga_handler::enqueue_net(net_fpga *in_net, std::vector<int> &inputs, bool reload, bool same_in, bool big_nets)
{
    shared_in = same_in;
    int max_neurons = *max_element(in_net->n_p_l, in_net->n_p_l + in_net->n_layers - 1);
    int inout_max = max(in_net->n_ins, max_neurons);

    if (inout_net_free_mem > inout_max || reload == false)
    {
        if (reload == true || net_list[nets_enqueued].big_net)
        {
            net_list[nets_enqueued].net = in_net;

            // Direcciones base delos buffers en la fpga
            net_list[nets_enqueued].enqueued = true;
            net_list[nets_enqueued].loaded = false;
            net_list[nets_enqueued].solved = false;
            net_list[nets_enqueued].readed = false;
            net_list[nets_enqueued].in_out_base = inout_net_free_mem;
            net_list[nets_enqueued].params_base = params_net_free_mem;
            net_list[nets_enqueued].bias_base = bias_net_free_mem;

            // Espacio reservado en los buffers de la fpga
            net_list[nets_enqueued].inout_mem_res = inout_max;

            if (big_nets || params_net_free_mem < in_net->n_params || bias_net_free_mem < in_net->n_neurons)
            {
                net_list[nets_enqueued].big_net = true;
                net_list[nets_enqueued].params_mem_res = N_NEURONS * 2 * inout_max;
                net_list[nets_enqueued].bias_mem_res = N_NEURONS * 2;
            }
            else
            {
                net_list[nets_enqueued].big_net = false;
                net_list[nets_enqueued].params_mem_res = in_net->n_params;
                net_list[nets_enqueued].bias_mem_res = in_net->n_neurons;
            }
        }

        inout_net_free_mem -= net_list[nets_enqueued].inout_mem_res;
        params_net_free_mem -= net_list[nets_enqueued].params_mem_res;
        bias_net_free_mem -= net_list[nets_enqueued].bias_mem_res;

        // Enqueues iniciales de entradas, parámetros y bias
        int next_ev = wr_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : wr_ev_ind++;

        if (reload == true || net_list[nets_enqueued].big_net)
        {
            net_list[nets_enqueued].loaded = false;
            if (same_in == false)
            {
                err = clEnqueueWriteBuffer(wr_queue, in_out_dev, CL_FALSE, net_list[nets_enqueued].in_out_base, in_net->n_ins, inputs.data(), 1, &(rd_events[net_list[nets_enqueued].rd_event]), NULL);
                checkError(err, "Failed to enqueue inputs");
                err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, net_list[nets_enqueued].params_base, in_net->n_ins*N_NEURONS, in_net->params, 0, NULL, NULL);
            }
            else
                err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, net_list[nets_enqueued].params_base, in_net->n_ins*N_NEURONS, in_net->params, 1, &(rd_events[net_list[nets_enqueued].rd_event]), NULL);

            checkError(err, "Failed to enqueue params");
            err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, net_list[nets_enqueued].bias_base, net_list[nets_enqueued].bias_mem_res / 2, in_net->bias, 0, NULL, NULL);
            checkError(err, "Failed to enqueue bias");
        }
        else
        {
            net_list[nets_enqueued].loaded = true;
            if (same_in == false)
            {
                err = clEnqueueWriteBuffer(wr_queue, in_out_dev, CL_FALSE, net_list[nets_enqueued].in_out_base, in_net->n_ins, inputs.data(), 1, &(rd_events[net_list[nets_enqueued].rd_event]), NULL);
                checkError(err, "Failed to enqueue inputs");
            }
        }

        int ins_dir = same_in ?  net_list[0].in_out_base : net_list[nets_enqueued].in_out_base;
        vector<int> configuration = {in_net->n_ins, ins_dir, net_list[nets_enqueued].in_out_base+NET_INOUT_BUF_SIZE/2, net_list[nets_enqueued].params_base, net_list[nets_enqueued].bias_base};
        err = clEnqueueWriteBuffer(wr_queue, configuration_dev, CL_FALSE, 0, 5, configuration.data(), 0, NULL, &(wr_events[next_ev]));
        checkError(err, "Failed to enqueue inputs");
        wr_ev_ind = next_ev;

        //Actualiza direcciones
        net_list[nets_enqueued].params_rel += net_list[nets_enqueued].params_mem_res;
        net_list[nets_enqueued].bias_rel += net_list[nets_enqueued].bias_mem_res / 2;

        //Enqueue de primera ejecución
        net_list[nets_enqueued].wr_event = wr_ev_ind;
        next_ev = exe_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : exe_ev_ind++;

        err = clEnqueueTask(exe_queue, kernel, 1, &(wr_events[wr_ev_ind]), &(exe_events[next_ev]));
        checkError(err, "Failed to enqueue task");
        net_list[nets_enqueued].exe_event = exe_ev_ind;
        exe_ev_ind = next_ev;
        
        nets_enqueued++;
        return nets_enqueued;
    }
    else{
        cout << "FPGA: Not enough memory to enqueue\n";
        return 0;
    }
        
}

void fpga_handler::solve_nets()
{
    for (int i = 0; i < nets_enqueued; i++)
    {   
        int pck_cnt = 0;      

        //Recorre las capas
        for(int l=0; l<net_list[i].net->n_layers; l++){
            
            int n_ins, ins_dir, outs_dir;

            if(l==0){
                n_ins = net_list[i].net->n_ins;
                ins_dir = shared_in ?  net_list[0].in_out_base : net_list[i].in_out_base;  
                outs_dir = net_list[0].in_out_base + NET_INOUT_BUF_SIZE/2;
            }
            else{
                n_ins = net_list[i].net->n_p_l[l-1];
                ins_dir = l%2==0 ? net_list[0].in_out_base : net_list[0].in_out_base+NET_INOUT_BUF_SIZE/2;
                outs_dir = l%2==0 ? net_list[0].in_out_base + NET_INOUT_BUF_SIZE/2 : net_list[0].in_out_base;
            }

            //Recorre los packs de ejecucción
            for (int p = l==0?1:0; p < net_list[i].net->n_p_l[0] / N_NEURONS; p++)
            {       
                int n_params = l==0 ? net_list[i].net->n_ins*N_NEURONS : net_list[i].net->n_p_l[l-1]*N_NEURONS;             
                int next_ev = wr_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : wr_ev_ind++;

                //Enqueue de bias y parametros
                if(!net_list[i].loaded && (net_list[i].big_net || pck_cnt == 0)){
                    int params_dir = pck_cnt%2==0 ? net_list[i].params_base : net_list[i].params_base + net_list[i].params_mem_res/2;
                    err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, params_dir, n_params,  net_list[i].net->params, 0, NULL, NULL);
                    checkError(err, "Failed to enqueue params");
                    int bias_dir = pck_cnt%2==0 ? net_list[i].bias_base : net_list[i].bias_base + net_list[i].bias_mem_res/2;
                    err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, bias_dir, N_NEURONS, net_list[i].net->bias, 0, NULL, NULL);
                    checkError(err, "Failed to enqueue bias");
                } 
                
                //Enqueue de configuracion
                vector<int> configuration = {n_ins, ins_dir, outs_dir+p*N_NEURONS, net_list[i].params_base+net_list[i].params_rel, net_list[i].bias_base+net_list[i].bias_rel};
                err = clEnqueueWriteBuffer(wr_queue, configuration_dev, CL_FALSE, 0, 5, configuration.data(), 1, &(exe_events[exe_ev_ind]), &(wr_events[next_ev]));
                checkError(err, "Failed to enqueue inputs");
                wr_ev_ind = next_ev;

                //Actualiza direcciones
                net_list[i].params_rel += net_list[i].params_mem_res/2;
                net_list[i].bias_rel += net_list[i].bias_mem_res/2;
                if(net_list[i].big_net){
                    if(net_list[i].bias_rel == net_list[i].bias_mem_res)
                        net_list[i].bias_rel = 0;
                    if(net_list[i].params_rel == net_list[i].params_mem_res)
                        net_list[i].params_rel = 0;
                }

                //Enqueue de ejecución
                net_list[i].wr_event = wr_ev_ind;
                next_ev = exe_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : exe_ev_ind++;

                err = clEnqueueTask(exe_queue, kernel, 1, &(wr_events[wr_ev_ind]), &(exe_events[next_ev]));
                checkError(err, "Failed to enqueue task");
                net_list[i].exe_event = exe_ev_ind;
                exe_ev_ind = next_ev;
                
                pck_cnt++;
            }

            net_list[i].layer_parity ^= 1;
        }        
        net_list[i].loaded = true;
    }
}

std::vector<int> fpga_handler::read_net(int identifier){
    
    int id = identifier-1;
    int n_outs = net_list[id].net->n_p_l[net_list[id].net->n_layers-1];
    vector<int> outs(n_outs,0);
    int next_ev = rd_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : rd_ev_ind++;

    err = clEnqueueReadBuffer(rd_queue, in_out_dev, CL_TRUE, 0, n_outs*sizeof(int), outs.data(), 1, &(exe_events[net_list[id].exe_event]), &(rd_events[rd_ev_ind]));
    checkError(err, "Failed to enqueue read outs");
    net_list[id].solved = true;
    net_list[id].readed = true;
    net_list[id].rd_event = rd_ev_ind;
    rd_ev_ind = next_ev;

    inout_net_free_mem -= net_list[id].inout_mem_res;
    params_net_free_mem -= net_list[id].params_mem_res;
    bias_net_free_mem -= net_list[id].bias_mem_res;
    nets_enqueued--;
}

void fpga_handler::_cleanup()
{

    if (im_the_handler)
    {
        if (kernel)
            clReleaseKernel(kernel);
        if (program)
            clReleaseProgram(program);
        if (wr_queue)
            clReleaseCommandQueue(wr_queue);
        if (rd_queue)
            clReleaseCommandQueue(rd_queue);
        if (exe_queue)
            clReleaseCommandQueue(exe_queue);
        if (context)
            clReleaseContext(context);

        for (int i = 0; i < MAX_SZ_ENQUEUE; i++)
        {
            clReleaseEvent(wr_events[i]);
            clReleaseEvent(exe_events[i]);
            clReleaseEvent(rd_events[i]);
        }
    }
}

void fpga_handler::_init_program(string prg_name, int prg_kind)
{
    // Limpia la configuración previa
    _cleanup();

    // Configuración principal de la FPGA
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "Failed to get platforms");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);
    checkError(err, "Failed to find device");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "Failed to create context");

    wr_queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Failed to create queue");
    rd_queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Failed to create queue");
    exe_queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Failed to create queue");

    const char *char_name = prg_name.c_str();
    std::string binary_file = getBoardBinaryFile(char_name, device); // Coge el aocx
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(err, "Failed to create program");

    // Inicializa los primeros eventos
    // wr_events[0] = clCreateUserEvent(context, NULL);
    // clSetUserEventStatus(wr_events[0], CL_COMPLETE);
    exe_events[0] = clCreateUserEvent(context, NULL);
    clSetUserEventStatus(exe_events[0], CL_COMPLETE);
    rd_events[0] = clCreateUserEvent(context, NULL);
    clSetUserEventStatus(rd_events[0], CL_COMPLETE);

    // Reserva de memoria en el PC
    switch (prg_kind)
    {
    case NN:
        net_outs_dirs.reserve(MAX_SZ_ENQUEUE);
        out_buff.reserve(NET_INOUT_BUF_SIZE / 2);
        break;
    case IMG:
        img_out_dirs.reserve(MAX_SZ_ENQUEUE);
        out_img_buff.reserve(IMG_BUF_SIZE);
        break;
    default:
        cout << "Tipo de programa no implementado en FPGA";
    }

    // Configuración del kernel y reserva de memoria en la FPGA
    switch (prg_kind)
    {
    case NN:

        // cout << "   Creating buffers:\n";
        inout_net_free_mem = NET_INOUT_BUF_SIZE / 2;
        in_out_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, NET_INOUT_BUF_SIZE * sizeof(int), NULL, &err); // CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        checkError(err, "Failed to create buffer inouts");
        params_net_free_mem = NET_PARAMS_BUF_SIZE;
        params_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, NET_PARAMS_BUF_SIZE * sizeof(int), NULL, &err);
        checkError(err, "Failed to create buffer params");
        bias_net_free_mem = NET_BIAS_BUF_SIZE;
        bias_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, NET_BIAS_BUF_SIZE * sizeof(int), NULL, &err);
        checkError(err, "Failed to create buffer bias");
        configuration_dev = clCreateBuffer(context, CL_MEM_READ_ONLY * sizeof(int), 5, NULL, &err);
        checkError(err, "Failed to create buffer bias");

        kernel = clCreateKernel(program, NET_KERNEL_NAME, &err);
        checkError(err, "Failed to create kernel");

        // cout << "   Setting Args:\n";
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_out_dev);
        checkError(err, "Failed to set arg inouts");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_dev);
        checkError(err, "Failed to set arg params");
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bias_dev);
        checkError(err, "Failed to set arg bias");
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&configuration_dev);
        checkError(err, "Failed to set arg configuration");
        break;

    case IMG:

        // cout << "   Creating buffers:\n";
        in_img_free_mem = IMG_BUF_SIZE;
        in_red_img_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, IMG_BUF_SIZE * sizeof(unsigned char), NULL, &err); // CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        checkError(err, "Failed to create buffer in red");
        in_green_img_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, IMG_BUF_SIZE * sizeof(unsigned char), NULL, &err); // CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        checkError(err, "Failed to create buffer in green");
        in_blue_img_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, IMG_BUF_SIZE * sizeof(unsigned char), NULL, &err); // CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        checkError(err, "Failed to create buffer in blue");
        out_img_free_mem = IMG_BUF_SIZE;
        out_img_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, IMG_BUF_SIZE * sizeof(unsigned char), NULL, &err);
        checkError(err, "Failed to create buffer out");

        kernel = clCreateKernel(program, IMG_KERNEL_NAME, &err);
        checkError(err, "Failed to create kernel");

        // cout << "   Setting Args:\n";
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_red_img_dev);
        checkError(err, "Failed to set arg inouts");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&in_green_img_dev);
        checkError(err, "Failed to set arg params");
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&in_blue_img_dev);
        checkError(err, "Failed to set arg bias");
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&out_img_dev);
        checkError(err, "Failed to set arg bias");
        break;

    default:
        cout << "Tipo de programa no implementado en FPGA";
    }
}