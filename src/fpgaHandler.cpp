#include <fpgaHandler.h>
// #include <math.h>
#include <algorithm>
#include <iostream>
// #include <assert.h>
#include <stdio.h>
#include <stdlib.h>
// #include <cstring>
#include <chrono>
#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"

#include <thread>

#define NET_KERNEL_NAME "MustangGT1965_1"
#define IMG_KERNEL_NAME ""

#define IMG_BUF_SIZE (256 * 1024 * 1024)
#define NET_INOUT_BUF_SIZE (2 * 16 * 1024)
#define NET_PARAMS_BUF_SIZE (16 * 1024 * 16 * 1024)
#define NET_BIAS_BUF_SIZE (2 * 16 * 1024)

using namespace std;
using namespace fpga;
using namespace aocl_utils;

bool fpga_handler::there_is_a_handler = false;
fpga_handler* cl_ptr = nullptr;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue wr_queue = nullptr;
static cl_command_queue rd_queue = nullptr;
static cl_command_queue exe_queue = nullptr;
static cl_program program = NULL;
static cl_kernel kernel = NULL;

int in_img_free_mem;
cl_mem in_red_img_dev = NULL;
cl_mem in_green_img_dev = NULL;
cl_mem in_blue_img_dev = NULL;
int out_img_free_mem;
cl_mem out_img_dev = NULL;

int inout_net_free_mem;
cl_mem in_out_dev = NULL;
int params_net_free_mem;
cl_mem params_dev = NULL;
int bias_net_free_mem;
cl_mem bias_dev = NULL;   
// cl_mem configuration_dev = NULL;  

void set_args_callback(cl_event event, cl_int event_command_exec_status, void *user_data);

fpga_handler::fpga_handler() : net_list(MAX_SZ_ENQUEUE), wr_events(MAX_SZ_ENQUEUE), exe_events(MAX_SZ_ENQUEUE), rd_events(MAX_SZ_ENQUEUE)
{

    if (there_is_a_handler)
        cout << "This FPGA device already has a handler\n";
    else
    {
        // ptr_handler = this;
        im_the_handler = true;
        there_is_a_handler = true;
        cl_ptr = this;
    }
}

fpga_handler::~fpga_handler()
{
    _cleanup();
}

void fpga_handler::activate_handler()
{
    if (im_the_handler)
        _init_program("engine_kernel", NN);
}

void fpga_handler::enqueue_image(std::string prg_name, std::vector<unsigned char> &in_image)
{
    cout << "Images not implemented yet\n";
}

bool fpga_handler::check_img_ready()
{
    cout << "Images not implemented yet\n";
    return false;
}

void fpga_handler::read_image(std::vector<unsigned char> out_image)
{
    cout << "Images not implemented yet\n";
}

int fpga_handler::enqueue_net(fpga_data &in_net, std::vector<long int> &inputs, bool reload, bool same_in, bool big_nets)
{
    cl_int err = 0;
    shared_in = same_in;
    int max_neurons = *max_element(in_net.n_p_l, in_net.n_p_l + in_net.n_layers - 1);
    int inout_max = max(in_net.n_ins, max_neurons);
    
    if (inout_net_free_mem > inout_max || reload == false)
    {
        if (reload == true || net_list[nets_enqueued].big_net)
        {
            cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 0", "");
            #if fpga_verbose == 1
                cout << BLUE << "   FPGA_HANDLER: NET " << nets_enqueued << "\n";
                cout << "   N_INS: " << in_net.n_ins << "\n" << "   N_P_L: ";
                for (int l = 0; l < in_net.n_layers; l++)
                    cout << in_net.n_p_l[l] << " ";
                cout << RESET << "\n";
            #endif

            net_list[nets_enqueued].net = in_net;

            // Direcciones base delos buffers en la fpga
            net_list[nets_enqueued].enqueued = true;
            net_list[nets_enqueued].loaded = false;
            net_list[nets_enqueued].solved = false;
            net_list[nets_enqueued].readed = false;
            net_list[nets_enqueued].in_out_base = NET_INOUT_BUF_SIZE/2 - inout_net_free_mem;
            net_list[nets_enqueued].params_base = NET_PARAMS_BUF_SIZE - params_net_free_mem;
            net_list[nets_enqueued].bias_base = NET_BIAS_BUF_SIZE - bias_net_free_mem;

            // Espacio reservado en los buffers de la fpga
            /* 
                Debido a las operaciones en el kernel de vload16 y vstore16 todas las dirreciones deben ser multiplo de 16,
                las reservas cumplen ya que netFPGA se asegura de que las entradas y todas las capas internas de la red
                sean múltiplos.
            */
            net_list[nets_enqueued].inout_mem_res = inout_max; 

            if (big_nets || params_net_free_mem < in_net.n_params || bias_net_free_mem < in_net.n_neurons)
            {
                net_list[nets_enqueued].big_net = true;
                net_list[nets_enqueued].params_mem_res = inout_max^2;
                net_list[nets_enqueued].bias_mem_res = 2 * inout_max;
            }
            else
            {
                net_list[nets_enqueued].big_net = false;
                net_list[nets_enqueued].params_mem_res = in_net.n_params;
                net_list[nets_enqueued].bias_mem_res = in_net.n_neurons;
            }
            cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 1", "");
        }

        inout_net_free_mem -= net_list[nets_enqueued].inout_mem_res;
        params_net_free_mem -= net_list[nets_enqueued].params_mem_res;
        bias_net_free_mem -= net_list[nets_enqueued].bias_mem_res;

        // Enqueues iniciales de entradas, parámetros y bias
        int next_ev = wr_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : wr_ev_ind + 1;

        if (reload == true || net_list[nets_enqueued].big_net)
        {
            net_list[nets_enqueued].loaded = false;
            int params2enq = net_list[nets_enqueued].big_net ? in_net.n_ins * net_list[nets_enqueued].net.n_p_l[0] : in_net.n_params / 2;
            int bias2enq = net_list[nets_enqueued].big_net ? net_list[nets_enqueued].net.n_p_l[0] : in_net.n_neurons / 2;

            if (same_in == false)
            {
                cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 2", ""); 

                err = clEnqueueWriteBuffer(wr_queue, in_out_dev, CL_FALSE, net_list[nets_enqueued].in_out_base * sizeof(long int), in_net.n_ins * sizeof(long int), inputs.data(), 1, &(rd_events[net_list[nets_enqueued].rd_event]), NULL);
                checkError(err, "Failed to enqueue inputs");
                
                #if fpga_verbose >0
                    cout << RED << "Params base: " << net_list[nets_enqueued].params_base << RESET << "\n";
                    cout << RED << "params2enq: " << params2enq << RESET << "\n";
                    for(int np=0;np<in_net.n_params;np++)                    
                        cout << RED << np << ": " << in_net.params[np] << RESET << "\n";                
                    cout << RED << "FINISH: " << in_net.n_params << RESET << "\n";

                    cout << RED << "Bias base: " << net_list[nets_enqueued].bias_base << RESET << "\n";
                    cout << RED << "bias2enq: " << bias2enq << RESET << "\n";
                    for(int np=0;np<in_net.n_neurons;np++)                    
                        cout << RED << np << ": " << in_net.bias[np] << RESET << "\n";                
                    cout << RED << "FINISH: " << in_net.n_neurons << RESET << "\n";
                #endif

                err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, net_list[nets_enqueued].params_base * sizeof(long int), params2enq * sizeof(long int), (void*)in_net.params, 0, NULL, NULL);
                checkError(err, "Failed to enqueue inputs");
                net_list[nets_enqueued].params_host_dir = params2enq;
            }
            else
            {
                err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, net_list[nets_enqueued].params_base * sizeof(long int), params2enq * sizeof(long int), in_net.params, 1, &(rd_events[net_list[nets_enqueued].rd_event]), NULL);
                checkError(err, "Failed to enqueue params");
                net_list[nets_enqueued].params_host_dir = params2enq;
            }
            err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, net_list[nets_enqueued].bias_base * sizeof(long int), bias2enq * sizeof(long int), in_net.bias, 0, NULL,  &(wr_events[next_ev]));
            checkError(err, "Failed to enqueue bias");
            net_list[nets_enqueued].bias_host_dir = bias2enq;
            cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 3", "");
        }
        else
        {
            cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 4", "");
            net_list[nets_enqueued].loaded = true;
            if (same_in == false)
            {
                err = clEnqueueWriteBuffer(wr_queue, in_out_dev, CL_FALSE, net_list[nets_enqueued].in_out_base * sizeof(long int), in_net.n_ins * sizeof(long int), inputs.data(), 1, &(rd_events[net_list[nets_enqueued].rd_event]), &(wr_events[next_ev]));
                checkError(err, "Failed to enqueue inputs");
            }
        }

        cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 5", "");
        int ins_dir = (same_in ? net_list[0].in_out_base : net_list[nets_enqueued].in_out_base)/N_NEURONS;
        int n_outs = net_list[nets_enqueued].net.n_p_l[0];
        int outs_dir = (net_list[nets_enqueued].in_out_base + NET_INOUT_BUF_SIZE/2)/N_NEURONS;
        net_list[nets_enqueued].params_exe_dir = net_list[nets_enqueued].params_base/N_NEURONS;
        net_list[nets_enqueued].bias_exe_dir = net_list[nets_enqueued].bias_base/N_NEURONS;

        vector<int> configuration = {in_net.n_ins, ins_dir, n_outs, outs_dir, net_list[nets_enqueued].params_exe_dir, net_list[nets_enqueued].bias_exe_dir};
        clSetEventCallback((wr_events[next_ev]), CL_COMPLETE, &set_args_callback, (void*)configuration.data());
        // err = clEnqueueWriteBuffer(wr_queue, configuration_dev, CL_FALSE, 0, 5 * sizeof(int), configuration.data(), 0, NULL, &(wr_events[next_ev]));
        // checkError(err, "Failed to enqueue config");
        wr_ev_ind = next_ev;
        net_list[nets_enqueued].params_exe_dir += net_list[nets_enqueued].big_net ? net_list[nets_enqueued].params_mem_res/2/N_NEURONS : in_net.n_ins*n_outs/N_NEURONS;
        net_list[nets_enqueued].bias_exe_dir += net_list[nets_enqueued].big_net ? net_list[nets_enqueued].bias_mem_res/2/N_NEURONS : n_outs/N_NEURONS;   

        cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 6", "");
        // Enqueue de primera ejecución
        net_list[nets_enqueued].wr_event = wr_ev_ind;
        next_ev = exe_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : exe_ev_ind + 1;

        cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 7", "");
        clWaitForEvents(1,&(wr_events[wr_ev_ind]));
        cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 7_1", "");
        err = clEnqueueTask(exe_queue, kernel, 1, &(wr_events[wr_ev_ind]), &(exe_events[next_ev]));
        checkError(err, "Failed to enqueue task");
        exe_ev_ind = next_ev;
        net_list[nets_enqueued].exe_event = exe_ev_ind;
        cout1(YELLOW, "   FPGA_HANDLER: ENQUEUE STEP 8", "");
        nets_enqueued++;
        return nets_enqueued;
    }
    else
    {
        cout << "FPGA: Not enough memory to enqueue\n";
        return 0;
    }
}

void fpga_handler::solve_nets()
{
    cl_int err = 0;
    for (int i = 0; i < nets_enqueued; i++)
    {
        cout1(YELLOW, "   FPGA_HANDLER: SOLVE STEP TOP_NET", "");

        // Recorre las capas
        for (int l = 1; l < net_list[i].net.n_layers; l++)
        {
            #if fpga_verbose == 1
            cout << BLUE << "   FPGA_HANDLER: NET " << i << " Intern values: \n";
            vector<long int> inter_values = read_net(i + 1, net_list[i].net.n_p_l[l-1]);
            cout << "   ";
            for (int v = 0; v < inter_values.size(); v++)
                cout << inter_values[v] << " ";
            cout << RESET << "\n";
            #endif  

            int next_ev;
            int n_ins, ins_dir, n_outs, outs_dir;
            cout1(YELLOW, "   FPGA_HANDLER: SOLVE STEP LAYER ", l);

            n_ins = net_list[i].net.n_p_l[l - 1];
            ins_dir = (l % 2 == 0 ? net_list[i].in_out_base : net_list[i].in_out_base + NET_INOUT_BUF_SIZE / 2)/N_NEURONS;
            n_outs = net_list[i].net.n_p_l[l];
            outs_dir = (l % 2 == 0 ? net_list[i].in_out_base + NET_INOUT_BUF_SIZE / 2 : net_list[i].in_out_base)/N_NEURONS;            
            
            // Enqueue de bias y parametros
            if (net_list[i].big_net || (!net_list[i].loaded && l == 1))
            {
                int params2enq;
                int bias2enq;

                if (net_list[i].big_net)
                {
                    params2enq = net_list[i].net.n_p_l[l - 1] * net_list[i].net.n_p_l[l];
                    bias2enq = net_list[i].net.n_p_l[l];
                }
                else
                {
                    params2enq = net_list[i].net.n_params / 2;
                    bias2enq = net_list[i].net.n_neurons / 2;
                }

                cout1(YELLOW, "   FPGA_HANDLER: SOLVE STEP ENQ_PAR_B", "");

                net_list[i].params_enq_dir = l % 2 == 0 ? net_list[i].params_base : net_list[i].params_base + net_list[i].params_mem_res / 2;
                cout1(YELLOW, "   FPGA_HANDLER: PARAMS2ENQ ", params2enq);
                cout1(YELLOW, "   FPGA_HANDLER: PARAMS ENQ DIR ", net_list[i].params_enq_dir);
                cout1(YELLOW, "   FPGA_HANDLER: PARAMS HOST DIR ", net_list[i].params_host_dir);
                err = clEnqueueWriteBuffer(wr_queue, params_dev, CL_FALSE, net_list[i].params_enq_dir * sizeof(long int), params2enq * sizeof(long int), &(net_list[i].net.params[net_list[i].params_host_dir]), 0, NULL, NULL);
                checkError(err, "Failed to enqueue params");
                net_list[i].params_host_dir += params2enq;
                
                net_list[i].bias_enq_dir = l % 2 == 0 ? net_list[i].bias_base : net_list[i].bias_base + net_list[i].bias_mem_res / 2;
                cout1(YELLOW, "   FPGA_HANDLER: BIAS2ENQ ", bias2enq);
                cout1(YELLOW, "   FPGA_HANDLER: BIAS ENQ DIR ", net_list[i].bias_enq_dir);
                cout1(YELLOW, "   FPGA_HANDLER: BIAS HOST DIR ", net_list[i].bias_host_dir);
                next_ev = wr_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : wr_ev_ind + 1;
                err = clEnqueueWriteBuffer(wr_queue, bias_dev, CL_FALSE, net_list[i].bias_enq_dir * sizeof(long int), bias2enq * sizeof(long int), &(net_list[i].net.bias[net_list[i].bias_host_dir]), 0, NULL, &(wr_events[next_ev]));
                checkError(err, "Failed to enqueue bias");
                net_list[i].bias_host_dir += bias2enq;                
                wr_ev_ind = next_ev;

                cout1(YELLOW, "   FPGA_HANDLER: SOLVE STEP FINISH_ENQ", "");
            }

            cout1(YELLOW, "   FPGA_HANDLER: SOLVE STEP ENQ_CONF", "");
            // Enqueue de configuracion
            vector<int> configuration = {n_ins, ins_dir, n_outs, outs_dir, net_list[i].params_exe_dir, net_list[i].bias_exe_dir};
            cout1(YELLOW, "   FPGA_HANDLER: N_INS ", n_ins);
            cout1(YELLOW, "   FPGA_HANDLER: INS_DIR ", ins_dir);
            cout1(YELLOW, "   FPGA_HANDLER: OUTS_DIR ", outs_dir);
            cout1(YELLOW, "   FPGA_HANDLER: PARAMS_EXE_DIR ", net_list[i].params_exe_dir);
            cout1(YELLOW, "   FPGA_HANDLER: BIAS_EXE_DIR ", net_list[i].bias_exe_dir);
            
            clSetEventCallback(exe_events[exe_ev_ind], CL_COMPLETE, &set_args_callback, (void*)configuration.data());
            // int next_ev = wr_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : wr_ev_ind + 1;
            // err = clEnqueueWriteBuffer(wr_queue, configuration_dev, CL_TRUE, 0, 5 * sizeof(int), configuration.data(), 1, &(exe_events[exe_ev_ind]), &(wr_events[next_ev]));
            // checkError(err, "Failed to enqueue inputs");
            // wr_ev_ind = next_ev;
            net_list[i].params_exe_dir = net_list[i].big_net ? (net_list[i].params_enq_dir)/N_NEURONS : net_list[i].params_exe_dir + (net_list[i].net.n_p_l[l-1]*net_list[i].net.n_p_l[l])/N_NEURONS;
            net_list[i].bias_exe_dir = net_list[i].big_net ? (net_list[i].bias_enq_dir)/N_NEURONS : net_list[i].bias_exe_dir + (net_list[i].net.n_p_l[l])/N_NEURONS;

            cout1(YELLOW, "   FPGA_HANDLER: SOLVE STEP FINISH_ENQ", "");
            // Enqueue de ejecución
            net_list[i].wr_event = wr_ev_ind;
            next_ev = exe_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : exe_ev_ind + 1;
            
            cout1(YELLOW, "   FPGA_HANDLER: SOLVE STEP TASK", "");
            // clFlush(wr_queue);
            // clFlush(exe_queue);
            // clWaitForEvents(1,&(wr_events[wr_ev_ind]));
            clWaitForEvents(1,&(exe_events[exe_ev_ind]));
            err = clEnqueueTask(exe_queue, kernel, 1, &(wr_events[wr_ev_ind]), &(exe_events[next_ev]));
            checkError(err, "Failed to enqueue task");
            exe_ev_ind = next_ev;
            net_list[i].exe_event = exe_ev_ind;
            cout1(YELLOW, "   FPGA_HANDLER: SOLVE STEP END_TASK", "");

            net_list[i].layer_parity ^= 1;          
        }
        net_list[i].loaded = true;
    }
}

std::vector<long int> fpga_handler::read_net(int identifier, int nouts, bool all)
{
    cl_int err = 0;
    cout1(YELLOW, "   FPGA_HANDLER: READ STEP 0", "");
    int id = identifier - 1;
    int n_outs = nouts == 0 ? net_list[id].net.n_p_l[net_list[id].net.n_layers - 1] : nouts;
    int dir_outs = net_list[id].in_out_base + net_list[id].layer_parity*NET_INOUT_BUF_SIZE/2;
    int next_ev = rd_ev_ind == MAX_SZ_ENQUEUE - 1 ? 0 : rd_ev_ind + 1;
    if(all){
        n_outs = NET_INOUT_BUF_SIZE;
        dir_outs = 0;
    }    
    vector<long int> outs(n_outs, 0);

    cout1(YELLOW, "   FPGA_HANDLER: READ STEP 1", "");
    cout1(YELLOW, "   FPGA_HANDLER: READING AT ", dir_outs);
#if fpga_verbose == 1
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    _show_events();
#endif
    err = clEnqueueReadBuffer(rd_queue, in_out_dev, CL_TRUE, dir_outs * sizeof(long int), n_outs * sizeof(long int), outs.data(), 1, &(exe_events[net_list[id].exe_event]), &(rd_events[next_ev]));
    checkError(err, "Failed to enqueue read outs");
    if (nouts == 0){
        net_list[id].solved = true;
        net_list[id].readed = true;
        rd_ev_ind = next_ev;
        net_list[id].rd_event = rd_ev_ind;
        cout1(YELLOW, "   FPGA_HANDLER: READ STEP 2", "");

        inout_net_free_mem += net_list[id].inout_mem_res;
        params_net_free_mem += net_list[id].params_mem_res;
        bias_net_free_mem += net_list[id].bias_mem_res;
        nets_enqueued--;
    }

    return outs;
}

void fpga_handler::_cleanup()
{
    cout1(YELLOW, "   IN CLEANUP", "");
    if (im_the_handler)
    {
        cout1(YELLOW, "   CLEANING", "");
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
    cl_int err = 0;
    cout1(YELLOW, "   INIT_PROGRAM", "");
    // Limpia la configuración previa
    // _cleanup();
    
    // Configuración principal de la FPGA
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "Failed to get platforms");

    cout1(YELLOW, "   PLATFORMS", "");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);
    checkError(err, "Failed to find device");

    cout1(YELLOW, "   DEVICES", "");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "Failed to create context");

    cout1(YELLOW, "   C_QUEUES", "");

    wr_queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Failed to create queue");
    rd_queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Failed to create queue");
    exe_queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Failed to create queue");

    cout1(YELLOW, "   PROGRAM", "");

    const char *char_name = prg_name.c_str();
    std::string binary_file = getBoardBinaryFile(char_name, device); // Coge el aocx
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(err, "Failed to create program");

    // Inicializa los primeros eventos
    // wr_events[0] = clCreateUserEvent(context, NULL);
    // clSetUserEventStatus(wr_events[0], CL_COMPLETE);
    exe_events[0] = clCreateUserEvent(context, &err);
    checkError(err, "Failed event");
    rd_events[0] = clCreateUserEvent(context, &err);
    checkError(err, "Failed event");
    clSetUserEventStatus(exe_events[0], CL_COMPLETE);
    clSetUserEventStatus(rd_events[0], CL_COMPLETE);

    // Reserva de memoria en el PC
    switch (prg_kind)
    {
    case NN:
        // net_outs_dirs.reserve(MAX_SZ_ENQUEUE);
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
        
        cout1(YELLOW, "   KERNEL", "");
        kernel = clCreateKernel(program, NET_KERNEL_NAME, &err);
        checkError(err, "Failed to create kernel");

        
        cout1(YELLOW, "   BUFFERS", "");
        inout_net_free_mem = NET_INOUT_BUF_SIZE / 2;
        in_out_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, NET_INOUT_BUF_SIZE * sizeof(long int), NULL, &err); // CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        checkError(err, "Failed to create buffer inouts");
        params_net_free_mem = NET_PARAMS_BUF_SIZE;
        params_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, NET_PARAMS_BUF_SIZE * sizeof(long int), NULL, &err);
        checkError(err, "Failed to create buffer params");
        bias_net_free_mem = NET_BIAS_BUF_SIZE;
        bias_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, NET_BIAS_BUF_SIZE * sizeof(long int), NULL, &err);
        checkError(err, "Failed to create buffer bias");
        // configuration_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, 5 * sizeof(int), NULL, &err);
        // checkError(err, "Failed to create buffer bias");
        
        cout1(YELLOW, "   ARGS", "");
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_out_dev);
        checkError(err, "Failed to set arg inouts");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_dev);
        checkError(err, "Failed to set arg params");
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bias_dev);
        checkError(err, "Failed to set arg bias");
        // err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&configuration_dev);
        // checkError(err, "Failed to set arg configuration");
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

void fpga_handler::_show_events()
{
    cout << "N_WR_EVENTS " << wr_ev_ind << RESET << "\n";
    for (int i = 0; i <= wr_ev_ind; i++)
    {
        cl_int status;
        size_t sz_st = sizeof(cl_int);
        clGetEventInfo(wr_events[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sz_st, &status, &sz_st);
        switch (status)
        {
        case CL_QUEUED:
            cout << MAGENTA << "WR_EVENT " << i << ": QUEUED" << RESET << "\n";
            break;
        case CL_SUBMITTED:
            cout << MAGENTA << "WR_EVENT " << i << ": SUBMITED" << RESET << "\n";
            break;
        case CL_RUNNING:
            cout << MAGENTA << "WR_EVENT " << i << ": RUNNING" << RESET << "\n";
            break;
        case CL_COMPLETE:
            cout << MAGENTA << "WR_EVENT " << i << ": COMPLETED" << RESET << "\n";
            break;
        default:
            cout << MAGENTA << "WR_EVENT " << i << ": UNKNOWN" << RESET << "\n";
        }
    }
    cout << "N_EXE_EVENTS " << exe_ev_ind + 1 << RESET << "\n";
    for (int i = 0; i <= exe_ev_ind; i++)
    {
        cl_int status;
        size_t sz_st = sizeof(cl_int);
        clGetEventInfo(exe_events[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sz_st, &status, &sz_st);
        switch (status)
        {
        case CL_QUEUED:
            cout << MAGENTA << "EXE_EVENT " << i << ": QUEUED" << RESET << "\n";
            break;
        case CL_SUBMITTED:
            cout << MAGENTA << "EXE_EVENT " << i << ": SUBMITED" << RESET << "\n";
            break;
        case CL_RUNNING:
            cout << MAGENTA << "EXE_EVENT " << i << ": RUNNING" << RESET << "\n";
            break;
        case CL_COMPLETE:
            cout << MAGENTA << "EXE_EVENT " << i << ": COMPLETED" << RESET << "\n";
            break;
        default:
            cout << MAGENTA << "EXE_EVENT " << i << ": UNKNOWN" << RESET << "\n";
        }
    }
    cout << "N_RD_EVENTS " << rd_ev_ind + 1 << RESET << "\n";
    for (int i = 0; i <= rd_ev_ind + 1; i++)
    {
        cl_int status;
        size_t sz_st = sizeof(cl_int);
        clGetEventInfo(rd_events[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sz_st, &status, &sz_st);
        switch (status)
        {
        case CL_QUEUED:
            cout << MAGENTA << "RD_EVENT " << i << ": QUEUED" << RESET << "\n";
            break;
        case CL_SUBMITTED:
            cout << MAGENTA << "RD_EVENT " << i << ": SUBMITED" << RESET << "\n";
            break;
        case CL_RUNNING:
            cout << MAGENTA << "RD_EVENT " << i << ": RUNNING" << RESET << "\n";
            break;
        case CL_COMPLETE:
            cout << MAGENTA << "RD_EVENT " << i << ": COMPLETED" << RESET << "\n";
            break;
        default:
            cout << MAGENTA << "RD_EVENT " << i << ": UNKNOWN" << RESET << "\n";
        }
    }
}

void cleanup()
{
    cl_ptr->_cleanup();
    // fpga::fpga_handler::ptr_handler->_cleanup();
}

void set_args_callback(cl_event event, cl_int event_command_exec_status, void *user_data){

    cl_int err;
    int n_ins = ((int*)user_data)[0]/N_INS;
    cout1(YELLOW, "   FPGA_HANDLER: n_ins", n_ins);
    int in_dir = ((int*)user_data)[1];
    cout1(YELLOW, "   FPGA_HANDLER: in_dir", in_dir);
    int n_outs = ((int*)user_data)[2]/N_NEURONS;
    cout1(YELLOW, "   FPGA_HANDLER: n_outs", n_outs);
    int out_dir = ((int*)user_data)[3];
    cout1(YELLOW, "   FPGA_HANDLER: out_dir", out_dir);
    int params_dir = ((int*)user_data)[4];
    cout1(YELLOW, "   FPGA_HANDLER: params_dir", params_dir);
    int bias_dir = ((int*)user_data)[5];
    cout1(YELLOW, "   FPGA_HANDLER: bias_dir", bias_dir);

    err = clSetKernelArg(kernel, 3, sizeof(int), (void *)&n_ins);
    checkError(err, "set n_ins");
    err = clSetKernelArg(kernel, 4, sizeof(int), (void *)&in_dir);
    checkError(err, "set in_dir");
    err = clSetKernelArg(kernel, 5, sizeof(int), (void *)&n_outs);
    checkError(err, "set n_outs");
    err = clSetKernelArg(kernel, 6, sizeof(int), (void *)&out_dir);
    checkError(err, "set out_dir");
    err = clSetKernelArg(kernel, 7, sizeof(int), (void *)&params_dir);
    checkError(err, "set params_dir");
    err = clSetKernelArg(kernel, 8, sizeof(int), (void *)&bias_dir);
    checkError(err, "set bias_dir");
}