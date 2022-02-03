#include <netFPGA.h>
#include <math.h>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <chrono>
#include "CL/cl.hpp"
#include "AOCLUtils/aocl_utils.h"

#define BATCH_SIZE 24

namespace fpga
{
    using namespace aocl_utils;
    using namespace std;
    using namespace chrono;

    //OpenCL & FPGA variables
    int net_fpga_counter = 0;
    bool program_init = false;
    bool forward_kernel_init = false;
    bool reload_params = true;        

    net::image_set out_image;

    // OpenCL runtime configuration
    cl_platform_id g_platform = NULL;
    cl_device_id g_device = NULL;
    cl_context g_context = NULL;
    cl_command_queue g_queue_in = NULL;
    cl_command_queue g_queue_out = NULL;
    cl_program g_program = NULL;
    cl_kernel g_kernel_in = NULL;
    cl_kernel g_kernel_out = NULL;
    cl_int g_err = 0;

    cl_mem g_inputs_dev = NULL;
    cl_mem g_params_dev = NULL;
    cl_mem g_bias_dev = NULL;
    cl_mem g_outs_dev = NULL;
    cl_mem g_npl_dev = NULL;

    int g_n_ins_buff = 0;
    int g_n_layers_buff = 0;
    int *g_n_p_l_buff = NULL;

    DATA_TYPE *g_params_buff = NULL;
    DATA_TYPE *g_bias_buff = NULL;
    DATA_TYPE *g_inputs_buff = NULL;
    DATA_TYPE *g_oputputs_buff = NULL;

    cl_event g_init_event = NULL;
    cl_event g_finish_event = NULL;

    cl_event g_im_init_event[BATCH_SIZE] = {nullptr};
    cl_event g_im_finish_event[BATCH_SIZE] = {nullptr};
    cl_event g_im_read_event[BATCH_SIZE] = {nullptr};

    bool g_are_images_init = false;
    unsigned char *g_in_images[BATCH_SIZE] = {nullptr};
    unsigned char *g_out_images[BATCH_SIZE] = {nullptr};
    int g_wr_batch_cnt = 0;
    int g_rd_batch_cnt = 0;
    int g_free_batch = BATCH_SIZE;

    net_fpga::net_fpga(const net::net_data &data, bool derivate, bool random)
        : n_layers(data.n_p_l.size()), n_sets(0), gradient_init(false), gradient_performance(0), forward_performance(0), n_ins(data.n_ins)

    {
        net_fpga_counter++;

        n_p_l = new int[n_layers];
        n_neurons = 0;
        n_params = 0;

        for (int i = 0; i < n_layers; i++)
        {
            n_p_l[i] = data.n_p_l[i];
            n_neurons += n_p_l[i];
            if (i == 0)
                n_params += n_p_l[i] * n_ins;
            else
                n_params += n_p_l[i] * n_p_l[i - 1];
        }

        params = new float[n_params];
        activations = 1; // 1 -> RELU2
        bias = new float[n_neurons];

        if (random)
        {
            for (int i = 0; i < n_params; i++)
                params[i] = float(rand() % 200 - 100) / 100;
            for (int i = 0; i < n_neurons; i++)
                bias[i] = float(rand() % 200 - 100) / 100;
        }
        else
        {
            int param_cnt = 0;
            int neuron_cnt = 0;

            for (int i = 0; i < n_layers; i++)
            {
                for (int j = 0; j < n_p_l[i]; j++)
                {
                    for (int k = 0; k < data.params[i][j].size(); k++)
                    {
                        params[param_cnt] = data.params[i][j][k];
                        param_cnt++;
                    }
                    bias[neuron_cnt] = data.bias[i][j];
                    neuron_cnt++;
                }
            }
        }
        // cout << "FPGA NET: CREATED\n";
    }

    net_fpga::net_fpga(net_fpga &&rh) : n_ins(rh.n_ins),
                                        n_layers(rh.n_layers),
                                        n_neurons(rh.n_neurons),
                                        n_params(rh.n_params),
                                        activations(rh.activations),
                                        n_sets(rh.n_sets),
                                        gradient_init(rh.gradient_init)

    {
        net_fpga_counter++;

        delete[] n_p_l;
        delete[] params;
        delete[] bias;

        n_p_l = rh.n_p_l;
        params = rh.params;
        bias = rh.bias;

        rh.n_p_l = NULL;
        rh.params = NULL;
        rh.bias = NULL;
    }

    net_fpga &net_fpga::operator=(net_fpga &&rh)
    {
        if (this != &rh)
        {
            net_fpga_counter++;

            n_ins = rh.n_ins;
            n_layers = rh.n_layers;
            n_neurons = rh.n_neurons;
            n_params = rh.n_params;

            activations = rh.activations;
            n_sets = rh.n_sets;
            gradient_init = rh.gradient_init;

            delete[] n_p_l;
            delete[] params;
            delete[] bias;

            n_p_l = rh.n_p_l;
            params = rh.params;
            bias = rh.bias;

            rh.n_p_l = NULL;
            rh.params = NULL;
            rh.bias = NULL;
        }

        return *this;
    }

    net_fpga &net_fpga::operator=(const net_fpga &rh)
    {
        if (this != &rh)
        {
            net_fpga_counter++;

            bool del = (n_ins != rh.n_ins) || (n_layers != rh.n_layers);
            for (int i = 0; i < n_layers && !del; i++)
                del = n_p_l[i] != rh.n_p_l[i];

            if (del)
            {
                n_ins = rh.n_ins;
                n_layers = rh.n_layers;
                n_neurons = rh.n_neurons;
                n_params = rh.n_params;

                activations = rh.activations;
                n_sets = rh.n_sets;
                gradient_init = rh.gradient_init;

                delete[] n_p_l;
                delete[] params;
                delete[] bias;

                n_p_l = new int[n_layers];
                params = new float[n_params];
                bias = new float[n_neurons];

                for (int i = 0; i < n_params; i++)
                    params[i] = rh.params[i];

                for (int i = 0; i < n_neurons; i++)
                    bias[i] = rh.bias[i];
            }
        }

        return *this;
    }

    net::net_data net_fpga::get_net_data() // TODO:implementar
    {
        net::net_data data;
        data.n_ins = n_ins;
        data.n_layers = n_layers;

        data.params.reserve(n_layers);
        int params_cnt = 0;
        int neurons_cnt = 0;

        for (int i = 0; i < n_layers; i++)
        {
            data.n_p_l[i] = n_p_l[i];
            int n_params = (i == 0) ? n_ins : n_p_l[i - 1];
            data.params.emplace_back(n_p_l[i], vector<DATA_TYPE>(n_params));
            data.bias.emplace_back(n_p_l[i]);

            for (int j = 0; j < n_p_l[i]; j++)
            {
                data.bias[i][j] = bias[neurons_cnt];
                neurons_cnt++;

                for (int k = 0; k < n_ins; k++)
                {
                    data.params[i][j][k] = params[params_cnt];
                    params_cnt++;
                }
            }
        }

        return data;
    }

    vector<DATA_TYPE> net_fpga::launch_forward(const vector<DATA_TYPE> &inputs) //* returns result
    {
        // cout << "FPGA NET: FORWARD\n";
        if (program_init == false)
        {
            net_fpga::_init_program(NET_KERNEL);
            program_init = true;
            // cout << "FPGA NET: PROGRAM CREATED\n";
        }
        if (forward_kernel_init == false)
        {
            net_fpga::_init_kernel("network_v1");
            forward_kernel_init = true;
            // cout << "FPGA NET: KERNEL CREATED\n";
        }
        if (g_n_ins_buff != n_ins || g_n_layers_buff != n_layers || g_n_p_l_buff != n_p_l || g_params_buff != params)
        {
            net_fpga::_load_params();
            delete[] g_inputs_buff;
            g_inputs_buff = new DATA_TYPE[n_ins];
            // cout << "FPGA NET: PARAMS LOADED\n";
        }else

#ifdef PERFORMANCE
        auto start = high_resolution_clock::now();
#endif

        for (int i = 0; i < n_ins; i++)
            g_inputs_buff[i] = inputs[i];

        DATA_TYPE outs[n_p_l[n_layers - 1]];

        // cl_event aux;

        g_err = clEnqueueWriteBuffer(g_queue_in, g_inputs_dev, CL_FALSE, 0, n_ins * sizeof(DATA_TYPE), g_inputs_buff, 1, &g_finish_event, &g_init_event);
        checkError(g_err, "Failed to enqueue inputs");
        g_err = clEnqueueTask(g_queue_in, g_kernel_in, 1, &g_init_event, &g_finish_event);
        checkError(g_err, "Failed to enqueue task");
        g_err = clEnqueueReadBuffer(g_queue_in, g_outs_dev, CL_TRUE, 0, n_p_l[n_layers - 1] * sizeof(DATA_TYPE), outs, 1, &g_finish_event, NULL);
        checkError(g_err, "Failed to enqueue read outs");

#ifdef PERFORMANCE
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        forward_performance = duration.count();
#endif
        std::vector<DATA_TYPE> vec_out(n_p_l[n_layers - 1]);
        // for (int i = 0; i < n_p_l[n_layers - 1]; i++)
        //     vec_out[i] = outs[i];

        return vec_out;
    }

    void net_fpga::filter_image(const net::image_set &set)
    {
        // cout << "FPGA NET: FORWARD\n";
        if (program_init == false)
        {
            net_fpga::_init_program(IMAGE_KERNEL);
            program_init = true;
            // cout << "FPGA NET: PROGRAM CREATED\n";
        }
        if (forward_kernel_init == false)
        {
            net_fpga::_init_kernel("image_process", set);
            forward_kernel_init = true;
            // cout << "FPGA NET: KERNEL CREATED\n";
        }

        if (g_free_batch > 0)
        {
            g_free_batch--;

            for (int i = 0; i < set.original_h * set.original_w; i++)
                g_in_images[g_wr_batch_cnt][i] = set.resized_image_data[i];

            unsigned char a = g_in_images[g_wr_batch_cnt][0];
            // cout << "Enqueuing image in\n";
            // unsigned char *test_buff = new unsigned char[set.original_h * set.original_w];
            // in_images[wr_batch_cnt]
            g_err = clEnqueueWriteBuffer(g_queue_in, g_inputs_dev, CL_FALSE, 0, set.original_h * set.original_w * sizeof(unsigned char), g_in_images[g_wr_batch_cnt], 1, &(g_im_finish_event[g_wr_batch_cnt]), &(g_im_init_event[g_wr_batch_cnt]));
            checkError(g_err, "Failed to enqueue inputs");
            // delete[] test_buff;
            int next_wr_batch = g_wr_batch_cnt == (BATCH_SIZE - 1) ? 0 : g_wr_batch_cnt + 1;

            // cout << "Enqueuing image in kernel\n";
            g_err = clEnqueueTask(g_queue_in, g_kernel_in, 1, &(g_im_init_event[g_wr_batch_cnt]), NULL);
            checkError(g_err, "Failed to enqueue task");

            // cout << "Enqueuing image out kernel\n";
            g_err = clEnqueueTask(g_queue_out, g_kernel_out, 1, &(g_im_init_event[g_wr_batch_cnt]), &(g_im_finish_event[next_wr_batch]));
            checkError(g_err, "Failed to enqueue task");

            // cout << "Enqueuing image out\n";
            g_err = clEnqueueReadBuffer(g_queue_out, g_outs_dev, CL_FALSE, 0, set.original_h * set.original_w * sizeof(unsigned char), g_out_images[g_wr_batch_cnt], 1, &(g_im_finish_event[next_wr_batch]), &(g_im_read_event[g_wr_batch_cnt]));
            checkError(g_err, "Failed to enqueue read outs");
            g_wr_batch_cnt = next_wr_batch;

            // cout << "Leaving fpga code\n";
        }
        else
        {
            cout << "PILA LLENA\n";
        }
    }

    net::image_set net_fpga::get_filtered_image()
    {
        // net::image_set out_image;
        out_image.original_x_pos = 0;
        out_image.original_y_pos = 0;
        out_image.original_h = IMAGE_HEIGHT;
        out_image.original_w = IMAGE_WIDTH;

        if (g_free_batch < BATCH_SIZE)
        {
            // cout << "Leyendo datos\n";
            // cout << "Freebatch " << free_batch << "\n";
            // cout << CL_QUEUED << "\n";
            // cout << CL_SUBMITTED << "\n";
            // cout << CL_RUNNING << "\n";
            // cout << CL_COMPLETE << "\n";
            
            // size_t size = sizeof(cl_int);
            // cl_int *ptr = new cl_int[1];
            // g_err = clGetEventInfo(g_im_finish_event[0],CL_EVENT_COMMAND_EXECUTION_STATUS, size, (void*)ptr, NULL);
            // checkError(g_err, "Failed get event info");
            // cout << "Finish event 0: " << ptr[0] << "\n";
            // clGetEventInfo(g_im_finish_event[1],CL_EVENT_COMMAND_EXECUTION_STATUS, size, (void*)ptr, NULL);
            // cout << "Finish event 1: " << ptr[0] << "\n";
            // clGetEventInfo(g_im_init_event[0],CL_EVENT_COMMAND_EXECUTION_STATUS, size, (void*)ptr, NULL);
            // cout << "Init event 0: " << ptr[0] << "\n";
            // delete[] ptr;
            g_free_batch++;
            clWaitForEvents(1, &(g_im_read_event[g_rd_batch_cnt]));
            // out_image.resized_image_data.reserve(IMAGE_HEIGHT * IMAGE_WIDTH);

            for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++)
                out_image.resized_image_data[i] = g_out_images[g_rd_batch_cnt][i];
                // out_image.resized_image_data.emplace_back(g_out_images[g_rd_batch_cnt][i]);

            g_rd_batch_cnt = g_rd_batch_cnt == (BATCH_SIZE - 1) ? 0 : g_rd_batch_cnt + 1;
            // cout << "Datos leidos\n";
        }
        else
        {
            // cout << "PILA VACIA\n";
        }

        // cout << out_image.resized_image_data.size() << "\n";
        return out_image;
    }

    void net_fpga::_init_program(int prg)
    {
        // clMapHostPipeIntelFPGA = (void * (*) (cl_mem, cl_map_flags, size_t, size_t *, cl_int *)) clGetExtensionFunctionAddress("clMapHostPipeIntelFPGA");
        // clUnmapHostPipeIntelFPGA = (cl_int (*) (cl_mem, void *, size_t, size_t *)) clGetExtensionFunctionAddress("clUnmapHostPipeIntelFPGA");
        out_image.resized_image_data.reserve(IMAGE_HEIGHT * IMAGE_WIDTH);
        for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++)
            out_image.resized_image_data.emplace_back(0);

        //Platform, device and context

        //cl_platform_id platform;
        g_err = clGetPlatformIDs(1, &g_platform, NULL);
        checkError(g_err, "Failed to get platforms");

        //cl_device_id device;
        g_err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &g_device, NULL);
        checkError(g_err, "Failed to find device");

        //cl_context context;
        g_context = clCreateContext(NULL, 1, &g_device, NULL, NULL, &g_err);
        checkError(g_err, "Failed to create context");

        //cl_command_queue queue;
        g_queue_in = clCreateCommandQueue(g_context, g_device, 0, &g_err);
        checkError(g_err, "Failed to create queue");
        g_queue_out = clCreateCommandQueue(g_context, g_device, 0, &g_err);
        checkError(g_err, "Failed to create queue");

        //cl_program
        const char *prg_name = (prg == NET_KERNEL ? "vector_kernels" : "image_kernels");
        std::string binary_file = getBoardBinaryFile(prg_name, g_device); //Coge el aocx
        g_program = createProgramFromBinary(g_context, binary_file.c_str(), &g_device, 1);

        g_err = clBuildProgram(g_program, 0, NULL, "", NULL, NULL);
        checkError(g_err, "Failed to create program");

        g_init_event = clCreateUserEvent(g_context, NULL);
        g_finish_event = clCreateUserEvent(g_context, NULL);

        clSetUserEventStatus(g_init_event, CL_COMPLETE);
        clSetUserEventStatus(g_finish_event, CL_COMPLETE);
    }

    void net_fpga::_init_kernel(const char *kernel_name)
    {
        // kernel = clCreateKernel(program, "my_kernel", &err);
        int n_bytes_npl = n_layers * sizeof(int);
        int n_bytes_inputs = n_ins * sizeof(DATA_TYPE);
        int n_bytes_params = n_params * sizeof(DATA_TYPE);
        int n_bytes_bias = n_neurons * sizeof(DATA_TYPE);
        int n_bytes_outs = n_p_l[n_layers - 1] * sizeof(DATA_TYPE);

        // cout << "   Creating buffers:\n";
        g_inputs_dev = clCreateBuffer(g_context, CL_MEM_READ_ONLY, n_bytes_inputs, NULL, &g_err); //CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        checkError(g_err, "Failed to create buffer inputs");
        g_params_dev = clCreateBuffer(g_context, CL_MEM_READ_ONLY, n_bytes_params, NULL, &g_err);
        checkError(g_err, "Failed to create buffer params");
        g_bias_dev = clCreateBuffer(g_context, CL_MEM_READ_ONLY, n_bytes_bias, NULL, &g_err);
        checkError(g_err, "Failed to create buffer bias");
        g_outs_dev = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, n_bytes_outs, NULL, &g_err);
        checkError(g_err, "Failed to create buffer outputs");
        g_npl_dev = clCreateBuffer(g_context, CL_MEM_READ_ONLY, n_bytes_npl, NULL, &g_err);
        checkError(g_err, "Failed to create buffer npl");

        g_kernel_in = clCreateKernel(g_program, kernel_name, &g_err);
        checkError(g_err, "Failed to create kernel");

        // cout << "   Setting Args:\n";
        g_err = clSetKernelArg(g_kernel_in, 0, sizeof(cl_mem), (void *)&g_inputs_dev);
        checkError(g_err, "Failed to set arg inputs");
        g_err = clSetKernelArg(g_kernel_in, 1, sizeof(cl_mem), (void *)&g_params_dev);
        checkError(g_err, "Failed to set arg params");
        g_err = clSetKernelArg(g_kernel_in, 2, sizeof(cl_mem), (void *)&g_bias_dev);
        checkError(g_err, "Failed to set arg bias");
        g_err = clSetKernelArg(g_kernel_in, 3, sizeof(cl_mem), (void *)&g_outs_dev);
        checkError(g_err, "Failed to set arg outputs");
        g_err = clSetKernelArg(g_kernel_in, 4, sizeof(cl_mem), (void *)&g_npl_dev);
        checkError(g_err, "Failed to set arg npl");
        // err = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&n_layers_buff);
        // checkError(err, "Failed to set arg n_layers");
        // err = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&n_ins_buff);
        // checkError(err, "Failed to set arg n_ins");
    }

    void net_fpga::_init_kernel(const char *kernel_name, const net::image_set &set)
    {
        // cout << "   Reserving images:\n";
        if (!g_are_images_init)
        {
            for (int i = 0; i < BATCH_SIZE; i++)
            {
                g_in_images[i] = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT]();
                g_out_images[i] = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT]();
            }
            g_are_images_init = true;
        }
        // kernel = clCreateKernel(program, "my_kernel", &err);
        int n_bytes_in_image = set.original_h * set.original_w * sizeof(unsigned char);
        int n_bytes_out_image = set.original_h * set.original_w * sizeof(unsigned char);

        // cout << "   Creating buffers:\n";
        g_inputs_dev = clCreateBuffer(g_context, CL_MEM_READ_ONLY, n_bytes_in_image, NULL, &g_err); //CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        checkError(g_err, "Failed to create buffer inputs");
        g_outs_dev = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, n_bytes_out_image, NULL, &g_err);
        checkError(g_err, "Failed to create buffer outputs");

        // cout << "   Creating kernels:\n";
        g_kernel_in = clCreateKernel(g_program, "image_process", &g_err);
        checkError(g_err, "Failed to create kernel");
        g_kernel_out = clCreateKernel(g_program, "image_borders", &g_err);
        checkError(g_err, "Failed to create kernel");

        // cout << "   Setting Args:\n";
        g_err = clSetKernelArg(g_kernel_in, 0, sizeof(cl_mem), (void *)&g_inputs_dev);
        checkError(g_err, "Failed to set arg inputs");
        g_err = clSetKernelArg(g_kernel_out, 0, sizeof(cl_mem), (void *)&g_outs_dev);
        checkError(g_err, "Failed to set arg outputs");
        // err = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&n_layers_buff);
        // checkError(err, "Failed to set arg n_layers");
        // err = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&n_ins_buff);
        // checkError(err, "Failed to set arg n_ins");

        // im_init_event[0] = clCreateUserEvent(context, NULL);
        g_im_finish_event[0] = clCreateUserEvent(g_context, NULL);

        // clSetUserEventStatus(im_init_event[0], CL_COMPLETE);
        clSetUserEventStatus(g_im_finish_event[0], CL_COMPLETE);
    }

    void net_fpga::_load_params()
    {
        g_n_ins_buff = n_ins;
        g_n_layers_buff = n_layers;
        g_n_p_l_buff = n_p_l;

        g_params_buff = params;
        g_bias_buff = bias;

        int n_bytes_npl = n_layers * sizeof(int);
        int n_bytes_inputs = n_ins * sizeof(DATA_TYPE);
        int n_bytes_params = n_params * sizeof(DATA_TYPE);
        int n_bytes_bias = n_neurons * sizeof(DATA_TYPE);
        int n_bytes_outs = n_p_l[n_layers - 1] * sizeof(DATA_TYPE);

        g_err = clSetKernelArg(g_kernel_in, 5, sizeof(cl_int), (void *)&g_n_layers_buff);
        checkError(g_err, "Failed to set arg n_layers");
        g_err = clSetKernelArg(g_kernel_in, 6, sizeof(cl_int), (void *)&g_n_ins_buff);
        checkError(g_err, "Failed to set arg n_ins");

        cl_event params_ev, bias_event;

        g_err = clEnqueueWriteBuffer(g_queue_in, g_params_dev, CL_FALSE, 0, n_bytes_params, g_params_buff, 0, NULL, &params_ev);
        checkError(g_err, "Failed to launch enqueue params");
        g_err = clEnqueueWriteBuffer(g_queue_in, g_bias_dev, CL_FALSE, 0, n_bytes_bias, g_bias_buff, 1, &params_ev, &bias_event);
        checkError(g_err, "Failed to launch enqueue bias");
        g_err = clEnqueueWriteBuffer(g_queue_in, g_npl_dev, CL_FALSE, 0, n_bytes_npl, g_n_p_l_buff, 1, &bias_event, NULL);
        checkError(g_err, "Failed to launch enqueue npl");

        clReleaseEvent(params_ev);
        clReleaseEvent(bias_event);
    }

    //^ HANDLER + IMPLEMENDATA_TYPEACIÓN (REVISAR MOVE OP)
    void net_fpga::init_gradient(const net::net_sets &sets)
    {
        // if (!gradient_init)
        // {
        //     size_t ins_num = sets.set_ins[0].size(); //* para guardar el tamaño de entradas, ya que el vector se hace 0 al moverlo
        //     acum_pos = sets.set_ins.size();          //* acum_pos=n of sets
        //     containers.reserve(acum_pos + 1);        //* para incluir al contenedor de acumulación
        //     fx_activations.reserve(n_layers);
        //     tmp_gradient.reserve(n_layers);

        //     for (size_t i = 0; i < n_layers; i++)
        //     {
        //         fx_activations.emplace_back(n_p_l[i], CERO);
        //         tmp_gradient.emplace_back(n_p_l[i], CERO);
        //     }

        //     for (size_t i = 0; i < acum_pos; i++)
        //         containers.emplace_back(n_p_l, sets.set_ins[i], sets.set_outs[i]);

        //     containers.emplace_back(n_p_l, ins_num); //* contenedor de acumulación
        //     gradient_init = true;
        // }
        // else
        //     // cout << "gradient already init!\n";
    }

    //^ HANDLER + IMPLEMENDATA_TYPEACIÓN (REVISAR MOVE OP)
    vector<DATA_TYPE> net_fpga::launch_gradient(size_t iterations, DATA_TYPE error_threshold, DATA_TYPE multiplier) //* returns it times errors
    {
        //         if (gradient_init)
        //         {
        // #ifdef PERFORMANCE
        //             auto start = high_resolution_clock::now();
        // #endif
        //             vector<DATA_TYPE> set_errors(iterations, 0);
        //             my_vec set_single_errors(acum_pos, CERO);

        //             for (size_t i = 0; i < iterations; i++)
        //             {
        //                 for (size_t j = 0; j < acum_pos; j++)
        //                 {
        //                     set_single_errors[j] = gradient(containers[j]).elems_abs().reduce();
        //                     containers[acum_pos] += containers[j];
        //                 }

        //                 containers[acum_pos].normalize_1();
        //                 gradient_update_params(containers[acum_pos]);
        //                 containers[acum_pos].reset();
        //                 set_errors[i] = set_single_errors.reduce();
        //             }
        // #ifdef PERFORMANCE
        //             auto end = high_resolution_clock::now();
        //             auto duration = duration_cast<microseconds>(end - start);
        //             gradient_performance = duration.count();
        // #endif
        //             return set_errors;
        //         }
        //         else
        //         {
        //             // cout << "initialize gradient!\n";
        return vector<DATA_TYPE>(iterations, 0);
        // }
    }

    void net_fpga::print_inner_vals()
    {
        // cout << "Valores internos\n\n";

        // for (auto &i : inner_vals)
        // {
        //     i.print();
        //     // cout << "\n";
        // }
    }

    int64_t net_fpga::get_gradient_performance()
    {
#ifdef PERFORMANCE
        return gradient_performance;
#else
        // cout << "performance not enabled\n";
        return 0;
#endif
    }

    int64_t net_fpga::get_forward_performance()
    {
#ifdef PERFORMANCE
        return forward_performance;
#else
        // cout << "performance not enabled\n";
        return 0;
#endif
    }

    net_fpga::~net_fpga()
    {
        net_fpga_counter--;

        if (net_fpga_counter == 0)
        {
            cleanup();
            // if (kernel)
            //     clReleaseKernel(kernel);
            // if (program)
            //     clReleaseProgram(program);
            // if (queue)
            //     clReleaseCommandQueue(queue);
            // if (context)
            //     clReleaseContext(context);

            program_init = false;
            forward_kernel_init = false;
        }

        delete[] n_p_l;
        delete[] params;
        delete[] bias;
    }
}

void cleanup()
{
    if (fpga::g_kernel_in)
        clReleaseKernel(fpga::g_kernel_in);
    if (fpga::g_kernel_out)
        clReleaseKernel(fpga::g_kernel_out);
    if (fpga::g_program)
        clReleaseProgram(fpga::g_program);
    if (fpga::g_queue_in)
        clReleaseCommandQueue(fpga::g_queue_in);
    if (fpga::g_queue_out)
        clReleaseCommandQueue(fpga::g_queue_out);
    if (fpga::g_context)
        clReleaseContext(fpga::g_context);
    clReleaseEvent(fpga::g_init_event);
    clReleaseEvent(fpga::g_finish_event);

    for (int i = 0; i<BATCH_SIZE; i++)
    {
        clReleaseEvent(fpga::g_im_init_event[i]);
        clReleaseEvent(fpga::g_im_finish_event[i]);
        clReleaseEvent(fpga::g_im_read_event[i]);
    }

    if (fpga::g_are_images_init)
    {
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            delete[] fpga::g_in_images[i];
            delete[] fpga::g_out_images[i];
            fpga::g_in_images[i] = NULL;
            fpga::g_out_images[i] = NULL;
        }
        // delete[] fpga::g_in_images;
        // delete[] fpga::g_out_images;
        fpga::g_are_images_init = false;
    }
}
