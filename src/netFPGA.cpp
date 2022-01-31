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

namespace fpga
{
    using namespace aocl_utils;
    using namespace std;
    using namespace chrono;

    //OpenCL & FPGA variables
    int net_fpga::net_fpga_counter = 0;
    bool net_fpga::program_init = false;
    bool net_fpga::forward_kernel_init = false;
    bool net_fpga::reload_params = true;

    // OpenCL runtime configuration
    cl_platform_id net_fpga::platform = NULL;
    cl_device_id net_fpga::device = NULL;
    cl_context net_fpga::context = NULL;
    cl_command_queue net_fpga::queue_in = NULL;
    cl_command_queue net_fpga::queue_out = NULL;
    // cl_command_queue net_fpga::queue = NULL;
    cl_kernel net_fpga::kernel_in = NULL;
    cl_kernel net_fpga::kernel_out = NULL;
    // cl_kernel net_fpga::kernel = NULL;
    cl_program net_fpga::program = NULL;
    cl_int net_fpga::err = 0;

    cl_mem net_fpga::in_image = NULL;
    cl_mem net_fpga::out_image_borders = NULL;
    // cl_mem net_fpga::inputs_dev = NULL;
    // cl_mem net_fpga::params_dev = NULL;
    cl_mem net_fpga::bias_dev = NULL;
    cl_mem net_fpga::outs_dev = NULL;
    cl_mem net_fpga::npl_dev = NULL;

    int net_fpga::n_ins_buff = 0;
    int net_fpga::n_layers_buff = 0;
    int *net_fpga::n_p_l_buff = 0;

    DATA_TYPE *net_fpga::params_buff = NULL;
    DATA_TYPE *net_fpga::bias_buff = NULL;
    DATA_TYPE *net_fpga::inputs_buff = NULL;
    DATA_TYPE *net_fpga::oputputs_buff = NULL;

    cl_event net_fpga::init_event = NULL;
    cl_event net_fpga::finish_event = NULL;

    cl_event net_fpga::im_init_event[BATCH_SIZE] = {nullptr};
    cl_event net_fpga::im_finish_event[BATCH_SIZE] = {nullptr};
    cl_event net_fpga::im_read_event[BATCH_SIZE] = {nullptr};

    bool net_fpga::are_images_init = false;
    // unsigned char *net_fpga::in_images[BATCH_SIZE] = {nullptr};
    // unsigned char *net_fpga::out_images[BATCH_SIZE] = {nullptr};

    int net_fpga::wr_batch_cnt = 0;
    int net_fpga::rd_batch_cnt = 0;
    int net_fpga::free_batch = BATCH_SIZE;

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
        //cout << "FPGA NET: CREATED\n";
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
        //cout << "FPGA NET: FORWARD\n";
//         if (program_init == false)
//         {
//             net_fpga::_init_program();
//             program_init = true;
//             //cout << "FPGA NET: PROGRAM CREATED\n";
//         }
//         if (forward_kernel_init == false)
//         {
//             net_fpga::_init_kernel("network_v1");
//             forward_kernel_init = true;
//             //cout << "FPGA NET: KERNEL CREATED\n";
//         }
//         if (n_ins_buff != n_ins || n_layers_buff != n_layers || n_p_l_buff != n_p_l || params_buff != params)
//         {
//             net_fpga::_load_params();
//             delete[] inputs_buff;
//             inputs_buff = new DATA_TYPE[n_ins];
//             //cout << "FPGA NET: PARAMS LOADED\n";
//         }

// #ifdef PERFORMANCE
//         auto start = high_resolution_clock::now();
// #endif

//         for (int i = 0; i < n_ins; i++)
//             inputs_buff[i] = inputs[i];

//         DATA_TYPE outs[n_p_l[n_layers - 1]];

//         // cl_event aux;

//         err = clEnqueueWriteBuffer(queue, inputs_dev, CL_FALSE, 0, n_ins * sizeof(DATA_TYPE), inputs_buff, 1, &finish_event, &init_event);
//         checkError(err, "Failed to enqueue inputs");
//         err = clEnqueueTask(queue, kernel, 1, &init_event, &finish_event);
//         checkError(err, "Failed to enqueue task");
//         err = clEnqueueReadBuffer(queue, outs_dev, CL_TRUE, 0, n_p_l[n_layers - 1] * sizeof(DATA_TYPE), outs, 1, &finish_event, NULL);
//         checkError(err, "Failed to enqueue read outs");

// #ifdef PERFORMANCE
//         auto end = high_resolution_clock::now();
//         auto duration = duration_cast<microseconds>(end - start);
//         forward_performance = duration.count();
// #endif
        std::vector<DATA_TYPE> vec_out(n_p_l[n_layers - 1]);
        // for (int i = 0; i < n_p_l[n_layers - 1]; i++)
        //     vec_out[i] = outs[i];

        return vec_out;
    }

    void net_fpga::filter_image(const net::image_set &set)
    {
        //cout << "FPGA NET: FORWARD\n";
        if (program_init == false)
        {
            net_fpga::_init_program(IMAGE_KERNEL);
            program_init = true;
            //cout << "FPGA NET: PROGRAM CREATED\n";
        }
        if (forward_kernel_init == false)
        {
            net_fpga::_init_kernel("image_process", set);
            forward_kernel_init = true;
            //cout << "FPGA NET: KERNEL CREATED\n";
        }

        if (free_batch > 0)
        {
            free_batch--;            

            size_t mapped_size;
            in_images[wr_batch_cnt] = (cl_uchar4 *)clMapHostPipeIntelFPGA(in_image,NULL,sizeof(cl_uchar4)*IMAGE_HEIGHT*IMAGE_WIDTH/4,&mapped_size,&err);
            checkError(err, "Failed to map in pipe");
            out_images[rd_batch_cnt] = (cl_uchar *)clMapHostPipeIntelFPGA(out_image_borders,NULL,sizeof(cl_uchar)*IMAGE_HEIGHT*IMAGE_WIDTH/16,&mapped_size,&err);
            checkError(err, "Failed to map in pipe");
            // for (int i = 0; i < set.original_h * set.original_w; i++)
            //     in_images[wr_batch_cnt][i] = set.resized_image_data[i];
            for (int i = 0; i < (set.original_h * set.original_w)/4; i++)
            {
                cl_uchar4 aux;
                int index = set.original_w*(i%set.original_h) + 4*floor(i/set.original_h);
                aux.x = set.resized_image_data[index];
                aux.y = set.resized_image_data[index+1];
                aux.z = set.resized_image_data[index+2];
                aux.w = set.resized_image_data[index+3];
                *(in_images[wr_batch_cnt] + i) = aux;
            }

            size_t unmapped_size;
            err = clUnmapHostPipeIntelFPGA(in_image, (void*)(in_images[wr_batch_cnt]), mapped_size, &unmapped_size);
            checkError(err, "Failed to unmap in pipe");

            // unsigned char a = in_images[wr_batch_cnt][0];
            // cout << "Executing image kernel\n";
            // unsigned char *test_buff = new unsigned char[set.original_h * set.original_w];
            // in_images[wr_batch_cnt]
            // err = clEnqueueWriteBuffer(queue, inputs_dev, CL_TRUE, 0, set.original_h * set.original_w * sizeof(unsigned char), in_images[wr_batch_cnt], 1, &(im_finish_event[wr_batch_cnt]), &(im_init_event[wr_batch_cnt]));
            // checkError(err, "Failed to enqueue inputs");
            // delete[] test_buff;
            int next_wr_batch = wr_batch_cnt == (BATCH_SIZE - 1) ? 0 : wr_batch_cnt + 1;

            // err = clEnqueueTask(queue, kernel, 1, &(im_init_event[wr_batch_cnt]), &(im_finish_event[next_wr_batch]));
            // checkError(err, "Failed to enqueue task");
            // err = clEnqueueReadBuffer(queue, outs_dev, CL_FALSE, 0, set.original_h * set.original_w * sizeof(unsigned char), out_images[wr_batch_cnt], 1, &(im_finish_event[next_wr_batch]), &(im_read_event[wr_batch_cnt]));
            // checkError(err, "Failed to enqueue read outs");

            wr_batch_cnt = next_wr_batch;
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

        if (free_batch < BATCH_SIZE)
        {
            //cout << "Leyendo datos\n";
            //cout << "Freebatch " << free_batch << "\n";
            free_batch++;

            // clWaitForEvents(1, &(im_read_event[rd_batch_cnt]));
            // out_image.resized_image_data.reserve(IMAGE_HEIGHT * IMAGE_WIDTH);

            // for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++)
            //     out_image.resized_image_data.emplace_back(out_images[rd_batch_cnt][i]);

            for (int i = 0; i < (IMAGE_HEIGHT * IMAGE_WIDTH)/16; i++)
            {
                for(int j = 0; j<4; j++)
                {
                    int index = IMAGE_WIDTH*((4*i+j)%IMAGE_HEIGHT) + 4*floor(i/IMAGE_HEIGHT);
                    out_image.resized_image_data[index] = *(out_images[rd_batch_cnt] + i);
                    out_image.resized_image_data[index+1] = *(out_images[rd_batch_cnt] + i);
                    out_image.resized_image_data[index+2] = *(out_images[rd_batch_cnt] + i);
                    out_image.resized_image_data[index+3] = *(out_images[rd_batch_cnt] + i);
                }
            }
            
            size_t unmapped_size;
            err = clUnmapHostPipeIntelFPGA(in_image, (void*)(in_images[wr_batch_cnt]), sizeof(cl_uchar)*IMAGE_HEIGHT*IMAGE_WIDTH/16, &unmapped_size);
            checkError(err, "Failed to unmap in pipe");

            rd_batch_cnt = rd_batch_cnt == (BATCH_SIZE - 1) ? 0 : rd_batch_cnt + 1;
            //cout << "Datos leidos\n";
        }
        else
        {
            cout << "PILA VACIA\n";
        }

        //cout << out_image.resized_image_data.size() << "\n";
        return out_image;
    }

    void net_fpga::_init_program(int prg)
    {
        clMapHostPipeIntelFPGA = (void * (*) (cl_mem, cl_map_flags, size_t, size_t *, cl_int *)) clGetExtensionFunctionAddress("clMapHostPipeIntelFPGA");
        clUnmapHostPipeIntelFPGA = (cl_int (*) (cl_mem, void *, size_t, size_t *)) clGetExtensionFunctionAddress("clUnmapHostPipeIntelFPGA");
        out_image.resized_image_data.reserve(IMAGE_HEIGHT * IMAGE_WIDTH);
        for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++)
                out_image.resized_image_data.emplace_back(0);

        //Platform, device and context

        //cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, NULL);
        checkError(err, "Failed to get platforms");

        //cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);
        checkError(err, "Failed to find device");

        //cl_context context;
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        checkError(err, "Failed to create context");

        //cl_command_queue queue;
        queue_in = clCreateCommandQueue(context, device, 0, &err);
        checkError(err, "Failed to create queue");
        queue_out = clCreateCommandQueue(context, device, 0, &err);
        checkError(err, "Failed to create queue");
        // queue = clCreateCommandQueue(context, device, 0, &err);
        // checkError(err, "Failed to create queue");

        //cl_program
        const char *prg_name = (prg == NET_KERNEL ? "vector_kernels" : "image_kernels_pipes");
        std::string binary_file = getBoardBinaryFile(prg_name, device); //Coge el aocx
        program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

        err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
        checkError(err, "Failed to create program");

        init_event = clCreateUserEvent(context, NULL);
        finish_event = clCreateUserEvent(context, NULL);

        clSetUserEventStatus(init_event, CL_COMPLETE);
        clSetUserEventStatus(finish_event, CL_COMPLETE);
    }

    void net_fpga::_init_kernel(const char *kernel_name)
    {
        // int n_bytes_npl = n_layers * sizeof(int);
        // int n_bytes_inputs = n_ins * sizeof(DATA_TYPE);
        // int n_bytes_params = n_params * sizeof(DATA_TYPE);
        // int n_bytes_bias = n_neurons * sizeof(DATA_TYPE);
        // int n_bytes_outs = n_p_l[n_layers - 1] * sizeof(DATA_TYPE);

        // //cout << "   Creating buffers:\n";
        // inputs_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bytes_inputs, NULL, &err); //CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        // checkError(err, "Failed to create buffer inputs");
        // params_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bytes_params, NULL, &err);
        // checkError(err, "Failed to create buffer params");
        // bias_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bytes_bias, NULL, &err);
        // checkError(err, "Failed to create buffer bias");
        // outs_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_bytes_outs, NULL, &err);
        // checkError(err, "Failed to create buffer outputs");
        // npl_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bytes_npl, NULL, &err);
        // checkError(err, "Failed to create buffer npl");

        // kernel = clCreateKernel(program, kernel_name, &err);
        // checkError(err, "Failed to create kernel");

        // //cout << "   Setting Args:\n";
        // err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputs_dev);
        // checkError(err, "Failed to set arg inputs");
        // err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_dev);
        // checkError(err, "Failed to set arg params");
        // err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bias_dev);
        // checkError(err, "Failed to set arg bias");
        // err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&outs_dev);
        // checkError(err, "Failed to set arg outputs");
        // err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&npl_dev);
        // checkError(err, "Failed to set arg npl");
    }

    void net_fpga::_init_kernel(const char *kernel_name, const net::image_set &set)
    {
        // if (!are_images_init)
        // {
        //     for (int i = 0; i < BATCH_SIZE; i++)
        //     {
        //         in_images[i] = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT]();
        //         out_images[i] = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT]();
        //     }
        //     are_images_init = true;
        // }
        // kernel = clCreateKernel(program, "my_kernel", &err);
        // int n_bytes_in_image = set.original_h * set.original_w * sizeof(unsigned char);
        // int n_bytes_out_image = set.original_h * set.original_w * sizeof(unsigned char);

        //cout << "   Creating buffers:\n";
        in_image = clCreatePipe(context, CL_MEM_HOST_WRITE_ONLY, sizeof(cl_uchar4), IMAGE_HEIGHT, NULL, &err);
        checkError(err, "Failed to create in pipe");
        out_image_borders = clCreatePipe(context, CL_MEM_HOST_READ_ONLY, sizeof(cl_uchar), 8, NULL, &err);
        checkError(err, "Failed to create out pipe");
        // inputs_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bytes_in_image, NULL, &err); //CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
        // checkError(err, "Failed to create buffer inputs");
        // outs_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_bytes_out_image, NULL, &err);
        // checkError(err, "Failed to create buffer outputs");

        kernel_in = clCreateKernel(program, "in_image_dws4", &err);
        checkError(err, "Failed to create kernel 1");
        kernel_out = clCreateKernel(program, "app_borders", &err);
        checkError(err, "Failed to create kernel 2");
        // kernel = clCreateKernel(program, kernel_name, &err);
        // checkError(err, "Failed to create kernel");

        //cout << "   Setting Args:\n";
        err = clSetKernelArg(kernel_in, 0, sizeof(cl_mem), (void *)&in_image);
        cout << "Error code" << err << " \n";
        checkError(err, "Failed to set inputs");
        err = clSetKernelArg(kernel_out, 0, sizeof(cl_mem), (void *)&out_image_borders);
        checkError(err, "Failed to set outputs");
        err = clEnqueueTask(queue_in, kernel_in, 0, NULL, NULL);
        checkError(err, "Failed to enqueue in task");
        err = clEnqueueTask(queue_out, kernel_out, 0, NULL, NULL);
        checkError(err, "Failed to enqueue out task");
        // err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputs_dev);
        // checkError(err, "Failed to set arg inputs");
        // err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outs_dev);
        // checkError(err, "Failed to set arg outputs");

        // im_init_event[0] = clCreateUserEvent(context, NULL);
        im_finish_event[0] = clCreateUserEvent(context, NULL);

        // clSetUserEventStatus(im_init_event[0], CL_COMPLETE);
        clSetUserEventStatus(im_finish_event[0], CL_COMPLETE);
    }

    void net_fpga::_load_params()
    {
        // n_ins_buff = n_ins;
        // n_layers_buff = n_layers;
        // n_p_l_buff = n_p_l;

        // params_buff = params;
        // bias_buff = bias;

        // int n_bytes_npl = n_layers * sizeof(int);
        // int n_bytes_inputs = n_ins * sizeof(DATA_TYPE);
        // int n_bytes_params = n_params * sizeof(DATA_TYPE);
        // int n_bytes_bias = n_neurons * sizeof(DATA_TYPE);
        // int n_bytes_outs = n_p_l[n_layers - 1] * sizeof(DATA_TYPE);

        // err = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&n_layers_buff);
        // checkError(err, "Failed to set arg n_layers");
        // err = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&n_ins_buff);
        // checkError(err, "Failed to set arg n_ins");

        // cl_event params_ev, bias_event;

        // err = clEnqueueWriteBuffer(queue, params_dev, CL_FALSE, 0, n_bytes_params, params_buff, 0, NULL, &params_ev);
        // checkError(err, "Failed to launch enqueue params");
        // err = clEnqueueWriteBuffer(queue, bias_dev, CL_FALSE, 0, n_bytes_bias, bias_buff, 1, &params_ev, &bias_event);
        // checkError(err, "Failed to launch enqueue bias");
        // err = clEnqueueWriteBuffer(queue, npl_dev, CL_FALSE, 0, n_bytes_npl, n_p_l_buff, 1, &bias_event, NULL);
        // checkError(err, "Failed to launch enqueue npl");

        // clReleaseEvent(params_ev);
        // clReleaseEvent(bias_event);
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
        //     //cout << "gradient already init!\n";
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
        //             //cout << "initialize gradient!\n";
        return vector<DATA_TYPE>(iterations, 0);
        // }
    }

    void net_fpga::print_inner_vals()
    {
        // //cout << "Valores internos\n\n";

        // for (auto &i : inner_vals)
        // {
        //     i.print();
        //     //cout << "\n";
        // }
    }

    int64_t net_fpga::get_gradient_performance()
    {
#ifdef PERFORMANCE
        return gradient_performance;
#else
        //cout << "performance not enabled\n";
        return 0;
#endif
    }

    int64_t net_fpga::get_forward_performance()
    {
#ifdef PERFORMANCE
        return forward_performance;
#else
        //cout << "performance not enabled\n";
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
    if (fpga::net_fpga::kernel_in)
        clReleaseKernel(fpga::net_fpga::kernel_in);
    if (fpga::net_fpga::kernel_out)
        clReleaseKernel(fpga::net_fpga::kernel_out);
    // if (fpga::net_fpga::kernel)
    //     clReleaseKernel(fpga::net_fpga::kernel);
    if (fpga::net_fpga::program)
        clReleaseProgram(fpga::net_fpga::program);
    if (fpga::net_fpga::queue_in)
        clReleaseCommandQueue(fpga::net_fpga::queue_in);
    if (fpga::net_fpga::queue_out)
        clReleaseCommandQueue(fpga::net_fpga::queue_out);
    // if (fpga::net_fpga::queue)
    //     clReleaseCommandQueue(fpga::net_fpga::queue);
    if (fpga::net_fpga::context)
        clReleaseContext(fpga::net_fpga::context);
    clReleaseEvent(fpga::net_fpga::init_event);
    clReleaseEvent(fpga::net_fpga::finish_event);
}
