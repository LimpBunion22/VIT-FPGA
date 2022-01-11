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
    cl_command_queue net_fpga::queue = NULL;
    cl_kernel net_fpga::kernel = NULL;
    cl_program net_fpga::program = NULL;
    cl_int net_fpga::err = NULL;

    cl_mem net_fpga::inputs_dev = NULL;
    cl_mem net_fpga::params_dev = NULL;
    cl_mem net_fpga::bias_dev = NULL;
    cl_mem net_fpga::outs_dev = NULL;
    cl_mem net_fpga::npl_dev = NULL;

    int net_fpga::n_ins_buff = NULL;
    int net_fpga::n_layers_buff = NULL;
    int *net_fpga::n_p_l_buff = NULL;

    DATA_TYPE *net_fpga::params_buff = NULL;
    DATA_TYPE *net_fpga::bias_buff = NULL;
    DATA_TYPE *net_fpga::inputs_buff = NULL;
    DATA_TYPE *net_fpga::oputputs_buff = NULL;

    cl_event net_fpga::init_event = NULL;
    cl_event net_fpga::finish_event = NULL;

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

        if (program_init == false)
        {
            net_fpga::_init_program();
            program_init = true;
        }
        if (forward_kernel_init == false)
        {
            net_fpga::_init_kernel("network_v1");
            forward_kernel_init = true;
        }
        if (n_ins_buff != n_ins || n_layers_buff != n_layers || n_p_l_buff != n_p_l || params_buff != params)
        {
            net_fpga::_load_params();
        }

#ifdef PERFORMANCE
        auto start = high_resolution_clock::now();
#endif

        for (int i = 0; i < n_ins; i++)
            inputs_buff[i] = inputs[i];

        DATA_TYPE *outs;

        err = clEnqueueWriteBuffer(queue, inputs_dev, CL_FALSE, 0, n_ins * sizeof(DATA_TYPE), inputs_buff, 1, &finish_event, &init_event);
        err = clEnqueueTask(queue, kernel, 1, &init_event, &finish_event);
        err = clEnqueueReadBuffer(queue, outs_dev, CL_TRUE, 0, n_p_l[n_layers - 1] * sizeof(DATA_TYPE), outs, 1, &finish_event, NULL);

#ifdef PERFORMANCE
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        forward_performance = duration.count();
#endif
        std::vector<DATA_TYPE> vec_out(n_p_l[n_layers - 1]);
        for (int i = 0; i < n_p_l[n_layers - 1]; i++)
            vec_out[i] = outs[i];

        return vec_out;
    }

    void net_fpga::_init_program()
    {
        //Platform, device and context

        //cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, NULL);

        //cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);

        //cl_context context;
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

        //cl_command_queue queue;
        queue = clCreateCommandQueue(context, device, 0, &err);

        //cl_program
        std::string binary_file = getBoardBinaryFile("vector_kernels", device); //Coge el aocx
        program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

        clSetUserEventStatus(init_event, CL_COMPLETE);
        clSetUserEventStatus(finish_event, CL_COMPLETE);
    }

    void net_fpga::_init_kernel(const char *kernel_name)
    {
        // kernel = clCreateKernel(program, "my_kernel", &err);
        kernel = clCreateKernel(program, kernel_name, &err);

        int n_bytes_npl = n_layers * sizeof(int);
        int n_bytes_inputs = n_ins * sizeof(DATA_TYPE);
        int n_bytes_params = n_params * sizeof(DATA_TYPE);
        int n_bytes_bias = n_neurons * sizeof(DATA_TYPE);
        int n_bytes_outs = n_p_l[n_layers - 1] * sizeof(DATA_TYPE);

        inputs_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, n_bytes_inputs, NULL, &err);
        params_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, n_bytes_params, NULL, &err);
        bias_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, n_bytes_bias, NULL, &err);
        outs_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, n_bytes_outs, NULL, &err);
        npl_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, n_bytes_npl, NULL, &err);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputs_dev);
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_dev);
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bias_dev);
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&outs_dev);
        err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&npl_dev);
        err = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&n_layers_buff);
        err = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&n_ins_buff);
    }

    void net_fpga::_load_params()
    {
        n_ins_buff = n_ins;
        n_layers_buff = n_layers;
        n_p_l_buff = n_p_l;

        params_buff = params;
        bias_buff = bias;

        int n_bytes_npl = n_layers * sizeof(int);
        int n_bytes_inputs = n_ins * sizeof(DATA_TYPE);
        int n_bytes_params = n_params * sizeof(DATA_TYPE);
        int n_bytes_bias = n_neurons * sizeof(DATA_TYPE);
        int n_bytes_outs = n_p_l[n_layers - 1] * sizeof(DATA_TYPE);

        err = clEnqueueWriteBuffer(queue, params_dev, CL_FALSE, 0, n_bytes_params, params_buff, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, bias_dev, CL_FALSE, 0, n_bytes_bias, bias_buff, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, npl_dev, CL_FALSE, 0, n_bytes_npl, n_p_l_buff, 0, NULL, NULL);
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
        //     cout << "gradient already init!\n";
    }

    //^ HANDLER + IMPLEMENDATA_TYPEACIÓN (REVISAR MOVE OP)
    vector<DATA_TYPE> net_fpga::launch_gradient(size_t iterations) //* returns it times errors
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
        //             cout << "initialize gradient!\n";
        return vector<DATA_TYPE>(iterations, 0);
        // }
    }

    void net_fpga::print_inner_vals()
    {
        // cout << "Valores internos\n\n";

        // for (auto &i : inner_vals)
        // {
        //     i.print();
        //     cout << "\n";
        // }
    }

    int64_t net_fpga::get_gradient_performance()
    {
#ifdef PERFORMANCE
        return gradient_performance;
#else
        cout << "performance not enabled\n";
        return 0;
#endif
    }

    int64_t net_fpga::get_forward_performance()
    {
#ifdef PERFORMANCE
        return forward_performance;
#else
        cout << "performance not enabled\n";
        return 0;
#endif
    }


    net_fpga::~net_fpga()
    {
        net_fpga_counter--;

        if(net_fpga_counter == 0)
        {
            if (kernel)
                clReleaseKernel(kernel);
            if (program)
                clReleaseProgram(program);
            if (queue)
                clReleaseCommandQueue(queue);
            if (context)
                clReleaseContext(context);

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

}
