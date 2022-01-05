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

    net_fpga::net_fpga(const net::net_data &data, bool derivate, bool random)
        : n_layers(data.n_p_l.size()), n_sets(0), gradient_init(false), gradient_performance(0), forward_performance(0), n_ins(data.n_ins)

    {       
        n_p_l = new int[n_layers];
        n_neurons = 0;
        n_params = 0;

        for(int i = 0; i<n_layers; i++){
            n_p_l[i] = data.n_p_l[i];
            n_neurons += n_p_l[i];
            if(i==0)
                n_params += n_p_l[i]*n_ins;
            else
                n_params += n_p_l[i]*n_p_l[i-1];
        }

        params = new float[n_params];
        activations = 1; // 1 -> RELU2
        bias = new float[n_neurons];

        if (random){
            for (int i = 0; i < n_params; i++)
                params[i] = float(rand()%200 -100)/100;
            for (int i = 0; i < n_neurons; i++)
                bias[i] = float(rand()%200 -100)/100;
        }
        else
        {
            int param_cnt = 0;
            int neuron_cnt = 0;

            for (int i = 0; i < n_layers; i++)
            {
                for(int j = 0; j < n_p_l[i]; j++)
                {
                    for(int k = 0; k < data.params[i][j].size(); k++)
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
            bool del = (n_ins != rh.n_ins) || (n_layers != rh.n_layers);               
            for(int i=0; i<n_layers && !del; i++)
                del = n_p_l[i]!=rh.n_p_l[i];

            if(del)
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

                for(int i=0; i<n_params; i++)
                   params[i] = rh.params[i];

                for(int i=0; i<n_neurons; i++)
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
        data.n_p_l = n_p_l;

        for (size_t i = 0; i < n_layers; i++)
        {
            data.params.emplace_back(params[i].rows(), vector<DATA_TYPE>(params[i].cols(), 0));

            for (size_t j = 0; j < params[i].rows(); j++)
                for (size_t k = 0; k < params[i].cols(); k++)
                    data.params[i][j][k] = params[i](j, k);

            data.bias.emplace_back(bias[i].size(), 0);

            for (size_t j = 0; j < bias[i].size(); j++)
                data.bias[i][j] = bias[i][j];
        }

        return data;
    }

    vector<DATA_TYPE> net_fpga::launch_forward(const vector<DATA_TYPE> &inputs) //* returns result
    {
#ifdef PERFORMANCE
        auto start = high_resolution_clock::now();
#endif
        vector<DATA_TYPE> inputs_copy = inputs;
        my_vec ins(inputs_copy);

        for (size_t i = 0; i < n_layers; i++)
        {
            if (i == 0)
            {
                my_vec x = params[i] * ins;
                x += bias[i];
                inner_vals[i] = activations[i].calculate(x);
            }
            else
            {
                my_vec x = params[i] * inner_vals[i - 1];
                x += bias[i];
                inner_vals[i] = activations[i].calculate(x);
            }
        }
#ifdef PERFORMANCE
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        forward_performance = duration.count();
#endif

        return inner_vals.back().copy_inner_vec();
    }

    //^ HANDLER + IMPLEMENDATA_TYPEACIÓN (REVISAR MOVE OP)
    void net_fpga::init_gradient(const net::net_sets &sets)
    {
        if (!gradient_init)
        {
            size_t ins_num = sets.set_ins[0].size(); //* para guardar el tamaño de entradas, ya que el vector se hace 0 al moverlo
            acum_pos = sets.set_ins.size();          //* acum_pos=n of sets
            containers.reserve(acum_pos + 1);        //* para incluir al contenedor de acumulación
            fx_activations.reserve(n_layers);
            tmp_gradient.reserve(n_layers);

            for (size_t i = 0; i < n_layers; i++)
            {
                fx_activations.emplace_back(n_p_l[i], CERO);
                tmp_gradient.emplace_back(n_p_l[i], CERO);
            }

            for (size_t i = 0; i < acum_pos; i++)
                containers.emplace_back(n_p_l, sets.set_ins[i], sets.set_outs[i]);

            containers.emplace_back(n_p_l, ins_num); //* contenedor de acumulación
            gradient_init = true;
        }
        else
            cout << "gradient already init!\n";
    }

    //^ HANDLER + IMPLEMENDATA_TYPEACIÓN (REVISAR MOVE OP)
    vector<DATA_TYPE> net_fpga::launch_gradient(size_t iterations) //* returns it times errors
    {
        if (gradient_init)
        {
#ifdef PERFORMANCE
            auto start = high_resolution_clock::now();
#endif
            vector<DATA_TYPE> set_errors(iterations, 0);
            my_vec set_single_errors(acum_pos, CERO);

            for (size_t i = 0; i < iterations; i++)
            {
                for (size_t j = 0; j < acum_pos; j++)
                {
                    set_single_errors[j] = gradient(containers[j]).elems_abs().reduce();
                    containers[acum_pos] += containers[j];
                }

                containers[acum_pos].normalize_1();
                gradient_update_params(containers[acum_pos]);
                containers[acum_pos].reset();
                set_errors[i] = set_single_errors.reduce();
            }
#ifdef PERFORMANCE
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            gradient_performance = duration.count();
#endif
            return set_errors;
        }
        else
        {
            cout << "initialize gradient!\n";
            return vector<DATA_TYPE>(iterations, 0);
        }
    }

    void net_fpga::print_inner_vals()
    {
        cout << "Valores internos\n\n";

        for (auto &i : inner_vals)
        {
            i.print();
            cout << "\n";
        }
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
}