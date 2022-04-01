#include <netFPGA.h>
#include <math.h>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <ctime>

namespace fpga
{
    using namespace std;
    using namespace chrono;

    net_fpga::net_fpga(): n_p_l(nullptr),params(nullptr),bias(nullptr)
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        ostringstream oss;
        oss << put_time(&tm, "%d-%m-%Y %H-%M-%S");
        net_ident = oss.str();
    }

    net_fpga::net_fpga(const net::net_data &data, bool random)
        : n_layers(data.n_p_l.size()), n_sets(0), gradient_init(false), gradient_performance(0), forward_performance(0)

    {

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        ostringstream oss;
        oss << put_time(&tm, "%d-%m-%Y %H-%M-%S");
        net_ident = oss.str();
        
        n_ins = data.n_ins%N_INS==0 ? data.n_ins : data.n_ins + (N_INS-data.n_ins%N_INS);
        n_outs = data.n_p_l[n_layers-1];
        n_p_l = new int[n_layers];
        n_neurons = 0;
        n_params = 0;

        for (int i = 0; i < n_layers; i++){
            n_p_l[i] = data.n_p_l[i]%N_NEURONS==0 ? data.n_p_l[i] : data.n_p_l[i] + (N_NEURONS-data.n_p_l[i]%N_NEURONS);
            n_neurons += n_p_l[i];
            if (i == 0)
                n_params += n_p_l[i] * n_ins;
            else
                n_params += n_p_l[i] * n_p_l[i - 1];
        }

        params = new int[n_params];
        activations = 1; // 1 -> RELU2
        bias = new int[n_neurons];

        if (random){
            for (int i = 0; i < n_params; i++)
                params[i] = rand() % 2*DECIMAL_FACTOR - DECIMAL_FACTOR;
            for (int i = 0; i < n_neurons; i++)
                bias[i] = rand() % 2*DECIMAL_FACTOR - DECIMAL_FACTOR;
        }else{

            for (int i = 0; i < n_params; i++)
                params[i] = 0.0f;
            for (int i = 0; i < n_neurons; i++)
                bias[i] = 0.0f;

            int param_cnt = 0;
            int neuron_cnt = 0;

            for (int i = 0; i < n_layers; i++){
                int n_par_l = data.params[i][0].size();
                for (int j = 0; j < n_p_l[i]; j++){
                    for (int k = 0; k < n_par_l; k++){
                        if(j<data.params[i].size())
                            params[param_cnt] = int(DECIMAL_FACTOR*data.params[i][j][k]);
                        param_cnt++;
                    }
                    if(j<data.bias[i].size())
                        bias[neuron_cnt] = int(DECIMAL_FACTOR*data.bias[i][j]);
                    neuron_cnt++;
                }
            }
        }
        // cout << "FPGA NET: CREATED\n";
    }

    net_fpga::net_fpga(net_fpga &&rh) : n_ins(rh.n_ins),
                                        n_outs(rh.n_outs),
                                        n_layers(rh.n_layers),
                                        n_neurons(rh.n_neurons),
                                        n_params(rh.n_params),
                                        activations(rh.activations),
                                        n_sets(rh.n_sets),
                                        gradient_init(rh.gradient_init),
                                        net_ident(rh.net_ident)

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
            net_ident = rh.net_ident;
            n_ins = rh.n_ins;
            n_outs = rh.n_outs;
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
            net_ident = rh.net_ident;
            n_ins = rh.n_ins;
            n_outs = rh.n_outs;
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
            params = new int[n_params];
            bias = new int[n_neurons];

            for (int i = 0; i < n_params; i++)
                params[i] = rh.params[i];

            for (int i = 0; i < n_neurons; i++)
                bias[i] = rh.bias[i];
            
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

        for (int i = 0; i < n_layers; i++){
            data.n_p_l[i] = n_p_l[i];
            int n_params = (i == 0) ? n_ins : n_p_l[i - 1];
            data.params.emplace_back(n_p_l[i], vector<float>(n_params));
            data.bias.emplace_back(n_p_l[i]);

            for (int j = 0; j < n_p_l[i]; j++){
                data.bias[i][j] = (float)bias[neurons_cnt];
                neurons_cnt++;

                for (int k = 0; k < n_ins; k++){
                    data.params[i][j][k] = (float)params[params_cnt];
                    params_cnt++;
                }
            }
        }
        return data;
    }

    vector<float> net_fpga::launch_forward(const vector<float> &inputs) //* returns result
    {
        vector<int> int_inputs(n_ins,0);
        for(int i=0; i<inputs.size(); i++)
            int_inputs[i] = int(inputs[i]*DECIMAL_FACTOR);

#ifdef PERFORMANCE
        auto start = high_resolution_clock::now();
#endif
        identifier = master->enqueue_net(this, int_inputs);
        if(identifier==0){
            cout << "FPGA memory full, unable to allocate "<< net_ident <<"\n";
            vector<float> vec_out(n_outs,0);
            return vec_out;
        }

        master->solve_nets();
        vector<int> int_out = master->read_net(identifier);

#ifdef PERFORMANCE
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        forward_performance = duration.count();
#endif   
        vector<float> vec_out(n_outs);     
        for (int i = 0; i < n_outs; i++)
            vec_out[i] = float(int_out[i])/DECIMAL_FACTOR;

        return vec_out;
    }

    void net_fpga::enqueue_net(const std::vector<float> &inputs)
    {
        vector<int> int_inputs(n_ins,0);
        for(int i=0; i<inputs.size(); i++)
            int_inputs[i] = int(inputs[i]*DECIMAL_FACTOR);

        identifier = master->enqueue_net(this, int_inputs);
        if(identifier==0)
            cout << "FPGA memory full, unable to allocate "<< net_ident <<"\n";
        
    }

    void net_fpga::solve_pack(){
        master->solve_nets();
    }

    std::vector<float> net_fpga::read_net()
    {
        vector<int> int_out = master->read_net(identifier);
        vector<float> vec_out(n_outs);     
        for (int i = 0; i < n_outs; i++)
            vec_out[i] = float(int_out[i])/DECIMAL_FACTOR;

        return vec_out;
    }

    //^ HANDLER + IMPLEMENfloatACIÓN (REVISAR MOVE OP)
    vector<float> net_fpga::launch_gradient(size_t iterations, float error_threshold, float multiplier) //* returns it times errors
    {
        //         if (gradient_init)
        //         {
        // #ifdef PERFORMANCE
        //             auto start = high_resolution_clock::now();
        // #endif
        //             vector<float> set_errors(iterations, 0);
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
        return vector<float>(iterations, 0);
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
        // cout << "Deleting FPGA\n";
        // cout << "Deleting net pointers\n";
        delete[] n_p_l;
        delete[] params;
        delete[] bias;
        // cout << "FPGA deleted\n";
    }
}

