#include <netFPGA.h>
#include "fpgaDefines.h"
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

    net_fpga::net_fpga(fpga_handler &handler) : master(handler)
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        ostringstream oss;
        oss << put_time(&tm, "%d-%m-%Y %H-%M-%S");
        net_ident = oss.str();
    }

    void net_fpga::set_input_size(int input_size){}
    void net_fpga::build_fully_layer(int layer_size, int activation){}
    void net_fpga::build_net(){}

    void net_fpga::build_net_from_file(const net::layout &layout)
    {
        n_layers = layout.n_p_l.size();
        n_sets = 0;
        gradient_init = false;
        gradient_performance = 0;
        forward_performance = 0;

        n_ins = layout.input_size % N_INS == 0 ? layout.input_size : layout.input_size + (N_INS - layout.input_size % N_INS);
        n_outs = layout.n_p_l[n_layers - 1];
        n_p_l.reserve(n_layers);
        n_neurons = 0;
        n_params = 0;

        for (int i = 0; i < n_layers; i++)
        {
            n_p_l.emplace_back(layout.n_p_l[i] % N_NEURONS == 0 ? layout.n_p_l[i] : layout.n_p_l[i] + (N_NEURONS - layout.n_p_l[i] % N_NEURONS));
            n_neurons += n_p_l[i];
            if (i == 0)
                n_params += n_p_l[i] * n_ins;
            else
                n_params += n_p_l[i] * n_p_l[i - 1];
        }

        params = vector<FPGA_DATA_TYPE>(n_params,0);
        activations = 1; // 1 -> RELU2
        bias = vector<FPGA_DATA_TYPE>(n_neurons,0);
        int param_cnt = 0;
        int neuron_cnt = 0;
        int total_cnt = 0;
        for (int i = 0; i < n_layers; i++)
        {
            int n_per_n = i == 0 ? n_ins : n_p_l[i - 1];
            int o_n_per_n = i == 0 ? layout.input_size : layout.n_p_l[i - 1];
            for (int j = 0; j < n_p_l[i]; j++)
            {
                for (int k = 0; k < n_per_n; k++)
                {
                    if (j < layout.n_p_l[i] && k < o_n_per_n){
                        params[param_cnt] = (FPGA_DATA_TYPE)(DECIMAL_FACTOR * layout.param_bias[total_cnt]);
                        total_cnt++;
                    }
                    param_cnt++;
                }
                if (j < layout.n_p_l[i]){
                    bias[neuron_cnt] = (FPGA_DATA_TYPE)(DECIMAL_FACTOR * DECIMAL_FACTOR * layout.param_bias[total_cnt]);
                    total_cnt++;
                }
                neuron_cnt++;
            }
        }
        my_data = get_fpga_data();
        // cout << "FPGA NET: CREATED\n";
    }


    void net_fpga::build_net_from_data(int input_size, const std::vector<int> &n_p_l, const std::vector<int> &activations)
    {
        n_layers = n_p_l.size();
        n_ins = input_size % N_INS == 0 ? input_size : input_size + (N_INS - input_size % N_INS);
        n_outs = n_p_l[n_layers - 1];
        this->n_p_l.reserve(n_layers);
        n_neurons = 0;
        n_params = 0;

        for (int i = 0; i < n_layers; i++)
        {
            int nnrs = n_p_l[i] % N_NEURONS == 0 ? n_p_l[i] : n_p_l[i] + (N_NEURONS - n_p_l[i] % N_NEURONS);
            this->n_p_l.emplace_back(nnrs);
            n_neurons += nnrs;
            if (i == 0)
                n_params += nnrs * this->n_ins;
            else
                n_params += nnrs * this->n_p_l[i - 1];
        }

        params = vector<FPGA_DATA_TYPE>(n_params,0);
        this->activations = 1; // 1 -> RELU2
        bias = vector<FPGA_DATA_TYPE>(n_neurons,0);

        int params_cnt = 0;
        int bias_cnt = 0;
        for (int l = 0; l < n_layers; l++)
        {   
            for(int n=0; n<this->n_p_l[l]; n++){
                int npb = l==0 ? this->n_ins : this->n_p_l[l-1];
                if(n<n_p_l[l]){
                    bias[bias_cnt] = rand() % 4 * DECIMAL_FACTOR - DECIMAL_FACTOR;
                    int nps = l==0 ? n_ins : n_p_l[l-1];
                    for(int p=0; p<npb; p++){
                        if(p<nps)
                            params[params_cnt] = rand() % 4 * DECIMAL_FACTOR - DECIMAL_FACTOR;
                        params_cnt++;
                    }
                }else{
                    for(int p=0; p<npb; p++)
                        params_cnt++;
                }
                bias_cnt++;
            }
        }
        my_data = get_fpga_data();

    }

    net_fpga::net_fpga(net_fpga &&rh) : n_ins(rh.n_ins),
                                        n_outs(rh.n_outs),
                                        n_layers(rh.n_layers),
                                        n_neurons(rh.n_neurons),
                                        n_params(rh.n_params),
                                        activations(rh.activations),
                                        n_sets(rh.n_sets),
                                        gradient_init(rh.gradient_init),
                                        net_ident(rh.net_ident),
                                        master(rh.master)

    {
        n_p_l = rh.n_p_l;
        params = rh.params;
        bias = rh.bias;

        my_data = get_fpga_data();
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

            n_p_l = rh.n_p_l;
            params = rh.params;
            bias = rh.bias;

            master = rh.master;

            my_data = get_fpga_data();
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

            n_p_l = rh.n_p_l;
            params = rh.params;
            bias = rh.bias;

            master = rh.master;

            my_data = get_fpga_data();
        }

        return *this;
    }

    net::layout net_fpga::get_net_data()const // TODO:implementar
    {
        net::layout data;
        data.input_size = n_ins;

        data.n_p_l.reserve(n_layers);
        data.param_bias.reserve(n_neurons+n_params);
        data.activation.reserve(n_layers);
        int params_cnt = 0;
        int neurons_cnt = 0;
        int total_cnt = 0;

        for (int i = 0; i < n_layers; i++)
        {
            data.n_p_l[i] = n_p_l[i];
            int l_params = (i == 0) ? n_ins : n_p_l[i - 1];

            for (int j = 0; j < n_p_l[i]; j++)
            {
                for (int k = 0; k < l_params; k++)
                {
                    data.param_bias[total_cnt] = (float)params[params_cnt]/DECIMAL_FACTOR;
                    params_cnt++;
                    total_cnt++;
                }
                data.param_bias[total_cnt] = (float)bias[neurons_cnt]/DECIMAL_FACTOR/DECIMAL_FACTOR;
                neurons_cnt++;
                total_cnt++;
            }
        }
        return data;
    }

    fpga_data net_fpga::get_fpga_data()
    {

        fpga_data my_data;
        my_data.n_ins = n_ins;
        my_data.n_layers = n_layers;
        my_data.n_params = n_params;
        my_data.n_neurons = n_neurons;
        my_data.n_p_l = n_p_l.data();
        my_data.params = params.data();
        my_data.bias = bias.data();

        return my_data;
    }

    vector<float> net_fpga::run_forward(const vector<float> &inputs) //* returns result
    {
#if fpga_performance == 1
        auto start = high_resolution_clock::now();
#endif
        // cout << BLUE << "   NET_FPGA: Launching forward" << RESET << "\n";
        vector<FPGA_DATA_TYPE> int_inputs(n_ins, 0);
        for (int i = 0; i < inputs.size(); i++)
            int_inputs[i] = (FPGA_DATA_TYPE)(inputs[i] * DECIMAL_FACTOR);

        cout3(BLUE, "   NET_FPGA: Enqueuing net", "");
        identifier = master.enqueue_net(my_data, int_inputs);
        cout3(BLUE, "   NET_FPGA: Net enqueued", "");
        if (identifier == 0)
        {
            cout << "FPGA memory full, unable to allocate " << net_ident << "\n";
            vector<float> vec_out(n_outs, 0);
            return vec_out;
        }

        cout3(BLUE, "   NET_FPGA: Solving net", "");
        master.solve_nets();
        cout3(BLUE, "   NET_FPGA: Net solved", "");

        cout3(BLUE, "   NET_FPGA: Reading net", "");
        vector<FPGA_DATA_TYPE> int_out = master.read_net(identifier);
        cout3(BLUE, "   NET_FPGA: Net read", "");

#if fpga_performance == 1
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        fpga_info("forward performance"<<duration.count()<<"us");
        // forward_performance = duration.count();
#endif
        #if fpga_verbose>=1
            fpga_info("Salidas\n");
            for (int i = 0; i < int_out.size(); i++)
                fpga_info(i<<": "<<int_out[i]);
        #endif

        vector<float> vec_out(n_outs, 0);
        for (int i = 0; i < n_outs; i++)
            vec_out[i] = float(int_out[i]) / DECIMAL_FACTOR;

        return vec_out;
    }

    void net_fpga::enqueue_net(const vector<float> &inputs, bool reload, bool same_in, bool big_nets)
    {
        cout3(BOLDYELLOW, "enqueue_net", "")
        vector<FPGA_DATA_TYPE> int_inputs(n_ins, 0);
        for (int i = 0; i < inputs.size(); i++)
            int_inputs[i] = (FPGA_DATA_TYPE)(inputs[i] * DECIMAL_FACTOR);

        identifier = master.enqueue_net(my_data, int_inputs,reload,big_nets);
        if (identifier == 0)
            cout << "FPGA memory full, unable to allocate " << net_ident << "\n";

        cout3(BOLDYELLOW, "end_enqueue_net", "")
    }

    void net_fpga::solve_pack()
    {
        cout3(BOLDYELLOW, "solve_pack", "")
#if fpga_performance == 1
        auto start = high_resolution_clock::now();
#endif
        master.solve_nets();
#if fpga_performance == 1
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        fpga_info("solve performance "<<duration.count()<<"us");
#endif
        cout3(BOLDYELLOW, "end_solve_pack", "")
    }

    vector<float> net_fpga::read_net()
    {
        cout3(BOLDYELLOW, "read_net", "")
        vector<FPGA_DATA_TYPE> int_out = master.read_net(identifier);
        vector<float> vec_out(n_outs, 0);
        for (int i = 0; i < n_outs; i++)
            vec_out[i] = float(int_out[i]) / DECIMAL_FACTOR;

        cout3(BOLDYELLOW, "end_read_net", "")
        return vec_out;
    }

    net::builder & net_fpga::attr(int attr, float value){}
    net::builder & net_fpga::attr(int attr, int value){}

    //^ HANDLER + IMPLEMENfloatACIÓN (REVISAR MOVE OP)
    vector<float> net_fpga::run_gradient(const net::set &set) //* returns it times errors
    {
        return vector<float>(set.labels.size(), 0);
    }

    signed long net_fpga::get_gradient_performance() const
    {
#ifdef PERFORMANCE
        return gradient_performance;
#else
        // cout << "performance not enabled\n";
        return 0;
#endif
    }

    signed long net_fpga::get_forward_performance() const
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
        // cout << "FPGA deleted\n";
    }
}
