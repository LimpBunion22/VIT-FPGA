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

    net_fpga::net_fpga(fpga_handler &handler) : master(handler)
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        ostringstream oss;
        oss << put_time(&tm, "%d-%m-%Y %H-%M-%S");
        net_ident = oss.str();
    }

    net_fpga::net_fpga(size_t n_ins, const vector<size_t> &n_p_l, const vector<int> &activation_type, fpga_handler &handler) : master(handler)
    {

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        ostringstream oss;
        oss << put_time(&tm, "%d-%m-%Y %H-%M-%S");
        net_ident = oss.str();

        n_layers = n_p_l.size();
        this->n_ins = n_ins % N_INS == 0 ? n_ins : n_ins + (N_INS - n_ins % N_INS);
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

        params = vector<long int>(n_params,0);
        activations = 1; // 1 -> RELU2
        bias = vector<long int>(n_neurons,0);

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

    net_fpga::net_fpga(const net::net_data &data, bool random, fpga_handler &handler)
        : n_layers(data.n_p_l.size()), n_sets(0), gradient_init(false), gradient_performance(0), forward_performance(0), master(handler)

    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        ostringstream oss;
        oss << put_time(&tm, "%d-%m-%Y %H-%M-%S");
        net_ident = oss.str();

        n_ins = data.n_ins % N_INS == 0 ? data.n_ins : data.n_ins + (N_INS - data.n_ins % N_INS);
        n_outs = data.n_p_l[n_layers - 1];
        n_p_l.reserve(n_layers);
        n_neurons = 0;
        n_params = 0;

        for (int i = 0; i < n_layers; i++)
        {
            n_p_l.emplace_back(data.n_p_l[i] % N_NEURONS == 0 ? data.n_p_l[i] : data.n_p_l[i] + (N_NEURONS - data.n_p_l[i] % N_NEURONS));
            n_neurons += n_p_l[i];
            if (i == 0)
                n_params += n_p_l[i] * n_ins;
            else
                n_params += n_p_l[i] * n_p_l[i - 1];
        }

        params = vector<long int>(n_params,0);
        activations = 1; // 1 -> RELU2
        bias = vector<long int>(n_neurons,0);
        int param_cnt = 0;
        int neuron_cnt = 0;
        for (int i = 0; i < n_layers; i++)
        {
            int n_per_n = i == 0 ? n_ins : n_p_l[i - 1];
            int o_n_per_n = i == 0 ? data.n_ins : data.n_p_l[i - 1];
            for (int j = 0; j < n_p_l[i]; j++)
            {
                for (int k = 0; k < n_per_n; k++)
                {
                    if (j < data.n_p_l[i] && k < o_n_per_n)
                        params[param_cnt] = random == true ? (rand()%4 * DECIMAL_FACTOR - DECIMAL_FACTOR) :(long int)(DECIMAL_FACTOR * data.params[i][j * o_n_per_n + k]);
                    param_cnt++;
                }
                if (j < data.n_p_l[i])
                    bias[neuron_cnt] = random == true ? (rand()%4 * DECIMAL_FACTOR - DECIMAL_FACTOR) :(long int)(DECIMAL_FACTOR * DECIMAL_FACTOR * data.bias[i][j]);
                neuron_cnt++;
            }
        }
        my_data = get_fpga_data();
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
            data.params.emplace_back(n_params);
            data.bias.emplace_back(n_p_l[i]);

            for (int j = 0; j < n_p_l[i]; j++)
            {
                data.bias[i][j] = (float)bias[neurons_cnt];
                neurons_cnt++;

                for (int k = 0; k < n_params; k++)
                {
                    data.params[i][j * n_params + k] = (float)params[params_cnt];
                    params_cnt++;
                }
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

    vector<float> net_fpga::launch_forward(const vector<float> &inputs) //* returns result
    {
        // cout << BLUE << "   NET_FPGA: Launching forward" << RESET << "\n";
        vector<long int> int_inputs(n_ins, 0);
        for (int i = 0; i < inputs.size(); i++)
            int_inputs[i] = (long int)(inputs[i] * DECIMAL_FACTOR);

#ifdef PERFORMANCE
        auto start = high_resolution_clock::now();
#endif
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
        vector<long int> int_out = master.read_net(identifier);
        cout3(BLUE, "   NET_FPGA: Net readed", "");

#ifdef PERFORMANCE
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        forward_performance = duration.count();
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
        vector<long int> int_inputs(n_ins, 0);
        for (int i = 0; i < inputs.size(); i++)
            int_inputs[i] = (long int)(inputs[i] * DECIMAL_FACTOR);

        identifier = master.enqueue_net(my_data, int_inputs);
        if (identifier == 0)
            cout << "FPGA memory full, unable to allocate " << net_ident << "\n";
    }

    void net_fpga::solve_pack()
    {
        master.solve_nets();
    }

    vector<float> net_fpga::read_net()
    {
        vector<long int> int_out = master.read_net(identifier);
        vector<float> vec_out(n_outs, 0);
        for (int i = 0; i < n_outs; i++)
            vec_out[i] = float(int_out[i]) / DECIMAL_FACTOR;

        return vec_out;
    }

    void net_fpga::set_gradient_attribute(int attribute, float value)
    {
    }

    //^ HANDLER + IMPLEMENfloatACIÓN (REVISAR MOVE OP)
    vector<float> net_fpga::launch_gradient(const net::net_set &set, size_t iterations, size_t batch_size) //* returns it times errors
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
        // cout << "FPGA deleted\n";
    }
}
