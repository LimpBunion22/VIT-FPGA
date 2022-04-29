#ifndef DEFINES_H
#define DEFINES_H

#include <vector>

#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */

namespace net
{
    //*attributes
    constexpr int ALPHA = 0;
    constexpr int ALPHA_DECAY = 1;
    constexpr int ERROR_THRESHOLD = 2;
    constexpr int REG_LAMBDA = 3;
    constexpr int NORM = 4;
    constexpr int DROPOUT_INTERVAL = 5;
    constexpr int ADAM = 6;
    constexpr int MOMENTUM_BETA = 7;
    constexpr int RMS_BETA = 8;

    //*selections
    constexpr size_t FULL_BATCH = 0;
    constexpr int ON = 1;
    constexpr int OFF = 0;

    //*normalization
    constexpr int DISABLE = -1;
    constexpr int REG = 0;
    constexpr int MAX = 1;
    constexpr int ABS = 2;
    constexpr int MODULO = 3;
    constexpr int MIN_MAX = 4;
    constexpr int STANDARIZATION = 5;
    constexpr int NORM_REG_0 = 6;
    constexpr int NORM_REG_1 = 7;
    constexpr int NORM_REG_2 = 8;

    //*activations
    constexpr int RELU = 0;
    constexpr int RELU2 = 1;
    constexpr int SIGMOID = 2;
    constexpr int RELU2_SOFT_MAX = 3;

    //*net definitions
    constexpr float RELU2_ALPHA = 0.1f;
    constexpr float FLOAT_CERO = 1e-10f;
    constexpr float FLOAT_INF = 1e20f;
    constexpr int RANDOM = 0;
    constexpr int CERO = 1;
    constexpr int DONT_CARE = 2;
    constexpr float MAX_RANGE = 1.0f;
    constexpr float MIN_RANGE = -MAX_RANGE;
    constexpr float RANGE = 2.0f * MAX_RANGE;

    typedef struct
    {
        size_t n_ins;
        size_t n_layers;
        std::vector<size_t> n_p_l;
        std::vector<std::vector<float>> params;
        std::vector<std::vector<float>> bias;
        std::vector<int> activation_type; //* valor numérico que indica qué función usar por capa
    } net_data;

    typedef struct
    {
        std::vector<std::vector<float>> set_ins;
        std::vector<std::vector<float>> set_outs;
        std::vector<int> labels;
    } net_set;

    typedef struct
    {
        std::vector<unsigned char> resized_image_data;
        size_t original_x_pos;
        size_t original_y_pos;
        size_t original_h;
        size_t original_w;
    } image_set;
}

namespace new_net
{
    constexpr int DISABLE = -1;

    // basic attributes
    constexpr int ALPHA = 0;
    constexpr int ALPHA_DECAY = 1;
    constexpr int ERROR_THRESHOLD = 2;
    constexpr int EPOCHS = 3;
    constexpr int BATCH_SIZE = 4;

    // algos
    constexpr int ADAM = 10;
    constexpr int MOMENTUM_BETA = 11;
    constexpr int RMS_BETA = 12;
    constexpr int REG = 13;
    constexpr int REG_LAMBDA = 14;
    constexpr int DROPOUT = 15;

    // normalization
    constexpr int MAX = 20;
    constexpr int ABS = 21;
    constexpr int MODULO = 22;
    constexpr int MIN_MAX = 23;
    constexpr int STANDARIZATION = 24;

    // activations
    constexpr int RELU = 0;
    constexpr int RELU2 = 1;
    constexpr int SIGMOID = 2;
    constexpr int RELU2_SOFT_MAX = 3;

    // net definitions
    constexpr float RELU2_ALPHA = 0.1f;
    constexpr float FLOAT_CERO = 1e-10f;
    constexpr float FLOAT_INF = 1e20f;
    constexpr int RANDOM = 0;
    constexpr int CERO = 1;
    constexpr int DONT_CARE = 2;
    constexpr float MAX_RANGE = 1.0f;
    constexpr float MIN_RANGE = -MAX_RANGE;

    typedef struct
    {
        int input_size;
        std::vector<int> n_p_l;
        std::vector<float> param_bias; // interleaved parmas_i, bias_i
        std::vector<int> activation;   // activation to use
    } layout;

    typedef struct
    {
        std::vector<float> input_data;
        std::vector<float> output_data;
        std::vector<int> labels;
        int data_num;
    } set;
}
#endif