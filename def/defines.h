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
    constexpr int RELU2 = 0;
    constexpr int SIGMOID = 1;
    constexpr int RELU2_SOFT_MAX = 2;
    constexpr size_t FULL_BATCH = 0;
    constexpr int NO_NORM_REG = -1;
    constexpr int REG_ONLY = 0;
    constexpr int NORM_0 = 1;
    constexpr int NORM_1 = 2;
    constexpr int NORM_2 = 3;
    constexpr int NORM_REG_0 = 4;
    constexpr int NORM_REG_1 = 5;
    constexpr int NORM_REG_2 = 6;

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
    } net_sets;

    typedef struct
    {
        std::vector<unsigned char> resized_image_data;
        size_t original_x_pos;
        size_t original_y_pos;
        size_t original_h;
        size_t original_w;
    } image_set;
}
#endif