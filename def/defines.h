#ifndef DEFINES_H
#define DEFINES_H

#include <vector>

namespace net
{
#define ASSERT
#define PERFORMANCE
#define DATA_TYPE float
    constexpr DATA_TYPE MAX_RANGE = 1;
    constexpr DATA_TYPE MIN_RANGE = -1;

    typedef struct
    {
        size_t n_ins;
        size_t n_layers;
        std::vector<size_t> n_p_l;
        std::vector<std::vector<std::vector<DATA_TYPE>>> params;
        std::vector<std::vector<DATA_TYPE>> bias;
        std::vector<std::vector<DATA_TYPE>> activations; //* valor numérico que indica qué función usar
                                                         // TODO: IMPLEMENTAR ACTIVATIONS
    } net_data;

    typedef struct
    {
        std::vector<std::vector<DATA_TYPE>> set_ins;
        std::vector<std::vector<DATA_TYPE>> set_outs;
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