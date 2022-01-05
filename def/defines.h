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
        size_t n_ins = 0;
        size_t n_layers = 0;
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
}
#endif