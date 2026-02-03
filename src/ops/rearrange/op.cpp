#include "op.hpp"
#include <cstring>
namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    size_t bytes = 1;
    for(auto s : in->shape()) bytes *= s;
    bytes *= sizeof(float);
    memcpy(out->data(), in->data(), bytes);
}
}
