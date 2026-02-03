#include "op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"
#include <cmath> 

namespace llaisys::ops {

template <typename T>
void rope_cpu(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    T* out_ptr = (T*)out->data();
    const T* in_ptr = (const T*)in->data();
    const int64_t* pos_ptr = (const int64_t*)pos_ids->data();

    size_t seq_len = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t head_dim = in->shape()[2];
    size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; ++s) {
        int64_t pos = pos_ptr[s];
        for (size_t h = 0; h < nhead; ++h) {
            for (size_t j = 0; j < half_dim; ++j) {
                // [精度提升] 使用 double 进行中间计算，防止大数值下的精度丢失
                double freq = 1.0 / std::pow((double)theta, (double)(2 * j) / (double)head_dim);
                double angle = (double)pos * freq;

                // 使用 double 计算 sin/cos，然后再转回 float
                float cos_val = (float)std::cos(angle);
                float sin_val = (float)std::sin(angle);

                size_t head_offset = s * (nhead * head_dim) + h * head_dim;
                size_t idx_a = head_offset + j;
                size_t idx_b = head_offset + j + half_dim;

                float x0 = utils::cast<float>(in_ptr[idx_a]);
                float x1 = utils::cast<float>(in_ptr[idx_b]);

                float y0 = x0 * cos_val - x1 * sin_val;
                float y1 = x1 * cos_val + x0 * sin_val;

                out_ptr[idx_a] = utils::cast<T>(y0);
                out_ptr[idx_b] = utils::cast<T>(y1);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    if (out->deviceType() != LLAISYS_DEVICE_CPU) EXCEPTION_UNSUPPORTED_DEVICE;
    switch (out->dtype()) {
        case LLAISYS_DTYPE_F32: rope_cpu<float>(out, in, pos_ids, theta); break;
        case LLAISYS_DTYPE_F16: rope_cpu<fp16_t>(out, in, pos_ids, theta); break;
        case LLAISYS_DTYPE_BF16: rope_cpu<bf16_t>(out, in, pos_ids, theta); break;
        default: throw std::runtime_error("RoPE: Unsupported data type");
    }
}

} // namespace llaisys::ops