#include "op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"
#include <cmath> // sqrt

namespace llaisys::ops {

template <typename T>
void rms_norm_cpu(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    T* out_ptr = (T*)out->data();
    const T* in_ptr = (const T*)in->data();
    const T* w_ptr = (const T*)weight->data();

    // 输入通常是 [Batch*Seq, Hidden_Dim]
    // 我们把所有前面的维度看作 "rows"，最后一个维度看作 "dim"
    size_t dim = in->shape().back();
    size_t rows = in->numel() / dim;

    for (size_t i = 0; i < rows; ++i) {
        // 1. 计算平方和 (必须用 float!)
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float val = utils::cast<float>(in_ptr[i * dim + j]);
            sum_sq += val * val;
        }

        // 2. 计算 RMS
        float rms = std::sqrt(sum_sq / dim + eps);
        float scale = 1.0f / rms;

        // 3. 归一化并乘以权重
        for (size_t j = 0; j < dim; ++j) {
            float val = utils::cast<float>(in_ptr[i * dim + j]);
            float w = utils::cast<float>(w_ptr[j]);
            
            float res = val * scale * w;
            out_ptr[i * dim + j] = utils::cast<T>(res);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    if (out->deviceType() != LLAISYS_DEVICE_CPU) EXCEPTION_UNSUPPORTED_DEVICE;

    switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            rms_norm_cpu<float>(out, in, weight, eps);
            break;
        case LLAISYS_DTYPE_F16:
            rms_norm_cpu<fp16_t>(out, in, weight, eps);
            break;
        case LLAISYS_DTYPE_BF16:
            rms_norm_cpu<bf16_t>(out, in, weight, eps);
            break;
        default:
            throw std::runtime_error("RMSNorm: Unsupported data type");
    }
}

} // namespace llaisys::ops