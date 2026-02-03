#include "op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"
#include <cmath> // exp

namespace llaisys::ops {

template <typename T>
void swiglu_cpu(tensor_t out, tensor_t gate, tensor_t up) {
    T* out_ptr = (T*)out->data();
    const T* g_ptr = (const T*)gate->data();
    const T* u_ptr = (const T*)up->data();

    size_t numel = out->numel();

    for (size_t i = 0; i < numel; ++i) {
        // 1. 转 float 计算
        float g_val = utils::cast<float>(g_ptr[i]);
        float u_val = utils::cast<float>(u_ptr[i]);

        // 2. SiLU = x / (1 + exp(-x))
        float silu = g_val / (1.0f + std::exp(-g_val));

        // 3. 结果 = SiLU(gate) * up
        float res = silu * u_val;

        // 4. 转回原类型
        out_ptr[i] = utils::cast<T>(res);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    if (out->deviceType() != LLAISYS_DEVICE_CPU) EXCEPTION_UNSUPPORTED_DEVICE;

    switch (out->dtype()) {
        case LLAISYS_DTYPE_F32: swiglu_cpu<float>(out, gate, up); break;
        case LLAISYS_DTYPE_F16: swiglu_cpu<fp16_t>(out, gate, up); break;
        case LLAISYS_DTYPE_BF16: swiglu_cpu<bf16_t>(out, gate, up); break;
        default: throw std::runtime_error("SwiGLU: Unsupported data type");
    }
}

} // namespace llaisys::ops