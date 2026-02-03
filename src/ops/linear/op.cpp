#include "op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"

namespace llaisys::ops {

// 1. 通用模板内核
template <typename T>
void linear_cpu(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    T* out_ptr = (T*)out->data();
    const T* in_ptr = (const T*)in->data();
    const T* w_ptr = (const T*)weight->data();
    const T* b_ptr = bias ? (const T*)bias->data() : nullptr;

    // 获取维度
    // In: [M, K]
    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    // Weight: [N, K] (因为是 X * W^T，所以 Weight 的第二维也是 K)
    size_t N = weight->shape()[0];

    // 朴素矩阵乘法
    for (size_t m = 0; m < M; ++m) {     // 遍历每一个输入样本
        for (size_t n = 0; n < N; ++n) { // 遍历每一个输出特征
            
            float sum = 0.0f; // 使用 float 累加，防止精度溢出
            
            for (size_t k = 0; k < K; ++k) {
                // 关键：行列索引计算 + 类型转换
                float x_val = utils::cast<float>(in_ptr[m * K + k]);
                float w_val = utils::cast<float>(w_ptr[n * K + k]); // Weight 没有转置，直接按行取
                sum += x_val * w_val;
            }
            
            // 加上 Bias (如果存在)
            if (b_ptr) {
                sum += utils::cast<float>(b_ptr[n]);
            }
            
            // 存回结果
            out_ptr[m * N + n] = utils::cast<T>(sum);
        }
    }
}

// 2. 主函数分发
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (out->deviceType() != LLAISYS_DEVICE_CPU) EXCEPTION_UNSUPPORTED_DEVICE;

    switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            linear_cpu<float>(out, in, weight, bias);
            break;
        case LLAISYS_DTYPE_F16:
            linear_cpu<fp16_t>(out, in, weight, bias);
            break;
        case LLAISYS_DTYPE_BF16:
            linear_cpu<bf16_t>(out, in, weight, bias);
            break;
        default:
            throw std::runtime_error("Linear: Unsupported data type");
    }
}

} // namespace llaisys::ops