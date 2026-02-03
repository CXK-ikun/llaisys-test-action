#include "op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"

namespace llaisys::ops {

// 1. 定义模板内核，T 可以是 float, fp16_t, bf16_t
template <typename T>
void argmax_cpu(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    const T* val_ptr = (const T*)vals->data();
    int64_t* idx_out = (int64_t*)max_idx->data();
    T* val_out = (T*)max_val->data();

    size_t numel = vals->numel();
    if (numel == 0) return;

    // 初始化：把第一个数转成 float 作为当前最大值
    float max_v = utils::cast<float>(val_ptr[0]);
    int64_t max_i = 0;

    for (size_t i = 1; i < numel; ++i) {
        // 关键点：用 utils::cast 转成 float 进行比较，解决 F16 乱码问题
        float val = utils::cast<float>(val_ptr[i]);
        if (val > max_v) {
            max_v = val;
            max_i = i;
        }
    }
    
    // 写入结果
    idx_out[0] = max_i;
    val_out[0] = utils::cast<T>(max_v); // 转回原类型
}

// 2. 主函数分发
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 简单的检查
    if (vals->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // 根据数据类型调用不同的模板
    switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            argmax_cpu<float>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_F16:
            argmax_cpu<fp16_t>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_BF16:
            argmax_cpu<bf16_t>(max_idx, max_val, vals);
            break;
        default:
            // 如果遇到不支持的类型，抛出异常
            throw std::runtime_error("Argmax: Unsupported data type");
    }
}

} // namespace llaisys::ops