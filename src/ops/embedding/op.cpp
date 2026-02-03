#include "op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"
#include <cstring> 
#include <iostream>

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    //std::cerr << "[DEBUG C++] embedding() called. out dev=" << out->deviceType() << " idx dev=" << index->deviceType() << std::endl;
    // 检查设备
    if (out->deviceType() != LLAISYS_DEVICE_CPU) EXCEPTION_UNSUPPORTED_DEVICE;

    // 1. 准备数据指针
    std::byte* out_ptr = out->data();
    const std::byte* w_ptr = weight->data();
    const int64_t* idx_ptr = (const int64_t*)index->data();

    // 2. 获取维度
    size_t num_tokens = index->shape()[0];
    size_t hidden_size = weight->shape()[1];
    size_t elem_size = weight->elementSize(); // 自动获取 float(4) 或 f16(2) 的大小
    
    size_t row_bytes = hidden_size * elem_size; // 一行数据的字节数

    // 3. 查表与复制
    for (size_t i = 0; i < num_tokens; ++i) {
        int64_t id = idx_ptr[i];
        
        // 找到 weight 中第 id 行的起始位置
        const std::byte* src = w_ptr + id * row_bytes;
        
        // 找到 out 中第 i 行的起始位置
        std::byte* dst = out_ptr + i * row_bytes;
        
        // 内存复制
        std::memcpy(dst, src, row_bytes);
    }
}

} // namespace llaisys::ops
