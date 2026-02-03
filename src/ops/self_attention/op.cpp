#include "op.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"
#include <cmath>
#include <vector>
#include <limits>

namespace llaisys::ops {

template <typename T>
void self_attention_cpu(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    T* out_ptr = (T*)attn_val->data();
    const T* q_ptr = (const T*)q->data();
    const T* k_ptr = (const T*)k->data();
    const T* v_ptr = (const T*)v->data();

    size_t seq_len = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t dh = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkv = k->shape()[1];
    size_t group_size = nh / nkv; 

    std::vector<float> scores(total_len);

    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t h = 0; h < nh; ++h) {
            size_t kv_h = h / group_size;
            float max_score = -1e38f; 

            for (size_t pos = 0; pos < total_len; ++pos) {
                float score = 0.0f;
                for (size_t d = 0; d < dh; ++d) {
                    float q_val = utils::cast<float>(q_ptr[t * nh * dh + h * dh + d]);
                    float k_val = utils::cast<float>(k_ptr[pos * nkv * dh + kv_h * dh + d]);
                    score += q_val * k_val;
                }
                score *= scale;
                
                size_t current_abs_pos = t + (total_len - seq_len);
                if (pos > current_abs_pos) {
                    score = -std::numeric_limits<float>::infinity();
                }
                
                scores[pos] = score;
                if (score > max_score) max_score = score;
            }

            float sum_exp = 0.0f;
            for (size_t pos = 0; pos < total_len; ++pos) {
                if (scores[pos] == -std::numeric_limits<float>::infinity()) {
                    scores[pos] = 0.0f;
                } else {
                    scores[pos] = std::exp(scores[pos] - max_score);
                }
                sum_exp += scores[pos];
            }
            for (size_t pos = 0; pos < total_len; ++pos) {
                scores[pos] /= (sum_exp + 1e-9f);
            }

            for (size_t d = 0; d < dh; ++d) {
                float val = 0.0f;
                for (size_t pos = 0; pos < total_len; ++pos) {
                    float v_val = utils::cast<float>(v_ptr[pos * nkv * dh + kv_h * dh + d]);
                    val += scores[pos] * v_val;
                }
                out_ptr[t * nh * dh + h * dh + d] = utils::cast<T>(val);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) EXCEPTION_UNSUPPORTED_DEVICE;
    switch (attn_val->dtype()) {
        case LLAISYS_DTYPE_F32: self_attention_cpu<float>(attn_val, q, k, v, scale); break;
        case LLAISYS_DTYPE_F16: self_attention_cpu<fp16_t>(attn_val, q, k, v, scale); break;
        case LLAISYS_DTYPE_BF16: self_attention_cpu<bf16_t>(attn_val, q, k, v, scale); break;
        default: throw std::runtime_error("SelfAttention: Unsupported data type");
    }
}

} // namespace llaisys::ops
