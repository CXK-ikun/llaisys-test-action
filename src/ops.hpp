// src/ops.hpp
// LLAISYS Project: Unified header for all operator APIs
// This file includes all the operator headers required for the model implementation.
#include "ops/add/op.hpp"
#include "ops/rearrange/op.hpp"
#include "ops/argmax/op.hpp"
#include "ops/embedding/op.hpp"
#include "ops/linear/op.hpp"
#include "ops/rms_norm/op.hpp"
#include "ops/rope/op.hpp"
#include "ops/self_attention/op.hpp"
#include "ops/swiglu/op.hpp"

// Note: If you have implemented the optional 'rearrange' operator, uncomment the line below:
// #include "ops/rearrange/op.hpp"