#include "llaisys/runtime.h"
#include "../device/runtime_api.hpp"
#include "../core/context/context.hpp" // 必须包含这个才能用 context()

extern "C" {

__export const LlaisysRuntimeAPI* llaisysGetRuntimeAPI(llaisysDeviceType_t device) {
    return llaisys::device::getRuntimeAPI(device);
}

// 补全缺失的函数
__export void llaisysSetContextRuntime(llaisysDeviceType_t device, int device_id) {
    llaisys::core::context().setDevice(device, device_id);
}

}
