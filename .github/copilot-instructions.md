# Copilot / AI Agent Instructions for LLAISYS

This file gives focused, actionable guidance for AI coding agents working on this repository.

1. Big picture
- **Backend (C++)**: core implementation in `src/` with public C APIs declared under `include/` (functions marked with `__export`). The shared library target is `llaisys` built by `xmake`.
- **Frontend (Python)**: `python/llaisys/libllaisys` contains ctypes bindings that mirror `include/` headers; `python/llaisys` provides higher-level Python wrappers and models (e.g. `models/qwen2.py`). Tests call the Python layer.

2. Build / run / test (exact commands)
- Compile C++ and produce shared libs: `xmake`
- Install shared lib into Python package: `xmake install` (copies `.so` to python/llaisys/libllaisys/ per `xmake.lua`)
- Install Python package: `pip install ./python/`
- Quick runtime test: `python test/test_runtime.py --device cpu`
- Inference example: `python test/test_infer.py --model [path/to/model]`

3. Project-specific patterns to follow
- Operator layout: each operator lives in `src/ops/<opname>/` with `op.cpp`, `op.hpp`. CPU-specific helper files are under `src/ops/<opname>/cpu/`. See `src/ops/add/` as canonical example.
- Data types: operator implementations must support Float32, Float16 and BFloat16 (tests expect these). Use helpers in `src/utils/` for naive casts.
- Tensor interface: implement and use `tensor_t` helpers in `src/tensor/` and follow the `storage`, `offset`, `meta` conventions described in `README.md`.
- API exposure: keep C ABI parity with `include/` headers; Python ctypes wrappers mirror header layout exactly.

4. Integration points & toggles
- GPU support: enable via `xmake` option `--nv-gpu` (or set `nv-gpu` config) — this defines `ENABLE_NVIDIA_API` and includes `xmake/nvidia.lua`.
- Model loading: Python model wrapper `python/llaisys/models/qwen2.py` calls into the C++ model object; tests use `transformers` tokenizer and HF `snapshot_download` in `test/test_infer.py`.

5. Debugging and validation tips
- For unit-level dev, run the corresponding Python test in `test/` (e.g. `test/ops/embedding.py`) to validate bindings + behavior.
- If a C++ change isn't visible in Python, ensure `xmake` rebuilt and `xmake install` copied the new `.so` into `python/llaisys/libllaisys/`, then reinstall or reload the Python package.

6. Files to inspect when onboarding
- Build rules: [xmake.lua](xmake.lua)
- Public API: [include/llaisys.h](include/llaisys.h)
- C++ runtime entry points: [src/llaisys](src/llaisys)
- Operator examples: [src/ops/add](src/ops/add)
- Python bindings: [python/llaisys/libllaisys](python/llaisys/libllaisys) and [python/llaisys/models/qwen2.py](python/llaisys/models/qwen2.py)

7. What agents should *not* assume
- Do not assume the Python layer will auto-reload C++ libraries — `xmake install` is required to copy the binary.
- Tests assume specific dtypes (e.g. `index` tensors are `int64`) — follow the test shapes/types.

If anything here is unclear or you want more examples (e.g. a sample operator implementation checklist), tell me which area to expand.
