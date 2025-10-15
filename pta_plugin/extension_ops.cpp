/**
 * @file extension_add.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using namespace at;

// flash_attention_tik
// register forward implementation for NPU device
at::Tensor flash_attention_tik_impl_npu(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value)
{
    at::Tensor result = at_npu::native::empty_with_format(query.sizes(),query.options(),at_npu::native::get_npu_format(query));

    at_npu::native::OpCommand cmd;

    cmd.Name("FlashAttentionTik")
            .Input(query)
            .Input(key)
            .Input(value)
            .Output(result)
            .Run();

    return result;
}

// register forward implementation for Meta device
at::Tensor flash_attention_tik_impl_meta(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value)
{
    return empty_like(query);
}

// register the schemas for my_op and my_op_backward in the myops namespace
TORCH_LIBRARY(mindiefatik, m)
{
    m.def("flash_attention_tik(Tensor query, Tensor key, Tensor value) -> Tensor");
}

// register forward and backward implementations for the NPU device
// the device name used by the NPU device in PyTorch 2.1 and above is PrivateUse1. 
// in versions below 2.1, XLA is used. If the version is below 2.1, PrivateUse1 needs to be changed to XLA.
TORCH_LIBRARY_IMPL(mindiefatik, PrivateUse1, m)
{
    m.impl("flash_attention_tik", &flash_attention_tik_impl_npu);
}

// register forward and backward implementations for the Meta device
TORCH_LIBRARY_IMPL(mindiefatik, Meta, m)
{
    m.impl("flash_attention_tik", &flash_attention_tik_impl_meta);
}
