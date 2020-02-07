// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear_matmul.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearMatMul,
    kOnnxDomain,
    10,
    int8_t,
    kSystolicExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>()),
    QLinearMatMul<int8_t, int8_t, int8_t>);

inline int nearestPowerOfTwo(int n)
{
    int v = n; 
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++; // next power of 2
    int x = v >> 1; // previous power of 2
    return (v - n) > (n - x) ? x : v;
}

template <>
Status QLinearMatMul<int8_t, int8_t, int8_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(3);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate offsets
  auto a_offset = ctx->Input<Tensor>(2);
  auto b_offset = ctx->Input<Tensor>(5);
  auto y_offset = ctx->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_offset),
              "QLinearMatmul : weight zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  auto a_offset_data = *(a_offset->template Data<int8_t>());
  auto b_offset_data = *(b_offset->template Data<int8_t>());
  auto y_offset_data = *(y_offset->template Data<int8_t>());
  ORT_ENFORCE(a_offset_data == 0, "Systolic can only handle zero offset for a");
  ORT_ENFORCE(b_offset_data == 0, "Systolic can only handle zero offset for b");
  ORT_ENFORCE(y_offset_data == 0, "Systolic can only handle zero offset for y");

  // validate scale
  auto a_scale = ctx->Input<Tensor>(1);
  auto b_scale = ctx->Input<Tensor>(4);
  auto y_scale = ctx->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale),
              "QLinearMatmul : weight scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  auto a_scale_data = *(a_scale->template Data<float>());
  auto b_scale_data = *(b_scale->template Data<float>());
  auto y_scale_data = *(y_scale->template Data<float>());

  ORT_ENFORCE(a_scale_data != 0, "a_scale_data cannot be 0");
  ORT_ENFORCE(b_scale_data != 0, "b_scale_data cannot be 0");
  ORT_ENFORCE(y_scale_data != 0, "y_scale_data cannot be 0");


  const float real_multiplier = (a_scale_data * b_scale_data) / y_scale_data;
  unsigned int rounded_divisor = nearestPowerOfTwo(y_scale_data / (a_scale_data * b_scale_data));

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    SystolicMultiplyi8i8_i8(static_cast<int>(helper.M()),
                            static_cast<int>(helper.N()),
                            static_cast<int>(helper.K()),
                            a->template Data<int8_t>() + helper.LeftOffsets()[i],
                            b->template Data<int8_t>() + helper.RightOffsets()[i],
                            y->template MutableData<int8_t>() + helper.OutputOffsets()[i],
                            rounded_divisor, real_multiplier);
  }

  return Status::OK();
}

} // namespace systolic
}  // namespace onnxruntime