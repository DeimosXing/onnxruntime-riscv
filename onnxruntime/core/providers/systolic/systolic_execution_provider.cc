// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "systolic_fwd.h"
#include "core/framework/compute_capability.h"

#include "fusion/fusion_ops.h"

namespace onnxruntime {

namespace systolic {

// Forward declarations of op kernels
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 10, int8_t, QLinearMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv_nhwc);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 1, int8_t, Fused_QLinearConv_Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 1, int8_t, Fused_QLinearConv_Relu_nhwc);

static Status RegisterSystolicKernels(KernelRegistry& kernel_registry) {
    static const BuildKernelCreateInfoFn function_table[] = {
     BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 10, int8_t, QLinearMatMul)>,
     BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv)>,
     BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv_nhwc)>,
     BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 1, int8_t, Fused_QLinearConv_Relu)>,
     BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 1, int8_t, Fused_QLinearConv_Relu_nhwc)>,  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

struct KernelRegistryAndStatus {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  Status st;
};

KernelRegistryAndStatus GetSystolicKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterSystolicKernels(*ret.kernel_registry);
  return ret;
}

}  // namespace systolic


void SystolicExecutionProvider::InsertFusedRules(FuseRuleFn rule) {
  fuse_rules_.push_back(rule);
}

std::shared_ptr<KernelRegistry> SystolicExecutionProvider::GetKernelRegistry() const {
  static systolic::KernelRegistryAndStatus k = onnxruntime::systolic::GetSystolicKernelRegistry();
  //throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

std::unique_ptr<IDataTransfer> SystolicExecutionProvider::GetDataTransfer() const {
  return onnxruntime::make_unique<CPUDataTransfer>();
}

char SystolicExecutionProvider::GetAcceleratorMode() const {
  return provider_info_.accelerator_mode;
}

std::vector<std::unique_ptr<ComputeCapability>>
SystolicExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                    const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>>
      result = IExecutionProvider::GetCapability(graph, kernel_registries);

  for (auto& rule : fuse_rules_) {
    rule(graph, result);
  }
  return result;
}

void SystolicExecutionProvider::SetupFusedRules() {
  InsertFusedRules(onnxruntime::systolic::qlinearconv_relu_fuse());
}

}  // namespace onnxruntime
