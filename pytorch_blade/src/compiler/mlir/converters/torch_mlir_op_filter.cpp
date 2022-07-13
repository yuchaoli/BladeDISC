// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "compiler/mlir/converters/torch_mlir_op_filter.h"
#include <torch/script.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "common_utils/utils.h"
#include "compiler/mlir/converters/mhlo_conversion_context.h"

namespace torch {
namespace blade {
const std::unordered_set<std::string>& GetTorchMlirWhiteList();

bool IsTorchMlirSupported(const torch::jit::Node& node) {
  auto schema = node.maybeSchema();
  if (schema) {
    return GetTorchMlirWhiteList().find(schema->operator_name().name) !=
        GetTorchMlirWhiteList().end();
  } else if (node.kind().is_prim()) {
    auto name = c10::OperatorName(node.kind().toQualString(), "").name;
    return GetTorchMlirWhiteList().find(name) != GetTorchMlirWhiteList().end();
  }
  return false;
}

// clang-format off
const std::unordered_set<std::string> &GetTorchMlirWhiteList() {
  static std::unordered_set<std::string> white_list{
      "aten::__and__",
      "aten::_log_softmax",
      "aten::_softmax",
      "aten::_softmax_backward_data",
      "aten::add",
      "aten::addmm",
      "aten::unbind",
      "aten::bitwise_not",
      "aten::bmm",
      "aten::chunk",
      "aten::contiguous",
      "aten::div",
      "aten::dropout",
      "aten::eq",
      "aten::erf",
      "aten::exp",
      "aten::expand",
      "aten::expand_as",
      "aten::flatten",
      "aten::flip",
      "aten::floor_divide",
      "aten::gelu",
      "aten::gelu_backward",
      "aten::glu",
      "aten::hardtanh",
      "aten::index_select",
      "aten::layer_norm",
      "aten::leaky_relu",
      "aten::linear",
      "aten::lt",
      "aten::matmul",
      "aten::mean",
      "aten::mm",
      "aten::mul",
      "aten::native_dropout",
      "aten::native_dropout_backward",
      "aten::native_layer_norm",
      "aten::nll_loss_backward",
      "aten::nll_loss_forward",
      "aten::ne",
      "aten::neg",
      "aten::neg",
      "aten::permute",
      "aten::relu",
      "aten::relu6",
      "aten::repeat",
      "aten::reshape",
      "aten::roll",
      "aten::rsqrt",
      "aten::rsub",
      "aten::select",
      "aten::sigmoid",
      "aten::silu",
      "aten::size",
      "aten::slice",
      "aten::softmax",
      "aten::squeeze",
      "aten::sub",
      "aten::sum",
      "aten::t",
      "aten::tanh",
      "aten::tanh_backward",
      "aten::to",
      "aten::to.dtype",
      "aten::transpose",
      "aten::type_as",
      "aten::unsqueeze",
      "aten::view",
      "aten::view_as",
      "prim::Constant",
      "prim::ListConstruct",
      "prim::ListUnpack"};

  static std::once_flag flag;
  std::call_once(flag, []() {
      auto custom_ops = env::ReadStringFromEnvVar("TORCH_MHLO_OP_WHITE_LIST", "");
      std::istringstream f(custom_ops);
      std::string s;
      std::cout << "User define white list: [";
      while (getline(f, s, ';')) {
          white_list.insert(s);
          std::cout << s << ", ";
      }
      std::cout << "]" << std::endl;
  });
  return white_list;
}
// clang-format off

} //  namespace blade
} //  namespace torch
