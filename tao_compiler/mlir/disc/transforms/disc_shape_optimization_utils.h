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

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"

#ifndef TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_UTILS_H_

namespace mlir {
namespace disc_ral {

using disc_shape::SymbolicDimOp;

// Return the symbolicDim ref attribute if there is an attached disc
// shape-constraint specific attribute filed. Return nullptr if there isn't an
// attached symbolic dim ref attributes.
llvm::Optional<SmallVector<FlatSymbolRefAttr>> getRankedValueSymbolicDimRefs(
    Value value);

using Visitor = std::function<LogicalResult(Value value, RankedTensorType ty,
                                            ArrayAttr attrs)>;
// Walk each ranked tensor type values inside op.
LogicalResult walkRankedTensorValue(Operation* op, Visitor visitor);

// Updates the function types according to the types of entry block arguments
// and the types of operands of the return op of the func op. This function
// suppose that there is only one block inside the function region.
LogicalResult updateFunctionType(func::FuncOp func);

// Updates the type of all functions inside the op.
LogicalResult updateFunctionType(Operation* op);

// Reprensets a symbolic expression of symbolicDims.
class SymbolicDimExpr {
 public:
  SymbolicDimExpr() = default;
  explicit SymbolicDimExpr(Value symbol);
  SymbolicDimExpr(int64_t val, MLIRContext* context);

  explicit operator bool() const { return static_cast<bool>(expr); }

  // Returns null if not a const SymbolicDimExpr, or the const value.
  llvm::Optional<int64_t> getConstValue();

  static SymbolicDimExpr buildMulExpr(const SymbolicDimExpr& lhs,
                                      const SymbolicDimExpr& rhs);

 private:
  template <typename Combiner>
  static SymbolicDimExpr buildBinaryExpr(const SymbolicDimExpr& lhs,
                                         const SymbolicDimExpr& rhs,
                                         Combiner&& combiner);

 private:
  // TODO(disc): AffineExpr is known not able to simplify even basic symbolic
  // expression. For example, `affine_map<(d0, d1) -> (d0 + d1 - d0 - 1)>` can
  // not be simplified to `affine_map<(d0, d1) -> (d1 - 1)>`
  //
  // We may change to a professional symbolic expression tool (e.g. SymPy in
  // python, or GiNaC in C++) in the future.
  AffineExpr expr;
  SmallVector<Value> symbols;
};

class SymbolicDimMgr {
 public:
  explicit SymbolicDimMgr(ModuleOp m);

  // Loads pre-defined SymbolicDim ops from the module this mgr runs on.
  LogicalResult load();

  // Returns a new symbolicDim instance. The returned symbolicDim is owned by
  // this mgr.
  SymbolicDimOp newSymbolicDim();

  // Returns a symbolicDim which have static dim size == `val`.
  SymbolicDimOp newConstantSymbolicDim(int64_t val);

  SmallVector<SymbolicDimOp> getOrCreateSymbolicDimsForRankedValue(Value value);

  // All symbolic-equal dims form a group.
  // Returns the root SymbolicDim of the symbolic-equal symbolic dim group that
  // this SymbolicDim belongs to.
  SymbolicDimOp getRootSymbolicDim(SymbolicDimOp symbol);

  // Returns true if lhs and rhs are known to be equal.
  bool isSymbolicDimEqual(SymbolicDimOp lhs, SymbolicDimOp rhs);

  // Marks lhs and rhs have same size and try to merge lhs & rhs static known
  // info. Returns failure if failed to merge lhs & rhs.
  LogicalResult mapSymbolicDimEqual(SymbolicDimOp lhs, SymbolicDimOp rhs);

  // mark group [a0, b0, ...] and group [a1, b1, c1, ...] are group
  // multiplication equal `a0 * b0 * ... = a1 * b1 * c1 * ...`
  bool isSymbolicDimProductEqual(const SmallVectorImpl<SymbolicDimOp>& lhs,
                                 const SmallVectorImpl<SymbolicDimOp>& rhs);

  // mark `product([a0, b0, ...]) == product([a1, b1, c1, ...])`
  LogicalResult mapSymbolicDimProductEqual(
      const SmallVectorImpl<SymbolicDimOp>& lhs,
      const SmallVectorImpl<SymbolicDimOp>& rhs);

  //   SymbolicDimOp getSymbolicDimUsingRef(const FlatSymbolRefAttr& ref);

  LogicalResult save();

 private:
  // Returns next unique name for a new SymbolicDim op.
  std::string getNextName();

  // Gives a consistent order of a list op SymbolicDim Ops
  bool compareSymbolicDimOpNames(StringRef lhs, StringRef rhs);

 private:
  // The module this SymbolicDimMgr runs on.
  ModuleOp m_;

  // A unique id to generate unique name.
  int64_t nextSymbolicOpIdx_ = 0;

  // Set of name of SymbolicDim.
  // It stores all the seen names and is used to give a unique name
  // to new created symbolic dim ops.
  std::unordered_set<std::string> symbolNameSet_;

  // map a symbolic dim -> its root SymbolicDim
  // Here root symbolic dim means the representative member in the
  // symbolic-equal symbolic dim set that this symbolic dim belongs to.
  DenseMap<SymbolicDimOp, SymbolicDimOp> symbolDimUnionSet_;

  // map a concret constant value to a symbolic dim instance that represents the
  // constant.
  // Note that here we do not use DenseMap since it does not support using large
  // int64 (e.g. 9223372036854775807 generated by torch frontend) as key.
  std::unordered_map<int64_t, SymbolicDimOp> constantSymbolicDimMap_;

  // Map pre-defined the name of SymbolicDimOp to its corresponding SymbolicDim
  // instance.
  // DenseMap<std::string, SymbolicDimOp> symbolRef2symbolicDim_;

  // DenseMap<disc_shape::SymbolicDimOp, SymbolicDimOp> symbolRef2symbolicDim_;
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_UTILS_H_