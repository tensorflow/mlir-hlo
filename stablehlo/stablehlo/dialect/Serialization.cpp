/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "stablehlo/dialect/Serialization.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/Passes.h"

#define DEBUG_TYPE "stablehlo-compat"

namespace mlir {
namespace stablehlo {

LogicalResult serializePortableArtifact(ModuleOp module,
                                        StringRef targetVersion,
                                        raw_ostream& os,
                                        bool allowOtherDialects) {
  MLIRContext* context = module.getContext();

  // Convert StableHLO --> VHLO.
  // If allowOtherDialects is true, we will allow other dialects to be present
  // in the module, otherwise will fail if there are any other dialects present.
  {
    PassManager pm(context);
    StablehloLegalizeToVhloPassOptions options;
    options.allowOtherDialects = allowOtherDialects;
    pm.addPass(stablehlo::createStablehloLegalizeToVhloPass(options));
    if (!succeeded(pm.run(module))) {
      return failure();
    }
  }

  // Convert VHLO --> VHLO(version x.y.z).
  // Doing separately for now since we need to improve error messaging around
  // target version failures.
  {
    PassManager pm(context);
    pm.addPass(stablehlo::createVhloToVersionPass({targetVersion.str()}));
    if (!succeeded(pm.run(module))) {
      return failure();
    }
  }

  // Write bytecode with producer string "StableHLO_vX.Y.Z"
  // using the bytecode format version associated with the StableHLO release.
  auto producer = "StableHLO_v" + targetVersion.str();
  BytecodeWriterConfig writerConfig(producer);
  auto bytecodeVersion =
      vhlo::Version::fromString(targetVersion)->getBytecodeVersion();
  if (failed(bytecodeVersion)) return failure();
  writerConfig.setDesiredBytecodeVersion(bytecodeVersion.value());
  return writeBytecodeToFile(module, os, writerConfig);
}

OwningOpRef<ModuleOp> deserializePortableArtifact(StringRef sourceStr,
                                                  MLIRContext* context) {
  context->loadDialect<vhlo::VhloDialect>();
  auto module = parseSourceString<ModuleOp>(sourceStr, context);
  if (!module) {
    emitError(UnknownLoc::get(context))
        << "failed to deserialize portable artifact using StableHLO_v"
        << vhlo::Version::getCurrentVersion();
    return nullptr;
  }

  // Convert VHLO --> VHLO(current) --> StableHLO
  PassManager pm(context);
  createStablehloDeserializePipeline(pm);
  if (!succeeded(pm.run(*module))) {
    return nullptr;
  }

  return module;
}

FailureOr<vhlo::Version> getPortableArtifactVersion(llvm::StringRef bytecode) {
  auto logFailure = [&](llvm::StringRef message) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to get portable artifact version: "
                            << message << "\n");
    return failure();
  };
  // Must start with MLiRxStableHLO_vX.Y.Z, minimum length of 19.
  constexpr size_t minHeaderLength = 19;
  if (bytecode.size() < minHeaderLength) return logFailure("min header");

  // Truncate to the end of the null-terminated producer string.
  size_t pos = bytecode.find('\0');
  if (pos == llvm::StringRef::npos) return logFailure("no terminator");
  bytecode = bytecode.substr(0, pos);

  // Check if the bytecode is valid, starts with MLiR magic number.
  if (!isBytecode(
          llvm::MemoryBuffer::getMemBuffer(bytecode)->getMemBufferRef()))
    return logFailure("not bytecode");

  // Skip 4 bytes for the magic number.
  std::string stablehloHeader = "StableHLO_v";
  size_t stablehloPos = bytecode.find(stablehloHeader);
  if (stablehloPos == llvm::StringRef::npos)
    return logFailure("not a StableHLO portable artifact");

  // Skip the 11 bytes for StableHLO_v to get the StableHLO version to parse.
  StringRef version = bytecode.substr(stablehloPos + stablehloHeader.size());
  return vhlo::Version::fromString(version);
}

}  // namespace stablehlo
}  // namespace mlir
