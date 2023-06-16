# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This build of StableHLO is meant to be embedded in MLIR-HLO.
# As a result, its root CMakeLists.txt is different from the original
# CMakeLists.txt from https://github.com/openxla/stablehlo.
# All other files of this build of StableHLO except for this one are the same
# as the original files.
# To get access to a standalone build of StableHLO, check out the
# openxla/stablehlo repository.

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

set(STABLEHLO_ENABLE_BINDINGS_PYTHON ${MHLO_ENABLE_BINDINGS_PYTHON})

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

set(STABLEHLO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STABLEHLO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(STABLEHLO_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

# Target that provides comprehensive testing for the StableHLO repository.
# It includes both quick and slow tests (see check-stablehlo-quick and
# check-stablehlo-slow below).
add_custom_target(check-stablehlo-ci)

# Target that aggregates slow tests. Long-running test targets should go
# into dependencies of this target, and we shouldn't expect humans to run them
# regularly during development.
add_custom_target(check-stablehlo-slow)
add_dependencies(check-stablehlo-ci check-stablehlo-slow)

# Target that aggregates fast tests. We shouldn't add long-running test targets
# as a dependency of this target, since we'd want humans to routinely use this
# target during development.
add_custom_target(check-stablehlo-quick)
add_dependencies(check-stablehlo-ci check-stablehlo-quick)

add_subdirectory(stablehlo)
