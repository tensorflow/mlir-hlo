# Copyright 2022 OpenXLA Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- Python -*-
# pylint: disable=undefined-variable

import os

import lit.formats
from lit.llvm import llvm_config

# Populate Lit configuration with the minimal required metadata.
# Some metadata is populated in lit.site.cfg.py.in.
config.name = 'STABLEHLO_TOSA_SUITE'
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

# Disallow reusing variables across CHECK-LABEL matches.
# A variable can eschew this (be made "global") by prefixing its name with $.
config.environment["FILECHECK_OPTS"] = "-enable-var-scope"

# Make LLVM and StableHLO tools available in RUN directives
tools = [
    'stablehlo-opt',
    'FileCheck',
    'stablehlo-translate',
]
tool_dirs = [
  config.llvm_tools_dir,
  config.stablehlo_tools_dir,
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
