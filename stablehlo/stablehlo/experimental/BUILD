load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")
load("//third_party/llvm/llvm-project/mlir:tblgen.bzl", "gentbl_cc_library")
load("//third_party/tensorflow:tensorflow.google.bzl", "get_compatible_with_portable")
load("//third_party/tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_applicable_licenses = ["//third_party/stablehlo:license"],  # copybara:comment
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "stablehlo_experimental_filegroup",
    srcs = glob(["**"]),
)

cc_library(
    name = "experimental_base",
    srcs = [
        "dialect/Base.cpp",
    ],
    hdrs = [
        "dialect/Base.h",
    ],
    includes = ["../.."],
    deps = [
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
    ],
)

cc_library(
    name = "experimental_stablehlo_ops",
    srcs = [
        "dialect/StablehloOps.cpp",
    ],
    hdrs = [
        "dialect/StablehloOps.h",
    ],
    includes = ["../.."],
    deps = [
        ":experimental_base",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/stablehlo:stablehlo_ops",
    ],
)

gentbl_cc_library(
    name = "experimental_stablehlo_pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
            ],
            "transforms/Passes.h.inc",
        ),
    ],
    tblgen = "//third_party/llvm/llvm-project/mlir:mlir-tblgen",
    td_file = "transforms/Passes.td",
    deps = ["//third_party/llvm/llvm-project/mlir:PassBaseTdFiles"],
)

cc_library(
    name = "experimental_stablehlo_passes",
    srcs = [
        "transforms/StablehloCanonicalizeDynamism.cpp",
        "transforms/StablehloRefineShapes.cpp",
    ],
    hdrs = [
        "transforms/Passes.h",
    ],
    includes = ["../.."],
    deps = [
        ":experimental_stablehlo_ops",
        ":experimental_stablehlo_pass_inc_gen",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:InferTypeOpInterface",
        "//third_party/llvm/llvm-project/mlir:Pass",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/llvm/llvm-project/mlir:TransformUtils",
        "//third_party/llvm/llvm-project/mlir:Transforms",
        "//third_party/stablehlo:base",
        "//third_party/stablehlo:chlo_ops",
        "//third_party/stablehlo:stablehlo_ops",
        "//third_party/stablehlo:stablehlo_ops_inc_gen",
        "//third_party/stablehlo:stablehlo_type_inference",
    ],
)

cc_binary(
    name = "experimental-stablehlo-opt",
    srcs = [
        "tools/StablehloOptMain.cpp",
    ],
    includes = ["../.."],
    deps = [
        ":experimental_stablehlo_passes",
        "//third_party/llvm/llvm-project/mlir:AllExtensions",
        "//third_party/llvm/llvm-project/mlir:AllPassesAndDialects",
        "//third_party/llvm/llvm-project/mlir:MlirOptLib",
        "//third_party/llvm/llvm-project/mlir:TosaDialect",
        "//third_party/stablehlo:interpreter_ops",
        "//third_party/stablehlo:register",
        "//third_party/stablehlo:stablehlo_passes",
        "//third_party/stablehlo:test_utils",
        "//third_party/stablehlo:tosa_passes",
    ],
)

filegroup(
    name = "test_data",
    testonly = True,
    data = [
        ":experimental-stablehlo-opt",
        "//third_party/stablehlo:test_data",
    ],
)
