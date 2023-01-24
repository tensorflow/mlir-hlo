# Legalizations and transforms for MHLO -> TOSA.

load("//third_party/tensorflow/tsl:tsl.google.bzl", "get_compatible_with_cloud")
load("//third_party/llvm/llvm-project/mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["//third_party/tensorflow:license"],  # copybara:comment
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [],
)

gentbl_cc_library(
    name = "MHLOTOSAPDLLPatternsIncGen",
    tbl_outs = [
        (
            ["-x=cpp"],
            "legalize_mhlo/legalize_mhlo.pdll.h.inc",
        ),
    ],
    tblgen = "//third_party/llvm/llvm-project/mlir:mlir-pdll",
    td_file = "legalize_mhlo/legalize_mhlo.pdll",
    deps = [
        "//third_party/llvm/llvm-project/mlir:OpBaseTdFiles",
        "//third_party/llvm/llvm-project/mlir:TosaDialectTdFiles",
        "//third_party/tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "MHLOTOSATransformsPassIncGen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MHLOTOSATransforms",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "//third_party/llvm/llvm-project/mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "//third_party/llvm/llvm-project/mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "MHLOTOSATransforms",
    srcs = [
        "legalize_mhlo/legalize_mhlo.cc",
        "prepare_mhlo/prepare_mhlo.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    includes = ["."],
    deps = [
        ":MHLOTOSAPDLLPatternsIncGen",
        ":MHLOTOSATransformsPassIncGen",
        "//:mlir_hlo",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Parser",
        "//third_party/llvm/llvm-project/mlir:Pass",
        "//third_party/llvm/llvm-project/mlir:QuantOps",
        "//third_party/llvm/llvm-project/mlir:TosaDialect",
        "//third_party/llvm/llvm-project/mlir:Transforms",
        "//third_party/tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
    ],
)
