# MHLO -> TOSA bridge.
load("//third_party/tensorflow:tensorflow.google.bzl", "get_compatible_with_cloud")
load("//third_party/llvm/llvm-project/mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = [
        ":internal",
        "//third_party/tensorflow/compiler/xla/mlir_hlo/tosa:__subpackages__",
    ],
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
            "include/mhlo_tosa/Transforms/legalize_mhlo.pdll.h.inc",
        ),
    ],
    tblgen = "//third_party/llvm/llvm-project/mlir:mlir-pdll",
    td_file = "lib/Transforms/legalize_mhlo.pdll",
    deps = [
        "//third_party/llvm/llvm-project/mlir:OpBaseTdFiles",
        "//third_party/llvm/llvm-project/mlir:TosaDialectTdFiles",
        "//third_party/tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "MHLOTOSATransformsPassIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=MHLOTOSATransforms",
            ],
            "include/mhlo_tosa/Transforms/passes.h.inc",
        ),
    ],
    tblgen = "//third_party/llvm/llvm-project/mlir:mlir-tblgen",
    td_file = "include/mhlo_tosa/Transforms/passes.td",
    deps = [
        "//third_party/llvm/llvm-project/mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "MHLOTOSATransforms",
    srcs = [
        "lib/Transforms/legalize_mhlo.cc",
    ],
    hdrs = [
        "include/mhlo_tosa/Transforms/passes.h",
    ],
    includes = ["include"],
    textual_hdrs = [
        "lib/Transforms/passes_detail.h",
    ],
    deps = [
        ":MHLOTOSAPDLLPatternsIncGen",
        ":MHLOTOSATransformsPassIncGen",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Parser",
        "//third_party/llvm/llvm-project/mlir:Pass",
        "//third_party/llvm/llvm-project/mlir:QuantOps",
        "//third_party/llvm/llvm-project/mlir:TosaDialect",
        "//third_party/llvm/llvm-project/mlir:Transforms",
    ],
)

cc_binary(
    name = "mhlo-tosa-opt",
    srcs = ["mhlo_tosa_opt.cc"],
    deps = [
        ":MHLOTOSATransforms",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:AllPassesAndDialects",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:MlirOptLib",
        "//third_party/llvm/llvm-project/mlir:Pass",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/llvm/llvm-project/mlir:TosaDialect",
        "//third_party/tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
    ],
)
