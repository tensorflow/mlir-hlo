load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
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
            "transforms/legalize_mhlo/legalize_mhlo.pdll.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-pdll",
    td_file = "transforms/legalize_mhlo/legalize_mhlo.pdll",
    deps = [
        "//:hlo_ops_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:TosaDialectTdFiles",
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
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "MHLOTOSATransforms",
    srcs = [
        "transforms/legalize_mhlo/legalize_mhlo.cc",
        "transforms/prepare_mhlo/prepare_mhlo.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    includes = ["."],
    deps = [
        ":MHLOTOSAPDLLPatternsIncGen",
        ":MHLOTOSATransformsPassIncGen",
        "//:mhlo_passes",
        "//:mlir_hlo",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_binary(
    name = "mhlo-tosa-opt",
    srcs = ["mhlo_tosa_opt.cc"],
    deps = [
        ":MHLOTOSATransforms",
        "//:hlo_dialect_registration",
        "//stablehlo:register",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
    ],
)
