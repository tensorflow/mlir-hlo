load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//stablehlo:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "stablehlo_builder_filegroup",
    srcs = glob(["**"]),
)

#####
# TableGen
#####

cc_binary(
    name = "mlir_builder_tblgen",
    srcs = ["MlirBuilderTblgen.cpp"],
    includes = ["."],
    deps = [
        "//third_party/absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

####
## MlirBuilder base class
####

cc_library(
    name = "mlir_builder",
    srcs = ["MlirBuilder.cpp"],
    hdrs = ["MlirBuilder.h"],
    includes = ["."],
    deps = [
        ":attr_type_builder_util",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

####
## Attr / Type Builder Helpers
####

cc_library(
    name = "attr_type_builder_util",
    srcs = ["AttrTypeBuilderUtil.cpp"],
    hdrs = ["AttrTypeBuilderUtil.h"],
    includes = ["../../../.."],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "attr_type_builder_util_test",
    srcs = ["AttrTypeBuilderUtilTest.cpp"],
    includes = ["../../../.."],
    deps = [
        ":attr_type_builder_util",
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

#####
## Dialect-specific builders
####

gentbl_cc_library(
    name = "chlo_builder_inc",
    tbl_outs = {
        "ChloBuilder.h.inc": ["-gen-builder-decls"],
        "ChloBuilder.cpp.inc": ["-gen-builder-defs"],
        "ChloBuilder.md": ["-gen-builder-docs"],
    },
    tblgen = ":mlir_builder_tblgen",
    td_file = "//stablehlo:stablehlo/dialect/ChloOps.td",
    deps = [
        "//stablehlo:chlo_ops_td_files",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

cc_library(
    name = "chlo_builder",
    srcs = ["ChloBuilder.cpp"],
    hdrs = ["ChloBuilder.h"],
    includes = ["../../../.."],
    deps = [
        ":chlo_builder_inc",
        ":mlir_builder",
        "//stablehlo:chlo_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "func_builder_inc",
    tbl_outs = {
        "FuncBuilder.h.inc": ["-gen-builder-decls"],
        "FuncBuilder.cpp.inc": ["-gen-builder-defs"],
        "FuncBuilder.md": ["-gen-builder-docs"],
    },
    tblgen = ":mlir_builder_tblgen",
    td_file = "@llvm-project//mlir:include/mlir/Dialect/Func/IR/FuncOps.td",
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

cc_library(
    name = "func_builder",
    srcs = ["FuncBuilder.cpp"],
    hdrs = ["FuncBuilder.h"],
    includes = ["../../../.."],
    deps = [
        ":func_builder_inc",
        ":mlir_builder",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "stablehlo_builder_inc",
    tbl_outs = {
        "StablehloBuilder.h.inc": ["-gen-builder-decls"],
        "StablehloBuilder.cpp.inc": ["-gen-builder-defs"],
        "StablehloBuilder.md": ["-gen-builder-docs"],
    },
    tblgen = ":mlir_builder_tblgen",
    td_file = "//stablehlo:stablehlo/dialect/StablehloOps.td",
    deps = [
        "//stablehlo:base_td_files",
        "//stablehlo:stablehlo_ops_td_filegroup",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

cc_library(
    name = "stablehlo_builder",
    srcs = ["StablehloBuilder.cpp"],
    hdrs = ["StablehloBuilder.h"],
    includes = ["../../../.."],
    deps = [
        ":attr_type_builder_util",
        ":mlir_builder",
        ":stablehlo_builder_inc",
        "//stablehlo:stablehlo_ops",
        "//stablehlo:stablehlo_type_inference",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "stablehlo_builder_test",
    srcs = ["StablehloBuilderTest.cpp"],
    includes = ["../../../.."],
    deps = [
        ":attr_type_builder_util",
        ":func_builder",
        ":mlir_builder",
        ":stablehlo_builder",
        "//stablehlo:register",
        "//stablehlo:stablehlo_ops",
        "//testing/base/public:gunit_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "chlo_builder_test",
    srcs = ["ChloBuilderTest.cpp"],
    includes = ["../../../.."],
    deps = [
        ":attr_type_builder_util",
        ":chlo_builder",
        ":func_builder",
        ":mlir_builder",
        ":stablehlo_builder",
        "//stablehlo:register",
        "//stablehlo:stablehlo_broadcast_lowering",
        "//testing/base/public:gunit_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "mlir_builder_test",
    srcs = ["MlirBuilderTest.cpp"],
    includes = ["../../../.."],
    deps = [
        ":attr_type_builder_util",
        ":chlo_builder",
        ":func_builder",
        ":mlir_builder",
        ":stablehlo_builder",
        "//stablehlo:register",
        "//stablehlo:stablehlo_ops",
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
    ],
)
