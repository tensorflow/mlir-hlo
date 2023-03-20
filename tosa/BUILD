load("@bazel_skylib//rules:build_test.bzl", "build_test")

# MHLO -> TOSA bridge.
package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [],
)

build_test(
    name = "mhlo-tosa-opt_build_test",
    targets = [
        "mhlo-tosa-opt",
    ],
)

cc_binary(
    name = "mhlo-tosa-opt",
    srcs = ["mhlo_tosa_opt.cc"],
    deps = [
        "//:hlo_dialect_registration",
        "//stablehlo:register",
        "//third_party/tensorflow/compiler/xla/mlir_hlo/tosa/transforms:MHLOTOSATransforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
    ],
)
