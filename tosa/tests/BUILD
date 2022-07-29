load("//third_party/tensorflow:tensorflow.google.bzl", "filegroup")
load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//third_party/llvm/llvm-project/mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//third_party/llvm/llvm-project/llvm:FileCheck",
        "//third_party/tensorflow/compiler/xla/mlir_hlo/tosa:mhlo-tosa-opt",
    ],
)
