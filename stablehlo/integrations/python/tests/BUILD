package(licenses = ["notice"])

py_test(
    name = "smoketest",
    size = "small",
    srcs = ["smoketest.py"],
    main = "smoketest.py",
    deps = [
        # TODO(burmako): Uncomment after deleting MLIR-HLO's CHLO.
        # "//third_party/py/mlir:chlo_dialect",
        "//third_party/py/mlir:stablehlo_dialect",
    ],
)
