# TODO(b/160617323): Decouple MLIR HLO from TensorFlow/XLA

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
    ],
)

cc_library(
    name = "ral_base",
    hdrs = [
        "ral/ral_base.h",
    ],
    includes = ["."],
    alwayslink = 1,
)

cc_library(
    name = "ral_logging",
    srcs = [
        "ral/ral_logging.cc",
    ],
    hdrs = [
        "ral/ral_logging.h",
    ],
    includes = ["."],
    alwayslink = 1,
)

cc_test(
    name = "ral_logging_test",
    srcs = ["ral/ral_logging_test.cc"],
    local_defines = ["PLATFORM_GOOGLE"],
    deps = [
        ":ral_base",
        ":ral_logging",
        "//testing/base/public:gunit_main",
    ],
)

cc_library(
    name = "ral_registry",
    srcs = [
        "ral/ral_registry.cc",
    ],
    hdrs = [
        "ral/ral_registry.h",
    ],
    includes = ["."],
    deps = [
        ":ral_base",
        ":ral_logging",
    ],
    alwayslink = 1,
)

cc_test(
    name = "ral_registry_test",
    srcs = ["ral/ral_registry_test.cc"],
    local_defines = ["PLATFORM_GOOGLE"],
    deps = [
        ":ral_registry",
        "//testing/base/public:gunit_main",
    ],
)
