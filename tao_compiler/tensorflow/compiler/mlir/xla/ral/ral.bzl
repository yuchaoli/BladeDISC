def if_build_with_py_tf(if_true, if_false=[]):
    return select({
        "//tensorflow/compiler/mlir/xla/ral:build_with_py_tf": if_true,
        "//conditions:default": if_false
    })
