#pragma once

#include <iostream>
#include <cmath>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using TensorDType = nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>>;

TensorDType pad2d(
    const TensorDType& in,
    const int pad_width = 1
);

TensorDType conv2d_vanilla(
    const TensorDType& padded_in,
    const TensorDType& filter,
    TensorDType& out_x,
    TensorDType& out_y
);

TensorDType conv2d_gradient_vanilla(
    const TensorDType& padded_in,
    const TensorDType& filter,
    const TensorDType& out_x,
    const TensorDType& out_y,
    const TensorDType& grad_output
);
