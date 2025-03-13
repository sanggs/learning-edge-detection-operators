#include "conv2d_vanilla.h"
#include <cstring>

namespace nb = nanobind;

TensorDType pad2d(
    const TensorDType& img,
    const int pad_width
) {
    assert(img.device_type() == nb::device::cpu::value);
    assert(img.ndim() == 2);

    const int px = img.shape(0);
    const int py = img.shape(1);

    const int nx = img.shape(0) + (int)(2 * pad_width);
    const int ny = img.shape(1) + (int)(2 * pad_width);

    size_t output_size = size_t(nx * ny * sizeof(float));
    float* output = (float*) std::malloc(output_size);
    memset(output, 0, output_size);
    
    for (int i = 0; i < px; i++) {
        for (int j = 0; j < py; j++) {
            assert(i + pad_width < nx);
            assert(j + pad_width < ny);
            int lidx = (i + pad_width) * ny + (j + pad_width);
            assert(lidx < (nx * ny));
            output[lidx] = img(i, j);
        }
    }

    return nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>>(output, {(size_t)nx, (size_t)ny});
}

TensorDType conv2d_vanilla(
    const TensorDType& padded_in,
    const TensorDType& filter,
    TensorDType& out_x,
    TensorDType& out_y
) {
    assert(padded_in.device_type() == nb::device::cpu::value);
    assert(filter.device_type() == nb::device::cpu::value);

    assert(padded_in.ndim() == 2);
    assert(filter.ndim()    == 2);
    assert(filter.shape(0)  == filter.shape(1)); // filter is square 

    const int nx = padded_in.shape(0) - 2;
    const int ny = padded_in.shape(1) - 2;
    assert(out_x.ndim() == 2);
    assert(out_y.ndim() == 2);

    assert(out_x.shape(0) == nx);
    assert(out_x.shape(1) == ny);
    assert(out_y.shape(0) == nx);
    assert(out_y.shape(1) == ny);
    
    const int f = filter.shape(0);
    const int fe = std::floor((float)f/2.0);

    float* output = (float*) std::malloc(size_t(nx * ny * sizeof(float)));
    if (!output) {
        std::string err_string = "Error : Memory allocation for output failed nx = " + std::to_string(nx) + " ny = " + std::to_string(ny);
        throw std::runtime_error(err_string);
    }

    if (f <= 0 || f % 2 == 0) {
        throw std::runtime_error("Error: Filter size must be a positive odd number.");
    }
    assert(fe > 0);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double val_x = 0.0;
            double val_y = 0.0;
            for (int k = -fe; k < fe+1; k++) {
                for (int l = -fe; l < fe+1; l++) {
                    val_y += (double)filter(k+fe, l+fe) * (double)padded_in(i+k+fe, j+l+fe);
                    val_x += (double)filter(l+fe, k+fe) * (double)padded_in(i+k+fe, j+l+fe);
                }
            }
            int lidx = i * ny + j;
            out_y(i, j) = (float)val_y;
            out_x(i, j) = (float)val_x;
            output[lidx] = (float)(val_x * val_x + val_y * val_y);
        }
    }
    
    return nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>>(output, {(size_t)nx, (size_t)ny});
}

TensorDType conv2d_gradient_vanilla(
    const TensorDType& padded_in,
    const TensorDType& filter,
    const TensorDType& out_x,
    const TensorDType& out_y,
    const TensorDType& grad_output
) {
    assert(padded_in.device_type() == nb::device::cpu::value);
    assert(filter.device_type() == nb::device::cpu::value);

    assert(padded_in.ndim() == 2);
    assert(filter.ndim() == 2);
    assert(filter.shape(0) == filter.shape(1)); // filter is square 

    const int nx = padded_in.shape(0) - 2;
    const int ny = padded_in.shape(1) - 2;
    
    const int f = filter.shape(0);
    const int fe = std::floor((float)f/2.0);
    if (f <= 0 || f % 2 == 0) {
        throw std::runtime_error("Error: Filter size must be a positive odd number.");
    }
    assert(fe > 0);

    size_t grad_filter_size = size_t(f * f * sizeof(float));
    float* grad_filter = (float*) std::malloc(grad_filter_size);
    if (!grad_filter) {
        std::string err_string = "Error : Memory allocation for filter gradient failed nx = " + std::to_string(nx) + " ny = " + std::to_string(ny);
        throw std::runtime_error(err_string);
    }
    memset(grad_filter, 0, grad_filter_size);
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = -fe; k < fe+1; k++) {
                for (int l = -fe; l < fe+1; l++) {
                    int lidx_y = (k+fe) * f + l+fe;
                    int lidx_x = (l+fe) * f + k+fe;
                    
                    // Compute gradient for both x and y components
                    double val_x = 2.0 * (double)padded_in(i+k+fe, j+l+fe) * (double)out_x(i, j) * (double)grad_output(i, j);
                    double val_y = 2.0 * (double)padded_in(i+k+fe, j+l+fe) * (double)out_y(i, j) * (double)grad_output(i, j);
                    
                    // Accumulate gradients (note the different indices for x and y)
                    grad_filter[lidx_x] += (float)(val_x);
                    grad_filter[lidx_y] += (float)(val_y);
                }
            }   
        }
    }
    return nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>>((void*)grad_filter,
            {(size_t)f, (size_t)f});
}

NB_MODULE(conv2d, m) {
    m.def("pad2d", &pad2d,
        nb::rv_policy::reference_internal);
    m.def("conv2d_vanilla", &conv2d_vanilla, 
        nb::rv_policy::reference_internal);
    m.def("conv2d_gradient_vanilla", &conv2d_gradient_vanilla,
        nb::rv_policy::reference_internal);
}
