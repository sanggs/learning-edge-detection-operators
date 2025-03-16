#include "conv2d.h"
#include <cstring>
#include <omp.h>

// #define USE_AVX512

#ifdef USE_AVX512
#include <immintrin.h>
#endif

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

    /*const int pixels = (int)(px * py);
    #pragma omp parallel for simd
    for (int p = 0; p < pixels; p++) {
        const int i = (int)p/py;
        const int j = (int)p%py;
        assert(i + pad_width < nx);
        assert(j + pad_width < ny);
        int lidx = (i + pad_width) * ny + (j + pad_width);
        assert(lidx < (nx * ny));
        output[lidx] = img(i, j);
    }*/

    #pragma omp parallel for shared(img, output)
    for (int i = 0; i < px; i++) {
        std::memcpy(output + (i + pad_width) * ny + pad_width, 
                    img.data() + i * py, 
                    py * sizeof(float));
    }

    return nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>>(output, {(size_t)nx, (size_t)ny});
}

TensorDType conv2d_forward(
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

    const int pixels = (int)(nx * ny);

#ifdef USE_AVX512
    #pragma omp parallel
    {

        __m512 filter_vecs[f][f];
        for (int k = 0; k < f; k++)
            for (int l = 0; l < f; l++) {
                filter_vecs[k][l] = _mm512_set1_ps(filter(k, l));
            }

        #pragma omp for
        for (int p = 0; p < pixels; p += 16) {
            
            int process_count = std::min(16, pixels - p);
            
            int base_i[16], base_j[16];
            for (int b = 0; b < process_count; b++) {
                int idx = p + b;
                base_i[b] = idx / ny;
                base_j[b] = idx % ny;
            }
            
            if (process_count == 16) {
                __m512 acc_val_x, acc_val_y;
                acc_val_x = _mm512_setzero_ps();
                acc_val_y = _mm512_setzero_ps();
                
                for (int k = -fe; k < fe+1; k++) {
                    for (int l = -fe; l < fe+1; l++) {
                        // Broadcast filter value
                        __m512 filter_val_y = filter_vecs[k + fe][l + fe]; //_mm512_set1_ps(filter(k+fe, l+fe));
                        __m512 filter_val_x = filter_vecs[l + fe][k + fe]; //_mm512_set1_ps(filter(l+fe, k+fe));
                        
                        // Load image pixel values
                        float input_vals[16];
                        for (int b = 0; b < 16; b++) {
                            input_vals[b] = padded_in(base_i[b]+k+fe, base_j[b]+l+fe);
                        }
                        __m512 input_vec = _mm512_load_ps(input_vals);
                        
                        // Fuse Multiply Add
                        acc_val_y = _mm512_fmadd_ps(filter_val_y, input_vec, acc_val_y);
                        acc_val_x = _mm512_fmadd_ps(filter_val_x, input_vec, acc_val_x);
                    }
                }
                
                // Copy result back to array
                float result_x[16] __attribute__((aligned(64)));
                float result_y[16] __attribute__((aligned(64)));
                
                _mm512_store_ps(result_x, acc_val_x);
                _mm512_store_ps(result_y, acc_val_y);
                
                for (int b = 0; b < 16; b++) {
                    int idx = p + b;
                    int i = base_i[b];
                    int j = base_j[b];
                    
                    out_x(i, j) = result_x[b];
                    out_y(i, j) = result_y[b];
                    output[idx] = result_x[b] * result_x[b] + result_y[b] * result_y[b];
                }
            } 
            else {
                // Process remaining pixels (< 16) using scalar operations
                for (int b = 0; b < process_count; b++) {
                    int i = base_i[b];
                    int j = base_j[b];
                    float val_x = 0.0f;
                    float val_y = 0.0f;
                    
                    // For each filter position
                    for (int k = -fe; k < fe+1; k++) {
                        for (int l = -fe; l < fe+1; l++) {
                            val_y += filter(k+fe, l+fe) * padded_in(i+k+fe, j+l+fe);
                            val_x += filter(l+fe, k+fe) * padded_in(i+k+fe, j+l+fe);
                        }
                    }
                    
                    // Store results
                    out_y(i, j) = val_y;
                    out_x(i, j) = val_x;
                    output[p + b] = val_x * val_x + val_y * val_y;
                }
            }
        }
    }
#else
    #pragma omp parallel for simd
    for (int p = 0; p < pixels; p++) {
        const int i = (int)p/ny;
        const int j = (int)p%ny;
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
#endif

    return nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>>(output, {(size_t)nx, (size_t)ny});
}

TensorDType conv2d_backward(
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
    
    const int pixels = (int)(nx * ny);

#ifdef USE_AVX512

    #pragma omp parallel
    {
        float thread_grad[9];
        memset(thread_grad, 0, grad_filter_size);

        float const_scale[16];
        for (int b = 0; b < 16; b++)
            const_scale[b] = 2.0;
        __m512 const_scale_vec = _mm512_load_ps(const_scale);

        #pragma omp for
        for (int p = 0; p < pixels; p += 16) {

            int process_count = std::min(16, pixels - p);
            
            int base_i[16], base_j[16];
            for (int b = 0; b < process_count; b++) {
                int idx = p + b;
                base_i[b] = idx / ny;
                base_j[b] = idx % ny;
            }
            
            if (process_count == 16) {
                // Load out_x, out_y, grad_vec into vectorized floats
                float out_x_vals[16];
                float out_y_vals[16];
                float grad_vals[16];
                for (int b = 0; b < 16; b++) {
                    out_x_vals[b] = out_x(base_i[b], base_j[b]);
                    out_y_vals[b] = out_y(base_i[b], base_j[b]);
                    grad_vals[b] = grad_output(base_i[b], base_j[b]);
                }
                __m512 out_x_vec = _mm512_load_ps(out_x_vals);
                __m512 out_y_vec = _mm512_load_ps(out_y_vals);
                __m512 grad_vec = _mm512_load_ps(grad_vals);
                
                // Compute (2 * out_x * grad_output) & (2 * out_y * grad_output)
                __m512 factor_x_tmp = _mm512_mul_ps(out_x_vec, const_scale_vec);
                __m512 factor_y_tmp = _mm512_mul_ps(out_y_vec, const_scale_vec);

                __m512 factor_x = _mm512_mul_ps(factor_x_tmp, grad_vec);
                __m512 factor_y = _mm512_mul_ps(factor_y_tmp, grad_vec);
                
                for (int k = -fe; k < fe+1; k++) {
                    for (int l = -fe; l < fe+1; l++) {
                        int lidx_y = (k+fe) * f + l+fe;
                        int lidx_x = (l+fe) * f + k+fe;

                        // Load image pixel values
                        float input_vals[16];
                        for (int b = 0; b < 16; b++) {
                            input_vals[b] = padded_in(base_i[b]+k+fe, base_j[b]+l+fe);
                        }
                        __m512 input_vec = _mm512_load_ps(input_vals);
                        
                        // Multiply
                        __m512 acc_val_y = _mm512_mul_ps(factor_y, input_vec);
                        __m512 acc_val_x = _mm512_mul_ps(factor_x, input_vec);
                        
                        // Add and reduce
                        float grad_y = _mm512_reduce_add_ps(acc_val_y);
                        float grad_x = _mm512_reduce_add_ps(acc_val_x);
                        
                        thread_grad[lidx_x] += (float)(grad_x);
                        thread_grad[lidx_y] += (float)(grad_y);
                    }
                }
            }
            else {
                for (int b = 0; b < process_count; b++) {
                    const int i = base_i[b];
                    const int j = base_j[b];
                    for (int k = -fe; k < fe+1; k++) {
                        for (int l = -fe; l < fe+1; l++) {
                            int lidx_y = (k+fe) * f + l+fe;
                            int lidx_x = (l+fe) * f + k+fe;
                            
                            // Compute gradient for both x and y components
                            double val_x = 2.0 * (double)padded_in(i+k+fe, j+l+fe) * (double)out_x(i, j) * (double)grad_output(i, j);
                            double val_y = 2.0 * (double)padded_in(i+k+fe, j+l+fe) * (double)out_y(i, j) * (double)grad_output(i, j);
                            
                            // Accumulate gradients (note the different indices for x and y)
                            thread_grad[lidx_x] += (float)(val_x);
                            thread_grad[lidx_y] += (float)(val_y);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < f*f; i++)
                grad_filter[i] += thread_grad[i];
        }
    }    

#else
    #pragma omp parallel
    {
        float* thread_grad = (float*) std::malloc(grad_filter_size);
        if (!thread_grad) {
            std::string err_string = "Error : Memory allocation for filter gradient failed nx = " + std::to_string(nx) + " ny = " + std::to_string(ny);
            throw std::runtime_error(err_string);
        }
        memset(thread_grad, 0, grad_filter_size);

        #pragma omp for
        for (int p = 0; p < pixels; p++) {
            const int i = (int)p/ny;
            const int j = (int)p%ny;
            for (int k = -fe; k < fe+1; k++) {
                for (int l = -fe; l < fe+1; l++) {
                    int lidx_y = (k+fe) * f + l+fe;
                    int lidx_x = (l+fe) * f + k+fe;
                    
                    // Compute gradient for both x and y components
                    double val_x = 2.0 * (double)padded_in(i+k+fe, j+l+fe) * (double)out_x(i, j) * (double)grad_output(i, j);
                    double val_y = 2.0 * (double)padded_in(i+k+fe, j+l+fe) * (double)out_y(i, j) * (double)grad_output(i, j);
                    
                    // Accumulate gradients (note the different indices for x and y)
                    thread_grad[lidx_x] += (float)(val_x);
                    thread_grad[lidx_y] += (float)(val_y);
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < f*f; i++)
            grad_filter[i] += thread_grad[i];
        }

        free(thread_grad);
        thread_grad = nullptr;
    }
#endif
    return nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>>((void*)grad_filter,
            {(size_t)f, (size_t)f});
}

NB_MODULE(conv2d, m) {
    m.def("pad2d", &pad2d,
        nb::rv_policy::reference_internal);
    m.def("conv2d_forward", &conv2d_forward, 
        nb::rv_policy::reference_internal);
    m.def("conv2d_backward", &conv2d_backward,
        nb::rv_policy::reference_internal);
}
