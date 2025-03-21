import warp as wp
import numpy as np
import torch

@wp.kernel
def pad_array_kernel(
    x_in        : wp.array2d(dtype=wp.float32),
    padded_x_in : wp.array2d(dtype=wp.float32)
):
    i, j = wp.tid()
    padded_x_in[i + 1, j + 1] = x_in[i, j]

# Define the sobel filter apply function
@wp.kernel
def apply_filter_kernel(
    padded_x_in : wp.array2d(dtype=wp.float32),
    f_in        : wp.array2d(dtype=wp.float32),
    out         : wp.array2d(dtype=wp.float32)
):
    i, j = wp.tid()
    val_x = wp.float64(0.0)
    val_y = wp.float64(0.0)
    for ox in range(-1, 2, 1):
        for oy in range(-1, 2, 1):
            val_x += wp.float64(padded_x_in[i+ox+1][j+oy+1]) * wp.float64(f_in[ox+1][oy+1])
            val_y += wp.float64(padded_x_in[i+ox+1][j+oy+1]) * wp.float64(f_in[oy+1][ox+1]) # transpose
    wp.atomic_add(out, i, j, wp.float32(val_x * val_x + val_y * val_y))

class EvalEdgeFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, f_in):
        nx, ny          = x_in.shape
        ctx.wp_device   = wp.device_from_torch(x_in.device)
        ctx.nx          = nx
        ctx.ny          = ny
        ctx.x_in        = wp.from_torch(x_in, dtype=wp.float32)
        ctx.f_in        = wp.from_torch(f_in, dtype=wp.float32, requires_grad=True)
        
        # allocate output
        ctx.padded_x_in = wp.zeros((nx+2, ny+2), dtype=wp.float32, 
                                   requires_grad=False, device=ctx.wp_device)
        ctx.out         = wp.zeros((ctx.nx, ctx.ny), dtype=wp.float32, 
                                   requires_grad=True, device=ctx.wp_device)

        wp.launch(
            kernel  = pad_array_kernel,
            dim     = (ctx.nx, ctx.ny),
            inputs  = [ctx.x_in],
            outputs = [ctx.padded_x_in],
            device  = ctx.wp_device
        )
        
        wp.launch(
            kernel  = apply_filter_kernel,
            dim     = (ctx.nx, ctx.ny),
            inputs  = [ctx.padded_x_in, ctx.f_in],
            outputs = [ctx.out],
            device  = ctx.wp_device
        )
        return wp.to_torch(ctx.out)
    
    @staticmethod
    def backward(ctx, adj_out):
        # map incoming Torch grads to our output variables
        ctx.out.grad    = wp.from_torch(adj_out)

        wp.launch(
            kernel      = apply_filter_kernel,
            dim         = (ctx.nx, ctx.ny),
            inputs      = [ctx.padded_x_in, ctx.f_in],
            outputs     = [ctx.out],
            adj_inputs  = [ctx.padded_x_in, ctx.f_in.grad],
            adj_outputs = [ctx.out.grad],
            adjoint     = True,
            device      = ctx.wp_device
        )

        # return adjoint w.r.t. inputs
        return (None, wp.to_torch(ctx.f_in.grad))

