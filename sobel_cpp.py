import numpy as np
import torch
from build import conv2d

class SobelCPPWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, filter):
        ctx.img         = img
        ctx.padded_in   = conv2d.pad2d(img, 1)
        ctx.filter      = filter
        nx = ctx.img.shape[0]
        ny = ctx.img.shape[1]
        ctx.Ix = torch.zeros((nx, ny), dtype=ctx.img.dtype, device=ctx.img.device)
        ctx.Iy = torch.zeros((nx, ny), dtype=ctx.img.dtype, device=ctx.img.device)
        ctx.Iedge = conv2d.conv2d_forward(ctx.padded_in, filter, ctx.Ix, ctx.Iy)
        ctx.Iedge.requires_grad = True
        return ctx.Iedge

    @staticmethod
    def backward(ctx, grad_output):
        grad_filter = conv2d.conv2d_backward(ctx.padded_in, ctx.filter, ctx.Ix, ctx.Iy, grad_output)
        return None, grad_filter
