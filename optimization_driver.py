import torch
from sobel_models import BaseEdgeFilter

import time

def optimize_for_weights(
    image           : torch.Tensor, 
    edge_gt         : torch.Tensor, 
    filter_model    : BaseEdgeFilter,
    num_opt_iters   : int = 1000,
    verbose         : bool = True
):
    if (verbose):
        print(f'Init parameters = {filter_model.fx.detach()}')

    avg_forward = 0.0
    fc = 0
    avg_backward = 0.0
    bc = 0

    optimizer = torch.optim.Adam(filter_model.parameters(), 1e-2)
    for i in range(num_opt_iters):
        optimizer.zero_grad()

        start = time.time()

        edge_pd = filter_model.forward(image)
        loss = filter_model.computeL1Loss(edge_pd, edge_gt)

        torch.cpu.synchronize()
        torch.cuda.synchronize()

        end = time.time()
        if (i > 10):
            avg_forward += end - start
            fc += 1

        start = time.time()

        loss.backward()

        torch.cpu.synchronize()
        torch.cuda.synchronize()
        end = time.time()

        if (i > 10):
            avg_backward += end - start
            bc += 1


        optimizer.step()
        if (verbose):
            if ((i+1) % 100 == 0):
                print(f'Iter = {i} L1 Loss = {loss.detach()}')
    
    fx = filter_model.fx.detach().squeeze().squeeze()
    if (verbose):
        print(f'Optimized fy = {torch.transpose(fx, -1, -2)}')
        print(f'Optimized fx = {fx}')

    print(f'Avg forward time = {avg_forward/fc}')
    print(f'Avg backward time = {avg_backward/bc}')

    return fx.detach()
