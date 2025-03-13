import torch
from sobel_models import BaseEdgeFilter

def optimize_for_weights(
    image           : torch.Tensor, 
    edge_gt         : torch.Tensor, 
    filter_model    : BaseEdgeFilter,
    num_opt_iters   : int = 1000,
    verbose         : bool = True
):
    if (verbose):
        print(f'Init parameters = {filter_model.fx.detach()}')

    optimizer = torch.optim.Adam(filter_model.parameters(), 1e-2)
    for i in range(num_opt_iters):
        optimizer.zero_grad()
        edge_pd = filter_model.forward(image)
        loss = filter_model.computeL1Loss(edge_pd, edge_gt)
        loss.backward()


        optimizer.step()
        if (verbose):
            if ((i+1) % 100 == 0):
                print(f'Iter = {i} L1 Loss = {loss.detach()}')
    
    fx = filter_model.fx.detach().squeeze().squeeze()
    if (verbose):
        print(f'Optimized fy = {torch.transpose(fx, -1, -2)}')
        print(f'Optimized fx = {fx}')

    return fx.detach()
