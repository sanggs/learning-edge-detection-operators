import torch
from sobel_models import BaseEdgeFilter
from torch.profiler import profile, record_function, ProfilerActivity

# import time

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

    for i in range(10):
        optimizer.zero_grad()
        edge_pd = filter_model.forward(image)
        loss = filter_model.computeL1Loss(edge_pd, edge_gt)
        loss.backward()
        optimizer.step()
        print(f'Warm up Iter = {i} L1 Loss = {loss.detach()}')

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        for i in range(10, num_opt_iters):
            with record_function(f"full_iteration"):

                optimizer.zero_grad()
                
                with record_function("forward"):
                    edge_pd = filter_model.forward(image)
                    loss = filter_model.computeL1Loss(edge_pd, edge_gt)
                
                with record_function("backward"):
                    loss.backward()
                
                with record_function("optimizer_step"):
                    optimizer.step()

                if (verbose):
                    if ((i+1) % 100 == 0):
                        print(f'Iter = {i} L1 Loss = {loss.detach()}')
    
    fx = filter_model.fx.detach().squeeze().squeeze()
    if (verbose):
        print(f'Optimized fy = {torch.transpose(fx, -1, -2)}')
        print(f'Optimized fx = {fx}')

    print("\n===== PROFILER RESULTS =====")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return fx.detach()
