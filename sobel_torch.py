import torch
import numpy as np

class EdgeFilter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fx  = torch.nn.Parameter(torch.zeros((3, 3), dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.init_parameters()
    
    def init_parameters(self):
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.fx, a=np.sqrt(3))
        
    def forward(self, x):
        fy = torch.transpose(self.fx, -2, -1)
        Ix = torch.nn.functional.conv2d(x, self.fx, bias=None, padding=1)
        Iy = torch.nn.functional.conv2d(x, fy, bias=None, padding=1)
        Iedge = Ix*Ix + Iy*Iy
        return Iedge

    def computeLoss(self, pred, gt):
        return torch.nn.functional.l1_loss(pred, gt)

def optimize_for_weights(
        image : torch.Tensor, edge_gt : torch.Tensor, filter_model : EdgeFilter
    ):

    print(f'Init parameters = {filter_model.fx.detach()}')

    optimizer = torch.optim.Adam(filter_model.parameters(), 1e-2)
    for i in range(1000):
        optimizer.zero_grad()
        edge_pd = filter_model.forward(image)
        loss = filter_model.computeLoss(edge_pd, edge_gt)
        loss.backward()
        
        optimizer.step()
        if ((i+1) % 100 == 0):
            print(f'Iter = {i} L1 Loss = {loss.detach()}')
    fx = filter_model.fx.detach().squeeze().squeeze()
    print(f'Optimized fy = {torch.transpose(fx, -1, -2)}')
    print(f'Optimized fx = {fx}')

if __name__=='__main__':
    device = "cuda:0"
    from scipy import ndimage, datasets
    ascent = datasets.ascent().astype('float32')
    ascent = ascent / 255.0

    sobel_h = ndimage.sobel(ascent, 0, mode='constant')  # horizontal gradient
    sobel_v = ndimage.sobel(ascent, 1, mode='constant')  # vertical gradient
    magnitude = sobel_h**2 + sobel_v**2

    image = torch.from_numpy(ascent).to(device).unsqueeze(0).unsqueeze(0)
    edgeImage = torch.from_numpy(magnitude).to(device).unsqueeze(0).unsqueeze(0)
    filter_model = EdgeFilter().to(device)

    optimize_for_weights(image, edgeImage, filter_model)



