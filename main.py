import torch
import warp as wp
from scipy import ndimage, datasets

from sobel_models import *
from optimization_driver import optimize_for_weights

if __name__=='__main__':
    device = "cuda:0"
    wp.init()
    
    # Load image
    ascent = datasets.ascent().astype('float32')
    ascent = ascent / 255.0

    sobel_h = ndimage.sobel(ascent, 0, mode='constant')  # horizontal gradient
    sobel_v = ndimage.sobel(ascent, 1, mode='constant')  # vertical gradient
    magnitude = sobel_h**2 + sobel_v**2

    image = torch.from_numpy(ascent).to(device)
    edgeImage = torch.from_numpy(magnitude).to(device)
    print(f'Optimizing using torch model')
    torch_filter_model = TorchEdgeFilter().to(device)
    optimize_for_weights(image.clone(), edgeImage.clone(), torch_filter_model)
    
    # print(f'Optimizing using warp model')
    # warp_filter_model = WarpEdgeFilter().to(device)
    # optimize_for_weights(image.clone(), edgeImage.clone(), warp_filter_model)

    print(f'Optimizing using cpp vanilla model')
    device = 'cpu'
    image = image.to(device)
    edgeImage = edgeImage.to(device)
    assert(device == 'cpu')
    cpp_filter_model = CppEdgeFilter().to(device)
    optimize_for_weights(image.clone(), edgeImage.clone(), cpp_filter_model)

    
