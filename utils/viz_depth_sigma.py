import numpy as np
import cv2
import torch


def viz_depth_sigma(depth_sigma_up, fix_range=False, name='Depth Sigma up', sigma_thresh=10.0, write=True):
    #ic(depth_sigma_up.shape)
    #ic(bg_img.shape)

    depth_sigma_up_viz = depth_sigma_up.squeeze().to(torch.float) # Visualize only the last depth...

    valid = (depth_sigma_up_viz > 0) # one shouldn't use exact equality on floats but for synthetic values it's ok
    dmin = depth_sigma_up_viz.min().item()
    if fix_range:
        dmax = sigma_thresh
    else:
        dmax = depth_sigma_up_viz.max().item()
    output = (depth_sigma_up_viz - dmin) / (dmax - dmin) # dmin -> 0.0, dmax -> 1.0
    output[output < 0] = 0 # saturate
    output[output > 1] = 1 # saturate
    #output = output / (output + 0.1)# 0 -> white, 1 -> black
    output[~valid] = 1 # black out invalid
    #output = output.pow(1/2) # most picture data is gamma-compressed
    output = (output.cpu().numpy()*255).astype(np.uint8) # scaling from 16bit to 8bit
    output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    cv2.imshow(name, output)
    if write:
        cv2.imwrite(name+".png", output)


gym = np.load('/home/pi/Documents/gauss_splat/GlORIE-SLAM/output/TUM_RGBD/demo/freiburg3_office/video.npz')
d = gym['depth_covs']
print(d.shape)
for i in range(0, d.shape[0]):
    di = torch.tensor(d[i])
    di = torch.sqrt(torch.sqrt(di))
    viz_depth_sigma(di)
    cv2.waitKey(1000)
