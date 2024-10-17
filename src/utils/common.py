import numpy as np
import random
import torch

from skimage.color import rgb2gray
from skimage import filters


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics (fx, fy, cx, cy).

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def random_select(l, k):
    """
    Random select k values from 0 to (l-1)

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


def get_rays_from_uv(i, j, c2w, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.
    i,j are flattened.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)

    rays_d = torch.sum(dirs * c2w[:3, :3], -1)

    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def select_uv(i, j, n, depth, color,mask, device='cuda:0'):
    """
    Select n pixels (u,v) from dense (u,v).

    """
    if mask is None:
        mask = (torch.ones_like(i)>0).to(device)
    i = i.reshape(-1)
    j = j.reshape(-1)

    valid_index = mask.reshape(-1).nonzero().reshape(-1)
    sample = torch.randint(valid_index.shape[0], (n,), device=device)
    sample = sample.clamp(0, i.shape[0])
    indices = valid_index[sample]
    i = i[indices]
    j = j[indices]
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)
    depth = depth[indices]
    color = color[indices]
    return i, j, depth, color

def get_sample_uv(H0, H1, W0, W1, n, depth, color,mask, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..(H1-1), W0..(W1-1)

    """
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    if mask is not None:
        mask = mask[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device), indexing='ij')
    i = i.t()
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, depth, color,mask, device=device)
    return i, j, depth, color


def get_sample_uv_with_grad(H0, H1, W0, W1, n, image, valid_mask): 
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1
    image (numpy.ndarray): color image or estimated normal image

    """
    intensity = rgb2gray(image.cpu().numpy())
    grad_y = filters.sobel_h(intensity)
    grad_x = filters.sobel_v(intensity)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    if valid_mask is not None:
        valid_mask_np = valid_mask.cpu().numpy()
        grad_mag[~valid_mask_np] = -1

    img_size = (image.shape[0], image.shape[1])
    selected_index = np.argpartition(grad_mag, -5*n, axis=None)[-5*n:]
    indices_h, indices_w = np.unravel_index(selected_index, img_size)
    mask = (indices_h >= H0) & (indices_h < H1) & (
        indices_w >= W0) & (indices_w < W1)
    indices_h, indices_w = indices_h[mask], indices_w[mask]
    selected_index = np.ravel_multi_index(
        np.array((indices_h, indices_w)), img_size)
    samples = np.random.choice(
        range(0, indices_h.shape[0]), size=n, replace=False)

    return selected_index[samples]

def get_samples(H0, H1, W0, W1, n, fx, fy, cx, cy, c2w, depth, color, device,
                depth_filter=False, return_index=False, depth_limit=None,mask=None):
    """
    Get n rays from the image region H0..H1, W0..W1.
    fx, fy, cx, cy: intrinsics.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    i, j, sample_depth, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, depth, color,mask, device=device)

    rays_o, rays_d = get_rays_from_uv(i, j, c2w, fx, fy, cx, cy, device)
    if depth_filter:
        mask = sample_depth > 0
        if depth_limit is not None:
            mask = mask & (sample_depth < depth_limit)
        rays_o, rays_d, sample_depth, sample_color = rays_o[
            mask], rays_d[mask], sample_depth[mask], sample_color[mask]
        i, j = i[mask], j[mask]

    if return_index:
        return rays_o, rays_d, sample_depth, sample_color, i.to(torch.int64), j.to(torch.int64)
    return rays_o, rays_d, sample_depth, sample_color


def get_samples_with_pixel_grad(H0, H1, W0, W1, n_color, H, W, fx, fy, cx, cy, c2w, depth, color, device,
                                valid_mask,
                                depth_filter=True, return_index=True, depth_limit=None):
    """
    Get n rays from the image region H0..H1, W0..W1 based on color gradients, normal map gradients and random selection
    H, W: height, width.
    fx, fy, cx, cy: intrinsics.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """

    assert (n_color > 0), 'invalid number of rays to sample.'

    index_color_grad, index_normal_grad = [], []
    if n_color > 0:
        index_color_grad = get_sample_uv_with_grad(
            H0, H1, W0, W1, n_color, color, valid_mask)

    merged_indices = np.union1d(index_color_grad, index_normal_grad)

    i, j = np.unravel_index(merged_indices.astype(int), (H, W))
    i, j = torch.from_numpy(j).to(device).float(), torch.from_numpy(
        i).to(device).float()  # (i-cx), on column axis
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, fx, fy, cx, cy, device)
    i, j = i.long(), j.long()
    sample_depth = depth[j, i]
    sample_color = color[j, i]
    if depth_filter:
        mask = sample_depth > 0
        if depth_limit is not None:
            mask = mask & (sample_depth < depth_limit)
        rays_o, rays_d, sample_depth, sample_color = rays_o[
            mask], rays_d[mask], sample_depth[mask], sample_color[mask]
        i, j = i[mask], j[mask]

    if return_index:
        return rays_o, rays_d, sample_depth, sample_color, i.to(torch.int64), j.to(torch.int64)
    return rays_o, rays_d, sample_depth, sample_color


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    Returns:
        tensor(N*3*4 if batch input or 3*4): Transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    R, T = RT[:3, :3], RT[:3, 3]

    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(R)
    quad = rot.as_quat()
    quad = np.roll(quad, 1)

    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def raw2outputs_nerf_color(raw, z_vals, rays_d, device='cuda:0', coef=0.1):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, (N_rays,N_samples,4) ): prediction from model. i.e. (R,G,B) and density σ
        z_vals (tensor, (N_rays,N_samples) ): integration time. i.e. the sampled locations on this ray
        rays_d (tensor, (N_rays,3) ): direction of each ray.
        device (str, optional): device. Defaults to 'cuda:0'.
        coef (float, optional): to multipy the input of sigmoid function when calculating occupancy. Defaults to 0.1.

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty along the ray, see eq(7) in paper.
        rgb_map (tensor, (N_rays,3)): estimated RGB color of a ray.
        weights (tensor, (N_rays,N_samples) ): weights assigned to each sampled color.
    """

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
        device).expand(dists[..., :1].shape)], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1]

    raw[..., -1] = torch.sigmoid(coef*raw[..., -1])
    alpha = raw[..., -1]

    weights = alpha.float() * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(device).float(), (1.-alpha + 1e-10).float()], -1).float(), dim=-1)[:, :-1]
    weights_sum = torch.sum(weights, dim=-1).unsqueeze(-1)+1e-10
    rgb_map = torch.sum(weights[..., None] * rgb, -2)/weights_sum
    depth_map = torch.sum(weights * z_vals, -1)/weights_sum.squeeze(-1)

    tmp = (z_vals-depth_map.unsqueeze(-1))
    depth_var = torch.sum(weights*tmp*tmp, dim=1)
    return depth_map, depth_var, rgb_map, weights


def get_rays(H, W, fx, fy, cx, cy, c2w, device, crop_edge_h=0, crop_edge_w=0, return_ij=False):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    i, j = torch.meshgrid(torch.linspace(crop_edge_w, W-1-crop_edge_w, W-2*crop_edge_w),
                          torch.linspace(crop_edge_h, H-1-crop_edge_h, H-2*crop_edge_h), indexing='ij')
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H-2*crop_edge_h, W-2*crop_edge_w, 1, 3)
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    if return_ij:
        return rays_o, rays_d, i.to(torch.int64), j.to(torch.int64)
    else:
        return rays_o, rays_d

def project_point3d_to_image_batch(c2ws, pts3d, fx,fy,cx,cy, device="cuda:0"):
    ''' project each 3d point to each camera '''
    if pts3d.shape[-2] == 3:
        pts3d_homo = torch.cat([pts3d, torch.ones_like(pts3d[:,0].view(-1,1,1))], dim=-2)
    elif pts3d.shape[-2] == 4:
        pts3d_homo = pts3d
    else:
        raise NotImplementedError
    
    pts3d_homo = pts3d_homo.to(device)
    bottom = torch.tensor([0.,0.,0.,1.], dtype=torch.float32).to(c2ws.device)
    if c2ws.shape[-2:] != (4,4):
        c2ws = torch.cat([c2ws, bottom.to(device) if (c2ws.shape == 2) else bottom.view(1,1,4).repeat(c2ws.shape[0],1,1)],dim=-2).to(device)
    w2cs = torch.inverse(c2ws)
    
    pts2d_homo = w2cs @ pts3d_homo[:,None,:,:] # [Cn, 4, 4] @ [Pn, 1, 4, 1] = [Pn, Cn, 4, 1]
    pts2d = pts2d_homo[:,:,:3]
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy],
                  [.0, .0, 1.0]]).reshape(3, 3)).to(device).float()
    pts2d[:,:,0] *= -1
    uv = K @ pts2d # [3,3] @ [Pn, Cn, 3, 1] = [Pn, Cn, 3, 1]
    z = uv[:,:,-1:] + 1e-5
    uv = uv[:,:,:2]/z  # [Pn, Cn, 2, 1]
    
    uv = uv.float()
    return uv,z


def get_sample_uv_by_indices_batch(H0, H1, W0, W1, depth, color, i, j, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    if depth is not None:
        depth = depth[:, H0:H1, W0:W1].float()
    else:
        # no depth
        pass
    color = color[:, H0:H1, W0:W1].float()

    # compute new idxs
    indices = torch.stack([i,j],dim=2).to(device).float()

    if depth is not None:
        depth = torch.nn.functional.grid_sample(depth.view(list(depth.shape[0:])+[1]).permute(0, 3, 1, 2),
                            indices.view(indices.shape[0],depth.shape[0],-1,2).permute(1,0,2,3),
                            mode="nearest",align_corners=False).permute(2,0,3,1).reshape(-1)
    color = torch.nn.functional.grid_sample(color.permute(0, 3, 1, 2),
                          indices.view(indices.shape[0],color.shape[0],-1,2).permute(1,0,2,3),
                          mode="bilinear",align_corners=False).permute(2,0,3,1).contiguous().view(-1,3)
    return depth, color

def update_cam(cfg):
    """
    Update the camera intrinsics according to the pre-processing config,
    such as resize or edge crop
    """
    # resize the input images to crop_size(variable name used in lietorch)
    if 'opt' in cfg['cam']:
        H, W = cfg['cam']['opt']['H'], cfg['cam']['opt']['W']
        fx, fy = cfg['cam']['opt']['fx'], cfg['cam']['opt']['fy']
        cx, cy = cfg['cam']['opt']['cx'], cfg['cam']['opt']['cy']
    else:
        H, W = cfg['cam']['H'], cfg['cam']['W']
        fx, fy = cfg['cam']['fx'], cfg['cam']['fy']
        cx, cy = cfg['cam']['cx'], cfg['cam']['cy']

    h_edge, w_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']
    H_out, W_out = cfg['cam']['H_out'], cfg['cam']['W_out']

    fx = fx * (W_out + w_edge * 2) / W
    fy = fy * (H_out + h_edge * 2) / H
    cx = cx * (W_out + w_edge * 2) / W
    cy = cy * (H_out + h_edge * 2) / H
    H, W = H_out, W_out

    cx = cx - w_edge
    cy = cy - h_edge
    return H,W,fx,fy,cx,cy    


@torch.no_grad()
def align_scale_and_shift(prediction, target, weights):

    '''
    weighted least squares problem to solve scale and shift: 
        min sum{ 
                  weight[i,j] * 
                  (prediction[i,j] * scale + shift - target[i,j])^2 
               }

    prediction: [B,H,W]
    target: [B,H,W]
    weights: [B,H,W]
    '''

    if weights is None:
        weights = torch.ones_like(prediction).to(prediction.device)
    if len(prediction.shape)<3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        weights = weights.unsqueeze(0)  
    a_00 = torch.sum(weights * prediction * prediction, dim=[1,2])
    a_01 = torch.sum(weights * prediction, dim=[1,2])
    a_11 = torch.sum(weights, dim=[1,2])
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(weights * prediction * target, dim=[1,2])
    b_1 = torch.sum(weights * target, dim=[1,2])
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b            
    det = a_00 * a_11 - a_01 * a_01
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    error = (scale[:,None,None]*prediction+shift[:,None,None]-target).abs()
    masked_error = error*weights
    error_sum = masked_error.sum(dim=[1,2])
    error_num = weights.sum(dim=[1,2])
    avg_error = error_sum/error_num

    return scale,shift,avg_error