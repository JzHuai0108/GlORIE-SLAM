import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def load_mono_depth(idx,cfg):
    dir_path = f"{cfg['data']['output']}/{cfg['scene']}_priors/depths"
    mono_depth_path = f"{dir_path}/{idx:05d}.npy"
    mono_depth = np.load(mono_depth_path)
    mono_depth_tensor = torch.from_numpy(mono_depth)
    return mono_depth_tensor  


class RRXIOParser:
    def __init__(self, input_folder, use_thermal, max_dt):
        self.input_folder = input_folder
        self.use_thermal = use_thermal
        self.load_poses(self.input_folder, max_dt, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, max_dt, frame_rate=-1):
        if self.use_thermal:
            if os.path.isfile(os.path.join(self.input_folder, 'gt_thermal.txt')):
                pose_list = os.path.join(self.input_folder, 'gt_thermal.txt')
            image_list = os.path.join(self.input_folder, 'thermal.txt')
            depth_list = os.path.join(self.input_folder, 'radart.txt')
        else:
            if os.path.isfile(os.path.join(self.input_folder, 'gt_visual.txt')):
                pose_list = os.path.join(self.input_folder, 'gt_visual.txt')
            image_list = os.path.join(self.input_folder, 'visual.txt')
            depth_list = os.path.join(self.input_folder, 'radarv.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []
        print('Found {} associations out of {} images, {} depth images and {} poses, keeping {} matches.'.format(
                len(associations), len(tstamp_image), len(tstamp_depth), len(tstamp_pose), len(indicies)))

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            from scipy.spatial.transform import Rotation as R
            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            rotation_matrix = R.from_quat(quat).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = trans
            self.poses += [T]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": T.tolist(),
            }

            self.frames.append(frame)


class BaseDataset(Dataset):
    def __init__(self, cfg, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.n_img = -1
        self.depth_paths = None
        self.color_paths = None
        self.poses = None
        self.image_timestamps = None

        if 'opt' in cfg['cam']:
            self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['opt']['H'], cfg['cam']['opt'][
                'W'], cfg['cam']['opt']['fx'], cfg['cam']['opt']['fy'], cfg['cam']['opt']['cx'], cfg['cam']['opt']['cy']
        else:
            self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
                'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.H_out, self.W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
        self.H_edge, self.W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        self.K_raw = self.K
        self.distorted = False
        self.distortion_model = None
        self.dist_coeffs = None
        if 'distortion' in cfg['cam']: # back compatibility
            self.distorted = True
            self.distortion_model = 'radtan'
            self.dist_coeffs = np.array(cfg['cam']['distortion'])
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.K_raw,
                self.dist_coeffs,
                np.eye(3),
                self.K,
                (self.W, self.H),
                cv2.CV_32FC1,
            )

        elif 'raw' in cfg['cam']: # new yaml format
            self.K_raw = np.array([[cfg['cam']['raw']["fx"], 0.0, cfg['cam']['raw']["cx"]],
                                   [0.0, cfg['cam']['raw']["fy"], cfg['cam']['raw']["cy"]],
                                   [0.0, 0.0, 1.0]])
            self.distorted = cfg['cam']['raw']["distorted"]
            if 'distortion_model' in cfg['cam']['raw'].keys():
                self.distortion_model = cfg['cam']['raw']['distortion_model']

            if self.distortion_model == 'radtan':
                self.dist_coeffs = np.array(
                    [
                        cfg['cam']['raw']["k1"],
                        cfg['cam']['raw']["k2"],
                        cfg['cam']['raw']["p1"],
                        cfg['cam']['raw']["p2"],
                        cfg['cam']['raw']["k3"],
                    ]
                )
                self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                    self.K_raw,
                    self.dist_coeffs,
                    np.eye(3),
                    self.K,
                    (self.W, self.H),
                    cv2.CV_32FC1,
                )
            elif self.distortion_model == "equidistant":
                self.dist_coeffs = np.array(
                    [
                        cfg['cam']['raw']["k1"],
                        cfg['cam']['raw']["k2"],
                        cfg['cam']['raw']["k3"],
                        cfg['cam']['raw']["k4"]
                    ]
                )
                self.map1x, self.map1y = cv2.fisheye.initUndistortRectifyMap(
                    self.K_raw,
                    self.dist_coeffs,
                    np.eye(3),
                    self.K,
                    (self.W, self.H),
                    cv2.CV_32FC1,
                )
            elif self.distortion_model is None:
                self.dist_coeffs = np.zeros(5)
                self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                    self.K_raw, self.dist_coeffs, np.eye(3), self.K,
                    (self.W, self.H), cv2.CV_32FC1)

        print(f"Distortion model: {self.distortion_model}")

        # retrieve input folder as temporary folder
        # tmpdir = os.environ.get('TMPDIR')
        # self.input_folder = tmpdir + '/' + cfg['data']['input_folder']
        
        # self.input_folder = cfg['data']['input_folder']
        self.input_folder = os.path.expandvars(cfg['data']['input_folder'])


    def __len__(self):
        return self.n_img

    def depthloader(self, index, depth_paths, depth_scale):
        if depth_paths is None:
            return None
        depth_path = depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        else:
            raise TypeError(depth_path)
        depth_data = depth_data.astype(np.float32) / depth_scale

        return depth_data

    def get_color(self,index):
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distorted:
            color_data_fullsize = cv2.remap(color_data_fullsize, self.map1x, self.map1y, cv2.INTER_LINEAR)

        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2

        color_data = cv2.resize(color_data_fullsize, (W_out_with_edge, H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
        return color_data
    
    def get_intrinsic(self):
        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H   
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge   
        return intrinsic 

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distorted:
            color_data_fullsize = cv2.remap(color_data_fullsize, self.map1x, self.map1y, cv2.INTER_LINEAR)

        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        outsize = (H_out_with_edge, W_out_with_edge)

        color_data = cv2.resize(color_data_fullsize, (W_out_with_edge, H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        depth_data_fullsize = self.depthloader(index,self.depth_paths,self.png_depth_scale)
        if depth_data_fullsize is not None:
            depth_data_fullsize = torch.from_numpy(depth_data_fullsize).float()
            depth_data = F.interpolate(
                depth_data_fullsize[None, None], outsize, mode='nearest')[0, 0]

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            depth_data = depth_data[:, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            depth_data = depth_data[edge:-edge, :]

        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float()
        else:
            pose = None

        return index, color_data, depth_data, pose


class Replica(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(Replica, self).__init__(cfg, device)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        max_frames = self.n_img if max_frames < 0 else max_frames 

        self.load_poses(f'{self.input_folder}/traj.txt')
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.n_img = len(self.color_paths)


    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            self.poses.append(c2w)

class ScanNet(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(ScanNet, self).__init__(cfg, device)
        stride = cfg['stride']
        max_frames = cfg['max_frames']

        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.n_img = len(self.color_paths)
        max_frames = self.n_img if max_frames < 0 else max_frames 

        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            self.poses.append(c2w)

class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

        stride = cfg['stride']
        max_frames = cfg['max_frames']
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            # timestamp tx ty tz qx qy qz qw
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class RRXIO(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(RRXIO, self).__init__(cfg, device)
        dataset_path = cfg["data"]["input_folder"]
        use_thermal = cfg['data']['modality'] == 'thermal'
        parser = RRXIOParser(dataset_path, use_thermal, 0.08)
        self.n_img = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses

        stride = cfg['stride']
        max_frames = cfg['max_frames']
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.n_img = len(self.color_paths)


class VIVIDParser:
    def __init__(self, input_folder, use_thermal, max_dt):
        self.input_folder = input_folder
        self.use_thermal = use_thermal
        self.load_poses(self.input_folder, max_dt, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, max_dt, frame_rate=-1):
        if self.use_thermal:
            if os.path.isfile(os.path.join(self.input_folder, 'gt_thermal.txt')):
                pose_list = os.path.join(self.input_folder, 'gt_thermal.txt')
            image_list = os.path.join(self.input_folder, 'thermal.txt')
            depth_list = os.path.join(self.input_folder, 'depth.txt')
        else:
            if os.path.isfile(os.path.join(self.input_folder, 'gt_visual.txt')):
                pose_list = os.path.join(self.input_folder, 'gt_visual.txt')
            image_list = os.path.join(self.input_folder, 'visual.txt')
            depth_list = os.path.join(self.input_folder, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt)
        print('Found {} associations out of {} images, {} depth images and {} poses'.format(
                len(associations), len(tstamp_image), len(tstamp_depth), len(tstamp_pose)))

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]

            from scipy.spatial.transform import Rotation as R
            rotation_matrix = R.from_quat(quat).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = trans
            self.poses += [T]
            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }
            self.frames.append(frame)


class VIVID(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(VIVID, self).__init__(cfg, device)
        dataset_path = cfg["data"]["input_folder"]
        use_thermal = cfg['data']['modality'] == 'thermal'
        parser = VIVIDParser(dataset_path, use_thermal, 0.2)
        self.n_img = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses

        stride = cfg['stride']
        max_frames = cfg['max_frames']
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.n_img = len(self.color_paths)


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGBD,
    "rrxio":   RRXIO,
    "vivid":   VIVID,
}

def get_dataset(cfg, device='cuda:0') -> BaseDataset:
    return dataset_dict[cfg['dataset']](cfg, device=device)