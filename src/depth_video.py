import numpy as np
import gtsam
import torch
import lietorch
import droid_backends_glorie_slam as droid_backends
import src.geom.ba
from torch.multiprocessing import Value

from src.modules.droid_net import cvx_upsample, cvx_upsample_pow
import src.geom.projective_ops as pops
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor

class DepthVideo:
    ''' store the estimated poses and depth maps, 
        shared between tracker and mapper '''
    def __init__(self, cfg, printer):
        self.cfg =cfg
        self.output = f"{cfg['data']['output']}/{cfg['setting']}/{cfg['scene']}"
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.counter = Value('i', 0) # current keyframe count
        buffer = cfg['tracking']['buffer']
        self.BA_type = cfg['tracking']['backend']['BA_type']
        self.mono_thres = cfg['tracking']['mono_thres']
        self.device = cfg['device']
        self.down_scale = 8
        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device=self.device, dtype=torch.uint8)

        # whether the valid_depth_mask is calculated/updated, if dirty, not updated, otherwise, updated
        self.dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_() 
        # whether the corresponding part of pointcloud is deformed w.r.t. the poses and depths 
        self.npc_dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()

        self.poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.zeros = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.depths_cov = torch.zeros(buffer, ht // self.down_scale, wd // self.down_scale, device=self.device,
                                dtype=torch.float).share_memory_()
        self.depths_cov_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.depth_scale = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.depth_shift = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.valid_depth_mask_small = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.bool).share_memory_()        

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, 1, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.printer = printer

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]


        if item[4] is not None:
            
            mono_depth = item[4][self.down_scale//2-1::self.down_scale,
                                 self.down_scale//2-1::self.down_scale]
            self.mono_disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()
        depth_cov_up = cvx_upsample_pow(self.depths_cov[ix].unsqueeze(-1), mask, pow=1.0)
        self.depths_cov_up[ix] = depth_cov_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.depths_cov[:self.counter.value] *= (s**2)
            self.poses[:self.counter.value,:3] *= s
            self.set_dirty(0,self.counter.value)


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d


    def get_gt_priors_and_values(self, kf_id, f_id):
        w2c = lietorch.SE3(self.poses[kf_id].clone()) # cam_T_world
        c2w = w2c.inv().matrix().cpu().numpy()
        gt_pose = c2w # world_T_cam

        pose_key = gtsam.symbol_shorthand.X(f_id)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]))
        pose_prior = gtsam.PriorFactorPose3(pose_key, gtsam.Pose3(gt_pose),
                                            pose_noise)

        x0 = gtsam.Values()
        x0.insert(pose_key, gtsam.Pose3(gt_pose))

        graph = gtsam.NonlinearFactorGraph()
        graph.push_back(pose_prior)
        return x0, graph

    def ba_for_cov(self,
           gru_estimated_flow,
           gru_estimated_flow_weight,
           damping,
           ii,
           jj,
           kf0=0,
           kf1=None) -> None:
        """
        To be consistent to nerf_slam cov computation, kf0 should be computed like
            kf0 = max(0, ii.min().item())
        But jhuai's test found that it is unneeded as kf0 will become greater than 0 after some frames anyway.
        The following variables should be prepared beforehand:
            world_T_cam0_t0 is predefined.
            world_T_body is the inverse of cam0_T_world which are for keyframes.
            cam0_T_body is identity.

        cam0_intrinsics corresponds to droid-slam depth_video.intrinsics
        cam0_T_world corresponds to droid-slam depth_video.poses
        cam0_idepths correponds to droid-slam depth_video.disps
        """
        if kf1 is None:
            kf1 = max(ii.max().item(), jj.max().item()) + 1
        cam0_T_body = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float)
        world_T_body = (lietorch.SE3(self.poses).inv() * lietorch.SE3(cam0_T_body[None])).vec()
        N = kf1 - kf0
        ht_down = self.ht // self.down_scale
        wd_down = self.wd // self.down_scale
        HW = ht_down * wd_down
        kx = torch.unique(ii)

        kf_ids = [i + kf0 for i in range(kf1 - kf0)]
        # f_ids = [self.kf_idx_to_f_idx[kf_id] for kf_id in kf_ids]
        f_ids = kf_ids # It does not make difference to use kf_ids.
        Xii = np.array([gtsam.symbol_shorthand.X(f_id) for f_id in f_ids])

        initial_priors = None
        if f_ids[0] == 0:
            # "You need to add initial prior, or you'll have ill-cond hessian!"
            _, initial_priors = self.get_gt_priors_and_values(
                kf_ids[0], f_ids[0])

        # construct the Hessian of the BA problem
        x0 = gtsam.Values()
        linear_factor_graph = gtsam.GaussianFactorGraph()

        for i in range(N):
            kf_id = i + kf0
            x0.insert(
                Xii[i],
                gtsam.Pose3(
                    lietorch.SE3(
                        world_T_body[kf_id]).matrix().cpu().numpy()))

        H, v, Q, E, w = droid_backends.reduced_camera_matrix(
            self.poses, self.disps,
            self.intrinsics[0], cam0_T_body,
            self.zeros, gru_estimated_flow,
            gru_estimated_flow_weight, damping, ii, jj, kf0, kf1)

        vision_factors = gtsam.GaussianFactorGraph()
        H = torch.nn.functional.unfold(H[None, None], (6, 6),
                                       stride=6).permute(2, 0, 1).view(
                                           N, N, 6, 6)
        v = torch.nn.functional.unfold(v[None, None], (6, 1),
                                       stride=6).permute(2, 0,
                                                         1).view(N, 6)
        H[range(N), range(N)] /= N
        v[:] /= N
        upper_triangular_indices = torch.triu_indices(N, N)
        for i, j in zip(upper_triangular_indices[0],
                        upper_triangular_indices[1]):
            if i == j:
                vision_factors.add(
                    gtsam.HessianFactor(Xii[i], H[i, i].cpu().numpy(),
                                        v[i].cpu().numpy(), 0.0))
            else:
                vision_factors.add(
                    gtsam.HessianFactor(Xii[i], Xii[j], H[i,
                                                          i].cpu().numpy(),
                                        H[i, j].cpu().numpy(),
                                        v[i].cpu().numpy(),
                                        H[j, j].cpu().numpy(),
                                        v[j].cpu().numpy(), 0.0))
        linear_factor_graph.push_back(vision_factors)

        if initial_priors is not None:
            linear_factor_graph.push_back(initial_priors.linearize(x0))

        # compute covariance
        H, v = linear_factor_graph.hessian()
        L = None
        try:
            L = torch.linalg.cholesky(
                torch.as_tensor(H, device=self.device, dtype=torch.float))
        except Exception as e:
            print(e)
        if L is not None:
            identity = torch.eye(L.shape[0], device=L.device)
            L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
            if torch.isnan(L_inv).any():
                print("NANs in L_inv!!")
                raise

            P = N
            D = L.shape[0] // P
            assert D == 6

            Ei = E[:P]
            Ejz = E[P:P + ii.shape[0]]
            M = Ejz.shape[0]
            assert M == ii.shape[0]
            kx, kk = torch.unique(ii, return_inverse=True)
            K = kx.shape[0]

            min_ii_jj = min(ii.min(), jj.min())

            Ej = torch.zeros(K, K, D, HW, device=self.device)
            Ej[jj - min_ii_jj, ii - min_ii_jj] = Ejz
            Ej = Ej[kf0 - min_ii_jj:kf1 - min_ii_jj].view(P, K, D, HW)
            Ej[range(P),
               kf0 - min_ii_jj:kf1 - min_ii_jj, :, :] = Ei[range(P), :, :]

            E_sum = Ej
            E_sum = E_sum.view(P, K, D, HW)
            E_sum = E_sum.permute(0, 2, 1, 3).reshape(P * D, K * HW)
            Q_ = Q.view(K * HW, 1)
            F = torch.matmul(Q_ * E_sum.t(), L_inv)
            # jhuai: I think F should be computed with L_inv.t() as below,
            # but it often causes extraordinary large values like 1e6 in depth_cov.
            # F = torch.matmul(Q_ * E_sum.t(), L_inv.t())
            F2 = torch.pow(F, 2)
            delta_cov = F2.sum(dim=-1)

            z_cov = Q_.squeeze() + delta_cov
            z_cov = z_cov.view(K, ht_down, wd_down)
            depth_cov = z_cov / self.disps[kx]**4

            # for j, idx in enumerate(kx):
            #     if not torch.allclose(self.depths_cov[idx], depth_cov[j], rtol=1e-2, atol=1):
            #         print("Depth cov mismatch at {}. Stored:\n{}\nComputed:\n{}".format(
            #             idx, self.depths_cov[idx, 15:20, 20:25], depth_cov[j, 15:20, 20:25]))

            self.depths_cov[kx] = depth_cov

    def dspo(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False, opt_type="pose_depth", depth_cov=False):
        """ Disparity, Scale and Pose Optimization (DSPO) layer, 
            checked the paper (and supplementary) for detailed explanation 

            opt_type: "pose_depth",  stage 1, optimize camera poses and disparity maps together, eq.16 in the paper,
                                              same as DBA
                      "depth_scale", stage 2, optimize disparity maps, scales and shifts together, eq.17 in the paper
            stage 1 and stage 2 are run alternatingly
        """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            if opt_type == "pose_depth":
                target = target.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
                weight = weight.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                    target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only, False)
                self.disps.clamp_(min=1e-5)
                if depth_cov and not motion_only:
                    # we do not call depth cov recovery in ba motion_only as
                    # ii does not cover [kf0, kf1) causing first dimension mismatch
                    # between eta and m in reduced_camera_matrix_cuda.
                    self.ba_for_cov(target, weight, eta, ii, jj, kf0=t0-1, kf1=t1)
                return True

            elif opt_type == "depth_scale":
                poses = lietorch.SE3(self.poses[None])
                disps = self.disps[None]
                scales = self.depth_scale
                shifts = self.depth_shift
                ignore_frames = 0
                self.update_valid_depth_mask(up=False)
                curr_idx = self.counter.value-1
                mono_d = self.mono_disps[:curr_idx+1]
                est_d = self.disps[:curr_idx+1]
                valid_d = self.valid_depth_mask_small[:curr_idx+1]
                scale_t, shift_t, error_t = align_scale_and_shift(mono_d, est_d, valid_d)
                avg_disps = est_d.mean(dim=[1,2])

                scales[:curr_idx+1]=scale_t
                shifts[:curr_idx+1]=shift_t

                target_t,weight_t,eta_t,ii_t,jj_t = target,weight,eta,ii,jj

                ################################################################
                if self.mono_thres:
                    # remove the edges which contains poses with bad mono depth
                    invalid_mono_mask = (error_t/avg_disps > self.mono_thres)| \
                                        (error_t.isnan())|\
                                        (scale_t < 0)|\
                                        (valid_d.sum(dim=[1,2]) < 
                                        valid_d.shape[1]*valid_d.shape[2]*0.5)
                    invalid_mono_index, = torch.where(invalid_mono_mask.clone())
                    invalid_ii_mask = (ii<0)
                    idx_in_ii = torch.unique(ii)
                    valid_eta_mask = (idx_in_ii >= 0)
                    for idx in invalid_mono_index:
                        invalid_ii_mask =  invalid_ii_mask | (ii == idx) | (jj == idx) 
                    target_t = target[:,~invalid_ii_mask]
                    weight_t = weight[:,~invalid_ii_mask]
                    ii_t = ii[~invalid_ii_mask]
                    jj_t = jj[~invalid_ii_mask]
                    idx_in_ii_t = torch.unique(ii_t)
                    valid_eta_mask = torch.tensor([idx in idx_in_ii_t for idx in idx_in_ii]).to(self.device)
                    eta_t = eta[valid_eta_mask]
                ################################################################
                success = False
                for _ in range(itrs):
                    if self.counter.value > ignore_frames and ii_t.shape[0]>0:
                        poses, disps, wqs = src.geom.ba.BA_with_scale_shift(
                            target_t,weight_t,eta_t,poses,disps,
                            self.intrinsics[None],ii_t,jj_t,
                            self.mono_disps[None],
                            scales[None],shifts[None],
                            self.valid_depth_mask_small[None], ignore_frames,
                            lm,ep,alpha=0.01
                        )
                        scales = wqs[0,:,0]
                        shifts = wqs[0,:,1]
                        success = True                    

                self.depth_scale = scales
                self.depth_shift = shifts

                self.disps = disps.squeeze(0)
                self.poses = poses.vec().squeeze(0)

                self.disps.clamp_(min=1e-5)
                return success

            else:
                raise NotImplementedError

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, motion_only=False, opt_type="pose_depth", depth_cov=False):
        if self.BA_type == "DSPO":
            success = self.dspo(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only, opt_type, depth_cov)
            if not success:
                self.dspo(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,"pose_depth", depth_cov)
        elif self.BA_type == "DBA":
            self.dspo(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,"pose_depth", depth_cov)
        else:
            raise NotImplementedError


    def get_depth_scale_and_shift(self,index, mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
        '''
        index: int
        mono_depth: [B,H,W]
        est_depth: [B,H,W]
        weights: [B,H,W]
        '''
        scale,shift,_ = align_scale_and_shift(mono_depth,est_depth,weights)
        self.depth_scale[index] = scale
        self.depth_shift[index] = shift
        return [self.depth_scale[index], self.depth_shift[index]]

    def get_pose(self,index,device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self,index,device):
        with self.get_lock():
            est_disp = self.disps_up[index].clone().to(device)  # [h, w]
            est_depth = 1.0 / (est_disp)
            depth_mask = self.valid_depth_mask[index].clone().to(device)
            c2w = self.get_pose(index,device)
        return est_depth, depth_mask, c2w

    def get_depth_cov(self,index,device):
        with self.get_lock():
            depth_cov = self.depths_cov_up[index].clone().to(device)  # [h, w]
        return depth_cov

    @torch.no_grad()
    def update_valid_depth_mask(self,up=True):
        '''
        For each pixel, check whether the estimated depth value is valid or not 
        by the two-view consistency check, see eq.4 ~ eq.7 in the paper for details

        up (bool): if True, check on the orignial-scale depth map
                   if False, check on the downsampled depth map
        '''
        if up:
            with self.get_lock():
                dirty_index, = torch.where(self.dirty.clone())
            if len(dirty_index) == 0:
                return
        else:
            curr_idx = self.counter.value-1
            dirty_index = torch.arange(curr_idx+1).to(self.device)
        # convert poses to 4x4 matrix
        disps = torch.index_select(self.disps_up if up else self.disps, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * (self.down_scale if up else 1.0)
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up if up else self.disps, intrinsic, dirty_index, thresh)
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        if up:
            self.valid_depth_mask[dirty_index] = masks 
            self.dirty[dirty_index] = False
        else:
            self.valid_depth_mask_small[dirty_index] = masks 

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True

    def save_video(self,path:str):
        poses = []
        depths = []
        depth_covs = []
        timestamps = []
        valid_depth_masks = []
        for i in range(self.counter.value):
            depth, depth_mask, pose = self.get_depth_and_pose(i,'cpu')
            depth_cov = self.get_depth_cov(i, 'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            depths.append(depth)
            depth_covs.append(depth_cov)
            timestamps.append(timestamp)
            valid_depth_masks.append(depth_mask)
        poses = torch.stack(poses,dim=0).numpy()
        depths = torch.stack(depths,dim=0).numpy()
        depth_covs = torch.stack(depth_covs,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy() 
        valid_depth_masks = torch.stack(valid_depth_masks,dim=0).numpy()       
        np.savez(path,poses=poses,depths=depths,depth_covs=depth_covs,timestamps=timestamps,valid_depth_masks=valid_depth_masks)
        self.printer.print(f"Saved final depth video: {path}",FontColor.INFO)

