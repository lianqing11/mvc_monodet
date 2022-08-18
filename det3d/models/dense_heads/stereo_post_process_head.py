from turtle import forward
from det3d.core.bbox.util import points_img2cam, projected_2d_box, projected_gravity_center
import torch
import numpy as np
import torch.nn.functional as F
from mmdet3d.core.bbox.structures import Box3DMode

import torch.nn as nn

from mmdet.models import HEADS


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import numpy as np
import math as m
import copy

def get_uv(input_size, template, cam2img ):
    '''
    image shape (h, w)
    '''
    f_h, f_w = input_size[-2:]

    u = torch.arange(f_w).to(template.device)   # 1 x f_w x 1
    v = torch.arange(f_h).to(template.device)
    uv = torch.stack(torch.meshgrid(v, u), dim=2)
    uv = torch.flip(uv, dims=[2]).float()

    # absolute_uv = uv
    norm_uv = uv.clone()
    norm_uv[..., 0] = (norm_uv[..., 0] - cam2img[0,2].reshape(-1,1,1)) / cam2img[0,0].reshape(-1,1,1)
    norm_uv[..., 1] = (norm_uv[..., 1] - cam2img[1,2].reshape(-1,1,1)) / cam2img[0,0].reshape(-1,1,1)
    return uv,norm_uv

@HEADS.register_module()
class StereoPostProcessModule(nn.Module):
    def __init__(self,
                valid_depth_threshold=40,
                conf_threshold=0.1,
                max_size=28,
                depth_range = 5,
                cam2img_use_depth=True,
                grid_num=50):
        super().__init__()
        self.valid_depth_threshold = valid_depth_threshold
        self.conf_threshold = conf_threshold
        self.max_size = max_size
        grid_middle = depth_range / 2.

        depth_enum = torch.range(0, grid_num) * \
            (depth_range / grid_num) - grid_middle
        self.register_buffer("depth_enum", depth_enum)
        self.cam2img_use_depth = cam2img_use_depth

    def get_corresponding_seg_mask(self,
                                   seg_mask,
                                   bbox_cam,
                                   img_metas):
        seg_mask = seg_mask.float().sum(dim=1, keepdim=True)
        seg_mask = seg_mask.clamp(max=1)
        return seg_mask
        # import pdb; pdb.set_trace()
        # # 1. get img shape

        # # 2. get left box2d

        # left_box2d = None

        # iou = overlap(left_box2d, )

    @torch.no_grad()
    def forward_test(self,
                  img,
                  target_img,
                  img_metas,
                  target_img_metas,
                  bbox_list,
                  gt_multiview_semantic_seg=None,
                  rescale=None):
        '''
        img: tensor with shape N, V, C, H, W (n=1; c=1)
        target_img: tensor with shape N, V, C, H, W (n=1; c=1)
        img_metas,
        target_img_metas,
        bbox_list: [(bbox, bbox_score, bbox_label)]:
            bbox: instance of bbox
            bbox_score: tensor with shape N
            bbox_labael: int tensor with shape N
        '''
        seg_mask = gt_multiview_semantic_seg[0]
        import pdb; pdb.set_trace()
        bbox, bbox_score, bbox_label = bbox_list[0]
        ori_bbox = bbox
        ori_bbox_score = bbox_score.clone()
        ori_bbox_label = bbox_label.clone()
        img_shape = img_metas[0]["pad_shape"][0][:2]
        source_index = torch.arange(len(bbox_score))

        # all_uvzs =
        # 1. convert back to the camera system
        lidar2cam = img.new_tensor(img_metas[0]["lidar2cam"][0])
        cam2img = img.new_tensor(img_metas[0]["cam2img"][0])

        target_cam2img = img.new_tensor(target_img_metas[0]["cam2img"][0])

        bbox_cam = bbox.convert_to(Box3DMode.CAM, rt_mat=lidar2cam)
        ori_bbox_cam = copy.deepcopy(bbox_cam)
        uv, norm_uv = get_uv(img_shape, img.squeeze(1), cam2img)

        # 2. valid mask
        valid_mask = bbox_cam.tensor[:,2] < self.valid_depth_threshold
        valid_mask = valid_mask & (bbox_score > self.conf_threshold)



        # valid_mask.fill_(True)

        if valid_mask.sum() == 0:
            return bbox_list
        bbox_cam.tensor = bbox_cam.tensor[valid_mask]
        source_index = source_index[valid_mask]

        seg_mask = self.get_corresponding_seg_mask(seg_mask, bbox_cam, img_metas)

        bbox_2d = projected_2d_box(bbox_cam, rt_mat=cam2img, img_shape=img_shape)

        patch_width, patch_height, left, right, top, down, max_pixels = \
            self.get_patch_width_height(bbox_2d,
                                        self.max_size,
                                        img_shape, offset=0.05)

        all_uvzs = img.new_zeros((
                            valid_mask.sum(),
                            self.depth_enum.shape[0],
                            max_pixels, 3))



        # TODO fix the valid mask
        all_valid_mask = torch.zeros_like(all_uvzs[...,0]).bool()

        center_loc = bbox_cam.gravity_center.clone()

        proj_center = projected_gravity_center(bbox_cam, rt_mat=cam2img)
        depth_enum = center_loc[:,2].unsqueeze(1).expand(-1, self.depth_enum.shape[0]) # N, K, 1
        depth_enum = depth_enum + self.depth_enum # N, K, 1

        proj_center_enum = proj_center[:, None, :].expand(-1, depth_enum.shape[1], -1) # N, K, 2
        loc_enum = points_img2cam(proj_center_enum.reshape(-1, 2),
                                  depth_enum.to(proj_center.device).reshape(-1,1), cam2img,
                                  use_depth=self.cam2img_use_depth)
        # loc_enum = loc_enum.reshape(depth_enum.shape[0], depth_enum.shape[1], 3)
        loc_enum = loc_enum.reshape(-1, depth_enum.shape[1], 3)


        loc_enum[:,:,1] += bbox_cam.dims[:,1][:, None]/2 # to kitti center

        loc_enum_pad = loc_enum.clone()
        loc_enum_pad[...,0] += cam2img[0,3] / cam2img[0,0]
        loc_enum_pad[...,1] += cam2img[1,3] / cam2img[1,1]
        for idx in range(valid_mask.sum()):
            # if valid_mask[idx] == False:
                # new_bbox.append(bbox_cam.tensor[:,idx])
                # continue
            for jdx in range(depth_enum.shape[1]):
                loc_pad = loc_enum_pad[idx, jdx].reshape(-1)
                # loc_pad[0] += cam2img[0, 3] / cam2img[0, 0]
                # loc_pad[1] += cam2img[1, 3] / cam2img[1, 1]
                stereo_rcnn_box3d = StereoRCNNBox3d(torch.cat([
                    loc_pad.reshape(-1), bbox_cam.dims[idx],
                    bbox_cam.tensor[idx,6:7]]))

                uv_idx = uv[top[idx] : down[idx] : patch_height[idx], \
                                        left[idx] : right[idx] : patch_width[idx]]
                norm_uv_idx = norm_uv[top[idx] : down[idx] : patch_height[idx], \
                                        left[idx] : right[idx] : patch_width[idx]]
                if norm_uv_idx.shape[0] == 0 or norm_uv_idx.shape[1] == 0:
                    continue
                valid_insec = stereo_rcnn_box3d.BoxRayInsec(norm_uv_idx)
                valid_uvz = torch.cat(
                        (uv_idx[..., 0].reshape(-1,1), uv_idx[..., 1].reshape(-1,1),
                                        valid_insec[..., 2].reshape(-1,1)), 1)
                all_uvzs[idx, jdx, :valid_uvz.shape[0], :] = valid_uvz
                all_valid_mask[idx,jdx, :valid_uvz.shape[0]] = (valid_insec[...,3].reshape(-1) == 1)
        fb = cam2img[0, 3] - target_cam2img[0, 3]

        N, K = depth_enum.shape[:2]

        # seg_mask = None
        fb, img, target_img, _, _, _ = reshape_semi_eval_metrics(fb, depth_enum, img[:, 0],
                                                            target_img[:, 0], None, None, None)

        all_uvzs = all_uvzs.reshape(-1, max_pixels, 3)
        # reshap all uvz and check how to get corresponding grid
        all_valid_mask = all_valid_mask.reshape(-1, max_pixels)

        semi_loss, left_seg_mask, left_offset_uncertainty = \
                    self.get_corresponding_grid(
                            all_uvzs, img_shape, fb, depth_enum, img, target_img,
                            all_valid_mask)
        if left_seg_mask is not None:
            left_seg_mask = left_seg_mask.squeeze(1).squeeze(1)
            all_valid_mask = all_valid_mask * left_seg_mask
            left_seg_mask = left_seg_mask.mean([-1])
        # all_valid_mask = all_valid_mask.permute(1,0,2)
        # semi_loss = semi_loss.permute(1,0,2)
        all_valid_mask = all_valid_mask.reshape(N, K, -1)
        semi_loss[all_valid_mask<0.1] = 0
        # if left_offset_uncertainty is not None:
        #     left_offset_uncertainty = 1 - torch.clamp(torch.exp(left_offset_uncertainty), min=0.01, max=1)
        #     semi_loss = semi_loss * left_offset_uncertainty.squeeze(1)
        #     all_valid_mask = all_valid_mask * left_offset_uncertainty.squeeze(1)
        semi_loss = semi_loss.sum([2]) / (all_valid_mask.sum([2]) + 1e-3).detach()
        semi_loss = semi_loss.reshape(-1, K)
        # left_seg_mask = left_seg_mask.reshape(-1,k)
        depth_enum = depth_enum.reshape(-1, K)
        min_depth_index = semi_loss.min(dim=1)[1]

        for idx in range(len(min_depth_index)):

            if min_depth_index[idx] == 0 or \
                    min_depth_index[idx] == semi_loss.shape[1]:
                continue
            min_depth = depth_enum[idx, min_depth_index[idx]]
            new_location = loc_enum[idx, min_depth_index[idx]]
            ori_bbox_cam.tensor[source_index[idx],:3] = new_location
            # if left_seg_mask[idx,min_depth_index[idx]] > 0.1:
                # new_location = loc_enum[idx, min_depth_index[idx]]
                # results[0][source_index[idx]][9:12] = new_location.detach()
        # min_depth = depth_enum[min_depth_index]
        # convert the ground truth
        ori_bbox = ori_bbox_cam.convert_to(Box3DMode.LIDAR, rt_mat = torch.inverse(lidar2cam))

        # TODO only convert part of the label

        return [(ori_bbox, bbox_score, bbox_label)]

    def get_patch_width_height(self,
                               bbox_2d,
                               max_size,
                               img_size,
                               offset=0):

        patch_width = (bbox_2d[:,2] - bbox_2d[:,0])/max_size
        patch_height = (bbox_2d[:,3] - bbox_2d[:,1])/max_size
        patch_width = torch.clamp(patch_width.long(), min=1)
        patch_height = torch.clamp(patch_height.long(), min=1)
        box_height = bbox_2d[:, 3] - bbox_2d[:, 1]
        middle_point_y = (bbox_2d[:,1] + bbox_2d[:,3]) / 2.

        left = (bbox_2d[:,0].clone() + 0.5).clamp(min=0, max=img_size[1] - 1).long()

        right = (bbox_2d[:,2].clone() + 0.5).clamp(min=0, max=img_size[1] - 1).long()
        if offset > 0:
            left = left.clone() + patch_width * offset
            right = right.clone() - patch_width * offset
            left = left.clamp(min=0, max=img_size[1] - 1).long()
            right = right.clamp(min=0, max=img_size[1] - 1).long()
        top = (middle_point_y  + 0.5).clamp(min=0, max=img_size[0] - 1).long()
        down = (bbox_2d[:,3] - box_height * 0.1 + 0.5).clamp(min=0, max=img_size[0] - 1).long()
        # down = down - 1
        max_pixels = ((down - top) // patch_height + 1 ).long() * ((right - left) // patch_width +1).long()
        try:
            max_pixels = int(max_pixels.max().item()+0.5)
        except:
            import pdb; pdb.set_trace()
        return patch_width, patch_height, left, right, top, down, max_pixels


    def get_corresponding_grid(self, all_uvz, input_size, fb, depth, img,
                                target_img, all_valid_mask,
                                seg_mask=None,
                                cross_view_offset=None,
                                cross_view_offset_uncertainty=None):
        f_h, f_w = input_size[-2:]
        f_w = f_w - 1
        f_h = f_h - 1
        grid_left = all_uvz.clone()[:,:,:2]
        max_pixels = grid_left.shape[1]
        grid_right = grid_left.clone()

        grid_left[:,:,0] = (grid_left[:,:,0] - f_w/2) / (f_w/2)
        grid_left[:,:,1] = (grid_left[:,:,1] - f_h/2) / (f_h/2)
        # grid_left = grid_left.reshape(n, k, max_pixels, 2)[:,0]
        NK, max_pixels = grid_left.shape[:2]
        K = len(self.depth_enum)
        N = int(NK / K)
        grid_left = grid_left.reshape(N, -1, max_pixels, 2)
        grid_left = grid_left[:,0]
        depth = depth.reshape(-1, 1).expand(-1, max_pixels)
        fb = fb.reshape(-1, 1).expand(-1, max_pixels)
        if cross_view_offset is not None:
            left_offset = F.grid_sample(cross_view_offset, grid_left.unsqueeze(1),
                                 padding_mode="border", align_corners=True)
            left_offset = torch.clamp(left_offset, min=self.cross_view_offset_threshold[0], max=self.cross_view_offset_threshold[1])
            if cross_view_offset_uncertainty is not None:
                left_offset_uncertainty = F.grid_sample(cross_view_offset_uncertainty, grid_left.unsqueeze(1),
                                     padding_mode="border", align_corners=True)
            else:
                left_offset_uncertainty = None
            delta_d = fb / (all_uvz[:,:,2] + depth + left_offset.squeeze(1).squeeze(1))
            depth_mask  = depth.unsqueeze(1).unsqueeze(1) \
                + left_offset + all_uvz[:,:,2].unsqueeze(1).unsqueeze(1)
            delta_d_nooffset = fb / (all_uvz[:,:,2] + depth)
        else:
            left_offset = None
            left_offset_uncertainty = None
            delta_d = fb / (all_uvz[:,:,2] + depth)
            depth_mask = depth.unsqueeze(1).unsqueeze(1) + all_uvz[:,:,2].unsqueeze(1).unsqueeze(1)
            delta_d_nooffset = delta_d.clone()
        grid_right[:,:,0] = (grid_right[:,:,0] - delta_d - f_w/2) / (f_w/2)
        grid_right[:,:,1] = (grid_right[:,:,1] - f_h/2) / (f_h/2)

        # grid_right_nooffset[:,:,0] = (grid_right_nooffset[:,:,0] - delta_d_nooffset - f_w/2) / (f_w/2)
        # grid_right_nooffset[:,:,1] = (grid_right_nooffset[:,:,1] - f_h/2) / (f_h/2)

        # img.requires_g     rad=True
        left_patch = F.grid_sample(img, grid_left.unsqueeze(1),
                                padding_mode="border", align_corners=True)
        # target_img.requires_grad=True
        right_patch = F.grid_sample(target_img, grid_right.unsqueeze(1),
                                padding_mode="border", align_corners=True)

        if seg_mask is not None:
            left_seg_mask = F.grid_sample(seg_mask, grid_left.unsqueeze(1),
                                padding_mode="border", align_corners=True)
        else:
            left_seg_mask = None

        # filter the depth that is smaller than 0 (handle the trancation case)
        if left_seg_mask is not None:
            left_seg_mask[depth_mask < 0] = 0
            all_valid_mask[depth_mask.squeeze(1).squeeze(1) < 0] = 0

        right_patch = right_patch.reshape(N, K, 3, -1)
        left_patch = left_patch.reshape(N, 1, 3, -1)
        semi_loss = (left_patch - right_patch).abs().mean(2)
        return semi_loss, left_seg_mask, left_offset_uncertainty




class StereoRCNNBox3d(nn.Module):
    def __init__(self, poses):
        super(StereoRCNNBox3d, self).__init__()
        self.T_c_o = poses[0:3]
        self.size = poses[3:6] * 0.9
        self.R_c_o = torch.FloatTensor([[ m.cos(poses[6]), 0 ,m.sin(poses[6])],
                                        [ 0,         1 ,     0],
                                        [-m.sin(poses[6]), 0 ,m.cos(poses[6])]]).type_as(self.T_c_o)

        self.P_o = poses.new(8,3).zero_()
        self.P_o[0,0],self.P_o[0,1], self.P_o[0,2] = -self.size[0]/2, 0, -self.size[2]/2.0
        self.P_o[1,0],self.P_o[1,1], self.P_o[1,2] = -self.size[0]/2, 0, self.size[2]/2.0
        self.P_o[2,0],self.P_o[2,1], self.P_o[2,2] = self.size[0]/2, 0, self.size[2]/2.0         #max
        self.P_o[3,0],self.P_o[3,1], self.P_o[3,2] = self.size[0]/2, 0, -self.size[2]/2.0

        self.P_o[4,0],self.P_o[4,1], self.P_o[4,2] = -self.size[0]/2, -self.size[1], -self.size[2]/2.0 # min
        self.P_o[5,0],self.P_o[5,1], self.P_o[5,2] = -self.size[0]/2, -self.size[1], self.size[2]/2.0
        self.P_o[6,0],self.P_o[6,1], self.P_o[6,2] = self.size[0]/2, -self.size[1], self.size[2]/2.0
        self.P_o[7,0],self.P_o[7,1], self.P_o[7,2] = self.size[0]/2, -self.size[1], -self.size[2]/2.0

        P_c = poses.new(8,3).zero_()
        for i in range(8):
            P_c[i] = torch.mm(self.R_c_o, self.P_o[i].unsqueeze(1)).squeeze(1) + self.T_c_o

        def creatPlane(p1, p2, p3):
            arrow1 = p2 - p1
            arrow2 = p3 - p1
            normal = torch.cross(arrow1, arrow2)
            plane = p1.new((4)).zero_()
            plane[0] = normal[0]
            plane[1] = normal[1]
            plane[2] = normal[2]
            plane[3] = -normal[0] * p1[0] - normal[1] * p1[1] - normal[2] * p1[2]
            return plane

        self.planes_c = poses.new(6,4).zero_()
        self.planes_c[0] = creatPlane(P_c[0], P_c[3], P_c[4])  #front 0
        self.planes_c[1] = creatPlane(P_c[2], P_c[3], P_c[6])  #right 1
        self.planes_c[2] = creatPlane(P_c[1], P_c[2], P_c[5])  #back 2
        self.planes_c[3] = creatPlane(P_c[0], P_c[1], P_c[4])  #left 3
        self.planes_c[4] = creatPlane(P_c[0], P_c[1], P_c[2])  #botom 4
        self.planes_c[5] = creatPlane(P_c[4], P_c[5], P_c[6])  #top 5

        # compute the nearest vertex
        self.nearest_dist = 100000000
        for i in range(P_c.size()[0]):
            if torch.norm(P_c[i]) < self.nearest_dist:
                self.nearest_dist = torch.norm(P_c[i])
                self.nearest_vertex = i  # find the nearest vertex with camera canter

    def mask_out_box(self, valid_insec, insection_c):
        DOUBLE_EPS = 0.01
        R_c_o_t = self.R_c_o.permute(1,0)
        insection_c = insection_c[:,:,0:3] - self.T_c_o
        insection_o = insection_c.new(insection_c.size()).zero_()
        insection_o[:,:,0] = R_c_o_t[0,0]*insection_c[:,:,0] + R_c_o_t[0,1]*insection_c[:,:,1] + R_c_o_t[0,2]*insection_c[:,:,2]
        insection_o[:,:,1] = R_c_o_t[1,0]*insection_c[:,:,0] + R_c_o_t[1,1]*insection_c[:,:,1] + R_c_o_t[1,2]*insection_c[:,:,2]
        insection_o[:,:,2] = R_c_o_t[2,0]*insection_c[:,:,0] + R_c_o_t[2,1]*insection_c[:,:,1] + R_c_o_t[2,2]*insection_c[:,:,2]

        mask = ((insection_o[:,:,0] >= self.P_o[4,0] - DOUBLE_EPS) &\
                (insection_o[:,:,1] >= self.P_o[4,1] - DOUBLE_EPS) &\
                (insection_o[:,:,2] >= self.P_o[4,2] - DOUBLE_EPS) &\
                (insection_o[:,:,0] <= self.P_o[2,0] + DOUBLE_EPS) &\
                (insection_o[:,:,1] <= self.P_o[2,1] + DOUBLE_EPS) &\
                (insection_o[:,:,2] <= self.P_o[2,2] + DOUBLE_EPS)).type_as(insection_o)
        #print('valid_insec',valid_insec[valid_insec[:,:,3]==0])
        #print('insection_o',insection_o[valid_insec[:,:,3]==0])
        valid_insec[:,:,0][valid_insec[:,:,3]==0] = insection_c[:,:,0][valid_insec[:,:,3]==0]
        valid_insec[:,:,1][valid_insec[:,:,3]==0] = insection_c[:,:,1][valid_insec[:,:,3]==0]
        valid_insec[:,:,2][valid_insec[:,:,3]==0] = insection_c[:,:,2][valid_insec[:,:,3]==0]
        valid_insec[:,:,3][valid_insec[:,:,3]==0] = mask[valid_insec[:,:,3]==0]

        return valid_insec

    def BoxRayInsec(self, pt2):
        plane_group = torch.IntTensor([[0, 3, 4],
                                [2, 3, 4],
                                [1, 2, 4],
                                [0, 1, 4],

                                [0, 3, 5],
                                [2, 3, 5],
                                [1, 2, 5],
                                [0, 1, 5]])
        homo_pt3 = torch.cat((pt2, torch.ones_like(pt2[:,:,0]).unsqueeze(2)),2)
        valid_insec = homo_pt3.new(homo_pt3.size()[0],homo_pt3.size()[1], 4).zero_() # x_o, y_o, z_o, mask
        for i in range(3):
            plane = self.planes_c[plane_group[self.nearest_vertex,i]]
            # get insection, t is a scalar
            t = homo_pt3[:,:,0]*plane[0] +  homo_pt3[:,:,1]*plane[1] + homo_pt3[:,:,2]*plane[2]
            t = -t.reciprocal()*plane[3]
            insection_c = homo_pt3 * t.unsqueeze(2)
            valid_insec = self.mask_out_box(valid_insec, insection_c)
        return valid_insec



def parallel_ray_searching(bbox, points):
    # bbox with shape of NK, 7 (box instance)
    # points with shape of NK, max_pixels, 3
    # 1. corners
    order_mapping = [3, 2, 6, 7, 0, 1, 5, 4]
    corners = bbox.corners # reorder to stereo RCNN type
    corners = corners[..., order_mapping]

    # 2. local corners
    local_corners = get_local_corners(bbox)

    def createPlane(p1, p2, p3):
        arrow1 = p2 - p1
        arrow2 = p3 - p1
        normal = torch.cross(arrow1, arrow2)
        plane = p1.new_zeros(len(p1), 4)
        plane[:,:3] = normal
        plane[3] = -normal[:, 0] * p1[:, 0] - \
                    normal[:,1] * p1[:,1] - \
                    normal[:,2] * p1[:,2]
        return plane
    plane_mapping = [
                (0, 3, 4),
                (2, 3, 6),
                (1, 2, 5),
                (0, 1, 4),
                (0, 1, 2),
                (4, 5, 6),]
    planes = []
    for plane_idx in plane_mapping:
        planes.append(
            createPlane(
                corners[plane_idx[0]],
                corners[plane_idx[1]],
                corners[plane_idx[2]],))
    min_vertex = torch.norm(corners, dim=-1).argmin(1)

    plane_group = [[0, 3, 4],
                   [2, 3, 4],
                   [1, 2, 4],
                   [0, 1, 4],
                   [0, 3, 5],
                   [2, 3, 5],
                   [1, 2, 5],
                   [0, 1, 5]]
    
    homo_points = torch.cat(
        [[points], torch.ones_like(points[...,0])], dim=-1)
    valid_insec = torch.zeros_like(homo_points)
    valid_insec = torch.cat(
        [valid_insec, torch.zeros_like(points[..., 0])], dim=-1)
    
    for i in range(3):
        import pdb; pdb.set_trace()
        # get the plan
        t = homo_points[..., 0] * plane[..., 0] + \
            homo_points[..., 1] * plane[..., 1] + \
            homo_points[..., 2] * plane[..., 2]
        insection_c = homo_points * t.unsqueeze(2)
        valid_insec = mask_out_bbox(valid_insec, insection_c)
    return valid_insec

def get_local_corners(bbox):
    dims = bbox.dims
    corners_norm = torch.from_numpy(
        np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
            device=dims.device, dtype=dims.dtype)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # corners_norm = corners_norm - dims.new_tensor([0.5, 1, 0.5])
    corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    return corners

def mask_out_bbox(valid_insec,
                  insection_c, 
                  loc_center, 
                  local_corners):
    '''
    Args:
        valid_insec:
        insection_c:
        loc_center:
        local_corners:
    '''

    

    # planes = [createPlane]
    # 3. generate plane

    # 4. find the minimum vertex
    pass


def get_patch_width_height(box2d, max_size, min_grid, img_size, offset=0):

    patch_width = (box2d[:,2] - box2d[:,0])/max_size
    patch_height = (box2d[:,3] - box2d[:,1])/max_size
    patch_width = torch.clamp(patch_width.long(), min=1)
    patch_height = torch.clamp(patch_height.long(), min=1)
    box_height = box2d[:, 3] - box2d[:, 1]
    middle_point_y = (box2d[:,1] + box2d[:,3]) / 2.

    left = (box2d[:,0].clone() + 0.5).clamp(min=0, max=img_size[1] - 1).long()

    right = (box2d[:,2].clone() + 0.5).clamp(min=0, max=img_size[1] - 1).long()
    if offset > 0:
        left = left.clone() + patch_width * offset
        right = right.clone() - patch_width * offset
        left = left.clamp(min=0, max=img_size[1] - 1).long()
        right = right.clamp(min=0, max=img_size[1] - 1).long()
    top = (middle_point_y  + 0.5).clamp(min=0, max=img_size[0] - 1).long()
    down = (box2d[:,3] - box_height * 0.1 + 0.5).clamp(min=0, max=img_size[0] - 1).long()
    # down = down - 1
    max_pixels = ((down - top) // patch_height + 1 ).long() * ((right - left) // patch_width +1).long()
    try:
        max_pixels = int(max_pixels.max().item()+0.5)
    except:
        import pdb; pdb.set_trace()
    return patch_width, patch_height, left, right, top, down, max_pixels




def reshape_semi_eval_metrics(fb,
                                depth_enum,
                                img,
                                target_img,
                                seg_mask=None,
                                cross_view_offset=None,
                                cross_view_offset_uncertainty=None):

    '''
        Input the coresponding metrics and reshape than that with -1, K
    '''
    n,k = depth_enum.shape[:2]
    if fb is not None:
        fb = fb.reshape(1,1).expand(n, k).reshape(-1)

    depth_enum = depth_enum.reshape(-1)
    c, h, w = img.shape[-3:]
    img = img.expand(n, -1, -1, -1)
    target_img = target_img.expand(n*k, -1, -1, -1)
    # images = images.unsqueeze(1).expand(-1,k, -1,-1,-1).reshape(n*k, c, h, w)

    if seg_mask is not None:
        seg_mask = seg_mask.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(n*k, 1, h, w)
    if cross_view_offset is not None:
        feature_h, feature_w = cross_view_offset.shape[-2:]
        cross_view_offset = \
            cross_view_offset.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(n*k, 1, feature_h, feature_w)
    if cross_view_offset_uncertainty is not None:
        cross_view_offset_uncertainty = \
            cross_view_offset_uncertainty.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(n*k, 1, feature_h, feature_w)

    return fb, img, target_img, seg_mask,\
                        cross_view_offset, cross_view_offset_uncertainty
