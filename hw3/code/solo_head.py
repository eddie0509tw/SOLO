import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial

class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                alpha=0.25,
                                weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        pass

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32
        cat_chn0 = self.in_channels + 2
        # initial the two branch head modulelist
        self.cate_head = nn.ModuleList()
        self.ins_head = nn.ModuleList()

        for i in range(self.stacked_convs):
            if i == 0:
                self.cate_convs.append(
                    nn.Sequential(
                        nn.Conv2d(
                                self.in_channels,
                                self.seg_feat_channels,
                                3,
                                stride=1,
                                padding=1,
                                bias= False),
                        nn.GroupNorm(num_channels=self.seg_feat_channels,
                                    num_groups=32),           
                        nn.ReLU(inplace=True)
                    )
                )
                self.ins_convs.append(
                    nn.Sequential(
                        nn.Conv2d(
                                cat_chn0,
                                self.seg_feat_channels,
                                3,
                                stride=1,
                                padding=1,
                                bias= False),
                        nn.GroupNorm(num_channels=self.seg_feat_channels,
                                    num_groups=32),           
                        nn.ReLU(inplace=True)
                    )
                )

            else:
                self.cate_convs.append(
                    nn.Sequential(
                        nn.Conv2d(
                                self.seg_feat_channels,
                                self.seg_feat_channels,
                                3,
                                stride=1,
                                padding=1,
                                bias= False),
                        nn.GroupNorm(num_channels=self.seg_feat_channels,
                                    num_groups=32),           
                        nn.ReLU(inplace=True)
                    )
                )
                self.ins_convs.append(
                    nn.Sequential(
                        nn.Conv2d(
                                self.seg_feat_channels,
                                self.seg_feat_channels,
                                3,
                                stride=1,
                                padding=1,
                                bias= False),
                        nn.GroupNorm(num_channels=self.seg_feat_channels,
                                    num_groups=32),           
                        nn.ReLU(inplace=True)
                    )
                )
        #end of for loop

        self.solo_ins_out = nn.ModuleList()
        for num_grid in self.seg_num_grids:
            self.solo_ins_out.append(
                nn.Conv2d(self.seg_feat_channels, num_grid**2, 1)
            )
        self.solo_cate_out = nn.Conv2d(self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

    # This function initialize weights for head network
    def _init_weights(self):
        ## TODO: initialize the weights
        for m in self.ins_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(m.weight, mean=0, std=0.01)

        for m in self.cate_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(m.weight, mean=0, std=0.01)


        bias_ = float(-np.log((1 - 0.01) / 0.01))
        for m in self.solo_ins_out: 
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, bias_)

        nn.init.normal_(self.solo_cate_out.weight, mean=0, std=0.01)
        if hasattr(self.solo_cate_out, 'bias') and self.solo_cate_out.bias is not None:
            nn.init.constant_(self.solo_cate_out.bias, bias_)



    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
        # TODO: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        assert len(new_fpn_list) == len(self.seg_num_grids)

        num_of_levels = list(range(len(self.seg_num_grids)))
        cate_pred_list, ins_pred_list =  self.MultiApply(self.forward_single, new_fpn_list,
                                                       num_of_levels,
                                                       eval=eval, upsample_shape=quart_shape)
        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        f1 = F.interpolate(fpn_feat_list[0], 
                            size=fpn_feat_list[1].shape[-2:],
                            mode='bilinear')
        f2 = fpn_feat_list[1]
        f3 = fpn_feat_list[2]
        f4 = fpn_feat_list[3]
        f5 = F.interpolate(fpn_feat_list[4], 
                           size=fpn_feat_list[3].shape[-2:], 
                           mode='bilinear')
        return f1,f2,f3,f4,f5


    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==False
            # cate_pred: (bz,C-1,S,S)
            # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        num_grid = self.seg_num_grids[idx]  # current level grid
        batch_size = fpn_feat.shape[0]
        # ins_pred

        x_idx = torch.arange(0, fpn_feat.shape[-1], 1, device=fpn_feat.device)
        y_idx = torch.arange(0, fpn_feat.shape[-2], 1, device=fpn_feat.device)
        #   normalize x,y indices to 1 -1
        x_min = x_idx.min().item()
        x_max = x_idx.max().item()
        x_idx = -1 + 2 * (x_idx - x_min) / (x_max - x_min)

        y_min = y_idx.min().item()
        y_max = y_idx.max().item()
        y_idx = -1 + 2 * (y_idx - y_min) / (y_max - y_min)

        y, x = torch.meshgrid(y_idx, x_idx)

        y = y.expand([batch_size, 1, -1, -1])
        x = x.expand([batch_size, 1, -1, -1])

        ins_pred = torch.cat([ins_pred, x, y], dim=1)

        for layer in self.ins_convs:
            ins_pred = layer(ins_pred)
        # scale the feature map into 2H 2W
        ins_feat = F.interpolate(ins_feat, scale_factor=2,
                                  mode='bilinear', 
                                  align_corners=False, 
                                  recompute_scale_factor=True)
        
        ins_pred = self.solo_ins_list[idx](ins_pred)

        # cate_pred
        # self.seg_num_grids=[40, 36, 24, 16, 12]
        cate_pred = F.interpolate(cate_pred, 
                                  size = self.seg_num_grids[idx],
                                  mode='bilinear')
        for layer in self.cate_convs:
            cate_pred = layer(cate_pred)

        cate_pred = self.solo_cate_out(cate_pred)

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO resize ins_pred
            ins_pred = F.interpolate(ins_pred.sigmoid(), size=upsample_shape, mode='bilinear')

            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1)

        # check flag
        if eval == False:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.
        pass



    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        pass

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        pass

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               gt_bbox_list,
               gt_label_list,
               gt_mask_list):
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output in training
        featmap_sizes = [featmap.size()[-2:] for featmap in ins_pred_list]
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            featmap_sizes=featmap_sizes)
        # check flag
        assert ins_gts_list[0][1].shape = (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape = (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape = (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def targer_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image

        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []


        # The area of sqrt(w*h) of each object for examine which range to fit
        gt_area = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):
            # initial the output tensor for each featmap
            ins_label = torch.zeros((num_grid**2, featmap_size[0]*2, featmap_size[1]*2),dtype=torch.uint8, device=device)
            ins_ind_label = torch.zeros((num_grid**2,),dtype=torch.bool, device=device)
            cate_label = torch.zeros((num_grid, num_grid),dtype=torch.int64, device=device)

            indices = (gt_area >= lower_bound) & (gt_area <= upper_bound)
            indices = torch.nonzero(indices)
            if indices.size()[0] == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            # select m objs that lies in this scale range
            gt_bboxes = gt_bboxes_raw[indices,...]
            gt_labels = gt_labels_raw[indices]
            gt_masks = gt_masks_raw[indices,...]

            scale_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.epsilon # the region we're going to consider
            scale_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.epsilon

            center_x, center_y = (gt_bboxes[:, 2] + gt_bboxes[:, 0]) / 2, (gt_bboxes[:, 3] + gt_bboxes[:, 1]) / 2

            output_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4) # 800, 1088

            coord_x = torch.floor(center_x / output_size[1]) * num_grid # rescale relative to the output size and count which grid it belongs to
            coord_y = torch.floor(center_y / output_size[0]) * num_grid

            # left, top, right, down
            top_box = torch.floor(((center_x - scale_w * 0.5) / output_size[0]) * num_grid)
            down_box = torch.floor(((center_x + scale_w * 0.5) / output_size[0]) * num_grid)
            left_box = torch.floor(((center_y - scale_h * 0.5) / output_size[1]) * num_grid)
            right_box = torch.floor(((center_y + scale_h * 0.5) / output_size[1]) * num_grid)
            top_box = torch.where(top_box < 0, torch.zeros_like(top_box), top_box)
            down_box = torch.where(down_box > num_grid - 1, torch.ones_like(down_box) * (num_grid - 1), down_box)
            left_box = torch.where(left_box < 0, torch.zeros_like(left_box), left_box)
            right_box = torch.where(right_box > num_grid - 1, torch.ones_like(right_box) * (num_grid - 1), right_box)

            
            top = torch.where(top_box > (coord_y - 1), top_box, torch.ones_like(coord_y - 1) * (coord_y - 1))
            down = torch.where(down_box < (coord_y + 1), down_box, torch.ones_like(coord_y + 1) * (coord_y + 1))
            left = torch.where(left_box > (coord_x - 1), left_box, torch.ones_like(coord_x - 1) * (coord_x - 1))
            right = torch.where(right_box < (coord_x + 1), right_box, torch.ones_like(coord_x + 1) * (coord_x + 1))

            num_of_objs = gt_bboxes.shape[0]
            scale = 2 / stride
            for i in range(num_of_objs):
                cate_label[top[i]:(down+1)[i], left:(right+1)[i]] = gt_labels[i]

                seg_mask = gt_masks[i]
                h, w = seg_mask[:2]
                new_w, new_h = int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)
                seg_mask = cv2.resize(seg_mask, (new_w, new_h),  
                                      interpolation=cv2.INTER_LINEAR)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
                
        # check flag
        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list

    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size):

        ## TODO: finish PostProcess
        pass


    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size):

        ## TODO: PostProcess on single image.
        pass

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        pass

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img):
        ## TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        ## This is an important visual check flag.
        pass

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        pass

from backbone import *
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target


        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)

