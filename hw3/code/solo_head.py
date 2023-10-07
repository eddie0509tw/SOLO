import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
from matplotlib import cm
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        #assert len(self.ins_out_list) == len(self.strides)
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
                self.cate_head.append(
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
                self.ins_head.append(
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
                self.cate_head.append(
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
                self.ins_head.append(
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
        for m in self.ins_head:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(con.weight, mean=0, std=0.01)

        for m in self.cate_head:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(con.weight, mean=0, std=0.01)


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
        cate_pred_list, ins_pred_list =  self.MultiApply(self.forward_single_level, new_fpn_list,
                                                       num_of_levels,
                                                       eval=eval, upsample_shape=quart_shape)
        # print(ins_pred_list[0].shape)
        # print(ins_pred_list[1].shape)
        # print(ins_pred_list[2].shape)
        # print(ins_pred_list[3].shape)
        # print(ins_pred_list[4].shape)
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

        # x_idx = torch.arange(0, fpn_feat.shape[-1], 1, device=fpn_feat.device)
        # y_idx = torch.arange(0, fpn_feat.shape[-2], 1, device=fpn_feat.device)
        # #   normalize x,y indices to 1 -1
        # x_min = x_idx.min().item()
        # x_max = x_idx.max().item()
        # x_idx = -1 + 2 * (x_idx - x_min) / (x_max - x_min)

        # y_min = y_idx.min().item()
        # y_max = y_idx.max().item()
        # y_idx = -1 + 2 * (y_idx - y_min) / (y_max - y_min)

        # y, x = torch.meshgrid(y_idx, x_idx)

        # y = y.expand([batch_size, 1, -1, -1])
        # x = x.expand([batch_size, 1, -1, -1])
        x_range = torch.linspace(-1, 1, fpn_feat.shape[-1], device=fpn_feat.device)
        y_range = torch.linspace(-1, 1, fpn_feat.shape[-2], device=fpn_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([fpn_feat.shape[0], 1, -1, -1])
        x = x.expand([fpn_feat.shape[0], 1, -1, -1])

        ins_pred = torch.cat([ins_pred, x, y], dim=1)

        for layer in self.ins_head:
            ins_pred = layer(ins_pred)
        # scale the feature map into 2H 2W
        ins_pred = F.interpolate(ins_pred, scale_factor=2,
                                  mode='bilinear', 
                                  align_corners=False, 
                                  recompute_scale_factor=True)
        
        ins_pred = self.solo_ins_out[idx](ins_pred)

        # cate_pred
        # self.seg_num_grids=[40, 36, 24, 16, 12]
        cate_pred = F.interpolate(cate_pred, 
                                  size = self.seg_num_grids[idx],
                                  mode='bilinear')
        for layer in self.cate_head:
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
        bz = len(ins_gts_list)
        fpn=len(ins_gts_list[0])
        # get all the active grid and concat them together -># list, len(fpn), float32 (# gt_grid_across_batch, 2H_feat, 2W_feat)
        ins_gts = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]   #
                for ins_labels_level_img, ins_ind_labels_level_img in 
                zip(ins_labels_level, ins_ind_labels_level)], 0)
            for ins_labels_level, ins_ind_labels_level in 
            zip(zip(*ins_gts_list), zip(*ins_ind_gts_list))]  
        # We get those active from prediction grid only also -># list, len(fpn), float32 (# gt_grid_across_batch, 2H_feat, 2W_feat)        
        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                for ins_preds_level_img, ins_ind_labels_level_img in 
                zip(ins_preds_level, ins_ind_labels_level)], 0)
            for ins_preds_level, ins_ind_labels_level in 
            zip(ins_pred_list, zip(*ins_ind_gts_list))]     

        #DiceLoss
        dice_loss = torch.zeros(1,device=ins_preds[0].device,dtype=torch.float32)
        n_pos = 0
        for input_level,target_level in zip(ins_preds, ins_gts):
            if input_level.size()[0] == 0:
                continue

            dice_loss_list = self.MultiApply(self.DiceLoss,torch.sigmoid(input_level), target_level)

            dice_loss += sum(dice_loss_list[0])
            n_pos += input_level.size()[0]

        mask_loss = dice_loss / n_pos


        #FocalLoss
        cate_gts = [
                    torch.cat([cate_labels_level_img.flatten() 
                                for cate_labels_level_img in cate_labels_level])
                    for cate_labels_level in zip(*cate_gts_list)
                    ]
        cate_gts = torch.cat(cate_gts)  #(7744,) torch {0,1,2,3}   int64

        cate_preds = [
                    cate_pred_level.permute(0,2,3,1).reshape(-1, self.cate_out_channels) 
                    for cate_pred_level in cate_pred_list
                    ]   #list, len()=5,each(bz*S*S,3) for each level       
        cate_preds = torch.cat(cate_preds, 0)    #(7744,3) torch   [0~1] float32

        cate_loss = self.FocalLoss(torch.sigmoid(cate_preds), cate_gts)

        total_loss=cate_loss+self.mask_loss_cfg["weight"]*mask_loss

        return cate_loss, mask_loss, total_loss

    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        pred_flat = mask_pred.contiguous().view(-1).float()
        gt_flat = mask_gt.contiguous().view(-1).float()
        intersection = torch.sum(pred_flat * gt_flat)
        pred_sum = torch.sum(pred_flat * pred_flat)
        gt_sum = torch.sum(gt_flat * gt_flat)
        dice_loss = 1 - (2 * intersection + 1e-9) / (pred_sum + gt_sum + 1e-9)
        #dice_loss = dice_loss.view(1)
        return dice_loss

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        alpha = self.cate_loss_cfg['alpha']
        gamma = self.cate_loss_cfg['gamma']
        N=cate_preds.shape[0]
        C=cate_preds.shape[1]+1

        one_hot = torch.zeros((N,C), device=cate_preds.device, dtype=torch.long)
        one_hot.scatter_(1, cate_gts.view(-1,1), 1)
        one_hot = one_hot[:,1:]

        n = N * (C-1)
        p_t = cate_preds * one_hot + (1 - cate_preds) * (1 - one_hot)
        alpha_t = alpha * one_hot + (1 - alpha) * (1 - one_hot)

        loss = - alpha_t * ((1 - p_t) ** gamma) * torch.log(p_t + 1e-9)

        return torch.sum(loss) / (n + 1e-9)
            

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
            self.targer_single_img,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            featmap_sizes=featmap_sizes)
        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

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
        # gt_labels_raw = torch.from_numpy(gt_labels_raw).to(device=device)
        gt_labels_raw = gt_labels_raw.to(device) if isinstance(gt_labels_raw, torch.Tensor) else torch.from_numpy(gt_labels_raw).to(device)
    
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):
            # initial the output tensor for each featmap
            ins_label = torch.zeros((num_grid**2, featmap_size[0], featmap_size[1]),dtype=torch.uint8, device=device)
            ins_ind_label = torch.zeros((num_grid**2,),dtype=torch.bool, device=device)
            cate_label = torch.zeros((num_grid, num_grid),dtype=torch.int64, device=device)

            indices = (gt_area >= lower_bound) & (gt_area <= upper_bound)
            indices = torch.nonzero(indices).squeeze(1)
            if indices.size()[0] == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            # select m objs that lies in this scale range
            gt_bboxes = gt_bboxes_raw[indices,...]
            gt_labels = gt_labels_raw[indices]
            gt_masks = gt_masks_raw[indices,...]

            w = (gt_bboxes[..., 2] - gt_bboxes[..., 0])   # the region we're going to consider
            h = (gt_bboxes[..., 3] - gt_bboxes[..., 1])

            scale_w,scale_h = w * self.epsilon, h * self.epsilon

            center_x, center_y = gt_bboxes[:, 0] + 0.5 * w, gt_bboxes[:, 1] + 0.5 * h

            output_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4) # 800, 1088

            coord_x = torch.floor((center_x / output_size[1]) * num_grid) # rescale relative to the output size and count which grid it belongs to
            coord_y = torch.floor((center_y / output_size[0]) * num_grid)

            # left, top, right, down
            top_box = torch.floor(((center_y - scale_w * 0.5) / output_size[0] * num_grid ))
            down_box = torch.floor(((center_y + scale_w * 0.5) / output_size[0] * num_grid) )
            left_box = torch.floor(((center_x - scale_h * 0.5) / output_size[1] * num_grid ))
            right_box = torch.floor(((center_x + scale_h * 0.5) / output_size[1] * num_grid))
            
            top_box = torch.where(top_box < 0, torch.zeros_like(top_box), top_box)
            down_box = torch.where(down_box > num_grid - 1, torch.ones_like(down_box) * (num_grid - 1), down_box)
            left_box = torch.where(left_box < 0, torch.zeros_like(left_box), left_box)
            right_box = torch.where(right_box > num_grid - 1, torch.ones_like(right_box) * (num_grid - 1), right_box)
            
            top = torch.where(top_box > (coord_y - 1), top_box, torch.ones_like(coord_y) * (coord_y - 1))
            down = torch.where(down_box < (coord_y + 1), down_box, torch.ones_like(coord_y) * (coord_y + 1))
            left = torch.where(left_box > (coord_x - 1), left_box, torch.ones_like(coord_x) * (coord_x - 1))
            right = torch.where(right_box < (coord_x + 1), right_box, torch.ones_like(coord_x) * (coord_x + 1))

            num_of_objs = gt_bboxes.shape[0]
            scale = 2 / stride

            for n in range(num_of_objs):
                t = top[n].long()
                d = down[n].long()
                l = left[n].long()
                r = right[n].long()
                cate_label[t:(d + 1), l:(r + 1)] = gt_labels[n]

                seg_mask = gt_masks[n].cpu().numpy()
                # print(np.any(seg_mask))
                h, w = seg_mask.shape[-2:]
                new_w, new_h = int(w * float(scale)+0.5), int(h * float(scale)+0.5)
                seg_mask = cv2.resize(seg_mask, (new_w, new_h),  
                                      interpolation=cv2.INTER_LINEAR)
                #seg_mask = skimage.transform.resize(seg_mask, (new_h, new_w))
                seg_mask = np.where(seg_mask > 0.0,1,0)
                # print(np.any(seg_mask))
                # print(np.any(seg_mask.astype(np.uint8)))
                seg_mask = torch.from_numpy(seg_mask).to(device=device)

                for i in range(t, d+1):
                    for j in range(l, r+1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True
            # print(ins_label)
            # print(torch.any(ins_label))
            # exit()
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
        bz = ins_pred_list[0].shape[0]# Get the batch size from the first element of the instance prediction list.
        NMS_sorted_scores_list = []
        NMS_sorted_cate_label_list = []
        NMS_sorted_ins_list = []

        # Get the number of Feature Pyramid Network (FPN) levels.
        N_fpn = len(ins_pred_list)
        # print(N_fpn)
        assert N_fpn == len(cate_pred_list)

        # Loop over each image in the batch.
        for img_i in range(bz):
            # print(ins_pred_list[0][img_i].shape)
            # print(ins_pred_list[1][img_i].shape)
            # print(ins_pred_list[2][img_i].shape)
            # print(ins_pred_list[3][img_i].shape)
            # print(ins_pred_list[4][img_i].shape)
            # re-arranged inputs
            # (all_level_S^2, ori_H/4, ori_W/4)
            # Concatenate instance predictions from all FPN levels for the current image.
            ins_pred_img = torch.cat([ins_pred_list[i][img_i] for i in range(N_fpn)], dim=0)

            tmp_list = []# Loop over each FPN level.
            for fpn_i in range(N_fpn):
                # Extract the category prediction for the current image and FPN level.
                cate_pred = cate_pred_list[fpn_i][img_i]        # (C-1, S, S)
                S_1, S_2, C = cate_pred.shape# Get the shape of the category prediction.
                # tmp_x = cate_pred.permute(C, S_1, S_2).view(C, S_1 * S_2)       # (C, S_1 * S_2)
                # tmp_list.append(tmp_x.permute(1, 0))
                print(cate_pred.shape)
                assert cate_pred.shape[1] == cate_pred.shape[0]
                tmp_x = cate_pred.view(S_1 * S_2,C)       
                tmp_list.append(tmp_x)
            # (all_level_S^2, C-1)
            cate_pred_img = torch.cat(tmp_list, dim=0)# Concatenate category predictions from all FPN levels for the current image.
            assert cate_pred_img.shape[1] == 3

            # Post-process the concatenated instance and category predictions for the current image.
            NMS_sorted_scores, NMS_sorted_cate_label, NMS_sorted_ins = self.PostProcessImg(ins_pred_img, cate_pred_img, ori_size)
            # Append the post-processed predictions for the current image to the output lists.
            NMS_sorted_scores_list.append(NMS_sorted_scores)
            NMS_sorted_cate_label_list.append(NMS_sorted_cate_label)
            NMS_sorted_ins_list.append(NMS_sorted_ins)

        return NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list


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
        score_max,label = torch.max(cate_pred_img, dim=1) # prediction confidence and label
        cate_indicator =  score_max > self.postprocess_cfg['cate_thresh'] # indicator of whether the prediction is confident enough
        if(cate_indicator.sum() == 0):# If none of the pixels surpass the confidence threshold, return default values.
          return torch.tensor([0]), torch.tensor([1]), torch.zeros((1,800,1088))

        indicator_map = ins_pred_img > self.postprocess_cfg['mask_thresh'] # Generate a binary indicator mask to determine which pixels in the instance prediction surpass the mask threshold.
        ins_pred_img = ins_pred_img * indicator_map #This effectively zeroes out values in the ins_pred_img that are below the threshold
        
        coeff = torch.sum(ins_pred_img, dim=(1,2))/torch.sum(indicator_map,dim=(1,2))# calculate the coefficient 
        scores = score_max * coeff # calculate the final  S_j in the slides
        
         # Replace any NaN scores with 0.
        nan_scores_idx = torch.isnan(scores)
        scores[nan_scores_idx] = 0

        # Sort the scores in descending order and select the top 'pre_NMS_num' indices.
        _, sorted_indice = torch.sort(scores, descending=True)
        sorted_indice = sorted_indice[0:self.postprocess_cfg['pre_NMS_num']]
        assert len(sorted_indice) == self.postprocess_cfg['pre_NMS_num']
        sorted_score = scores[sorted_indice]        # Note: should be of descending order
        sorted_label = label[sorted_indice]
        sorted_ins_bin = indicator_map[sorted_indice]       # hard binary mask
        sorted_ins = ins_pred_img[sorted_indice]

        # Apply MatrixNMS on the sorted binary masks and scores to suppress overlapping predictions.
        scores_nms = self.MatrixNMS(sorted_ins_bin, sorted_score)

        # Retain the top 'keep_instance' predictions after NMS based on their scores.
        # print("scores_nms.shape: {}".format(scores_nms.shape))
        # print("scores_nms.ndim: {}".format(scores_nms.ndim))
        _, max_indice = torch.sort(scores_nms, descending=True)
        max_indice = max_indice[0:self.postprocess_cfg['keep_instance']]
        NMS_sorted_scores = scores_nms[max_indice]
        # add back the background label
        NMS_sorted_cate_label = sorted_label[max_indice] + 1
        # resize to H_ori, W_ori
        # (C, H, W)
        resized_mask = torch.nn.functional.interpolate(sorted_ins[max_indice].unsqueeze(0), scale_factor=(4, 4))
        resized_mask = resized_mask.squeeze(0)
        NMS_sorted_ins = resized_mask

        # Filter out predictions with extremely low scores.
        high_prob_indice = NMS_sorted_scores > 0.0

        return NMS_sorted_scores[high_prob_indice], NMS_sorted_cate_label[high_prob_indice], NMS_sorted_ins[high_prob_indice]
    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        n = len(sorted_scores)
        sorted_masks = sorted_ins.reshape(n, -1)

        intersection = torch.mm(sorted_masks, sorted_masks.T)
        areas = sorted_masks.sum(dim=1).expand(n, n)
        union = areas + areas.T
        ious = (intersection / union).triu(diagonal=1)

        ious_cmin = ious.min(0)[0].expand(n, n).T
        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmin ** 2) / gauss_sigma)
        else:
            decay = (ious) / (ious_cmin)

        decay = decay.min(dim=0)[0]
        return sorted_scores * decay

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
        rgb_color_list = []
        for color_str in color_list:
            color_map = cm.ScalarMappable(cmap=color_str)
            rgb_value = np.array(color_map.to_rgba(0))[:3]
            rgb_color_list.append(rgb_value)

        ## This is an important visual check flag.
        for img_i in range(len(ins_gts_list)):
            img_single = img[img_i]         # (3,Ori_H, Ori_W) original color image
            for level_i in range(len(ins_gts_list[img_i])):
                ins_gts = ins_gts_list[img_i][level_i]      # (S^2, 2H_f, 2W_f)
                cate_gts = cate_gts_list[img_i][level_i]    # (S, S), {1,2,3}
                ins_ind_gts = ins_ind_gts_list[img_i][level_i]  # (S^2)
                # print(ins_gts.size())
                # print(torch.any(ins_gts))

                # synthesis the visualization for this level of FPN
                # img_vis = np.array(img_single.cpu().numpy())
                # ax.imshow(img_vis.transpose((1, 2, 0)))

                assert ins_gts.shape[1] % 2 == 0
                assert ins_gts.shape[2] % 2 == 0
                H_feat = int(ins_gts.shape[1] / 2)
                W_feat = int(ins_gts.shape[2] / 2)
                S = cate_gts.shape[0]

                # for all active channel, extract the mask and sum up
                mask_vis = np.zeros((2 * H_feat, 2 * W_feat, 3))        # (2*H_feat, 2*W_feat, 3)
                for flatten_tensor in torch.nonzero(ins_ind_gts, as_tuple=False):
                    flatten_idx = flatten_tensor.item()
                    grid_i = int(flatten_idx / S)
                    grid_j = flatten_idx % S
                    obj_label = cate_gts[grid_i, grid_j]
                    assert obj_label != 0.0

                    # assign color
                    rgb_color = rgb_color_list[obj_label - 1]       # (3,)
                    # add mask to visualization image
                    obj_mask = ins_gts[flatten_idx].cpu().numpy()   # (2*H_feat, 2*W_feat)
                    obj_mask_3 = np.stack([obj_mask, obj_mask, obj_mask], axis=2)  # (2*H_feat, 2*W_feat, 3)
                    mask_vis = mask_vis + obj_mask_3 * rgb_color


                # visualization
                mask_vis_resized = skimage.transform.resize(mask_vis, (img_single.shape[1], img_single.shape[2], 3))

                # base image to numpy array and perform transform
                img_vis = img_single.cpu().numpy().transpose((1, 2, 0))
                # use mask value if available, otherwise, use img value
                img_vis = mask_vis_resized + img_vis * (mask_vis_resized == 0)

                fig, ax = plt.subplots(1)
                ax.imshow(img_vis)
                #plt.show()
                # mask_np = mask_vis_resized
                # mask_np = np.ma.masked_where(mask_np == 0, mask_np)
                # ax.imshow(mask_np, alpha=0.5, interpolation='none')
                # save the file
                saving_id = 1
                saving_file = "../gt_plot/img_{}_fpn_{}.png".format(saving_id, level_i)
                while os.path.isfile(saving_file):
                    saving_id = saving_id + 1
                    saving_file = "../gt_plot/img_{}_fpn_{}.png".format(saving_id, level_i)
                fig.savefig(saving_file)



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

        # convert color_list to RGB ones
        rgb_color_list = []
        for color_str in color_list:
            color_map = cm.ScalarMappable(cmap=color_str)
            rgb_value = np.array(color_map.to_rgba(0))[:3]
            rgb_color_list.append(rgb_value)

        for img_i, data in enumerate(zip(NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list, img), 0):
            # score: (keep_instance,)
            # cate_label: (keep_instance,)
            # ins: (keep_instance, ori_H, ori_W)
            score, cate_label, ins, img_single = data
            img_vis = img_single.cpu().numpy().transpose((1, 2, 0))     # (H, W, 3)

            # save the original image
            fig, ax = plt.subplots(1)
            ax.imshow(img_vis)
            plt.show()
            os.makedirs("infer_result", exist_ok=True)
            saving_file = "infer_result/batch_{}_img_{}_ori.png".format(iter_ind, img_i)
            fig.savefig(saving_file)

            # overlap all instance's mask to mask_vis (with color)
            mask_vis = np.zeros_like(img_vis)               # (H, W, 3)
            for ins_id in range(len(score)):
                obj_label = cate_label[ins_id]
                ins_bin = (ins[ins_id] >= self.postprocess_cfg['ins_thresh']) * 1.0
                obj_mask = ins_bin.cpu().numpy()        # (H, W)
                print(np.any(obj_mask))
                # assign color
                # Note: the object label from prediction here includes background.
                rgb_color = rgb_color_list[obj_label - 1]  # (3,)
                # add mask to visualization image
                obj_mask_3 = np.stack([obj_mask, obj_mask, obj_mask], axis=2)  # (H, W, 3)
                mask_vis = mask_vis + obj_mask_3 * rgb_color

            # use mask value if available, otherwise, use img value
            img_vis = mask_vis + img_vis * (mask_vis == 0)

            # visualize
            fig, ax = plt.subplots(1)
            ax.imshow(img_vis)
            plt.show()

            # save the file
            os.makedirs("infer_result", exist_ok=True)
            saving_file = "infer_result/batch_{}_img_{}.png".format(iter_ind, img_i)
            fig.savefig(saving_file)


from backbone import *
if __name__ == '__main__':
    # file path and make a list
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device( "cpu")
    imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "../data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "../data/hw3_mycocodata_bboxes_comp_zlib.npy"
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
    batch_size = 4
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs("../gt_plot", exist_ok=True)
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target

        print(bbox_list)
        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        print("finsihed target single image")
        # visualize the ground truth
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)
        if iter > 5:
            break


