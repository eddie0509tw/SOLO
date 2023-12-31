import numpy as np
from solo_head import *
from backbone import *
from dataset import *
import torch.utils.data
import gc
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
gc.enable()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# file path and make a list
imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = "../data/hw3_mycocodata_labels_comp_zlib.npy"
bboxes_path = "../data/hw3_mycocodata_bboxes_comp_zlib.npy"

batch_size = 4


# set up output dir (for plotGT)
paths = [imgs_path, masks_path, labels_path, bboxes_path]
# load the data into data.Dataset
dataset = BuildDataset(paths)

full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size

torch.random.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

resnet50_fpn = Resnet50Backbone()
solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
resnet50_fpn = resnet50_fpn.to(device)
resnet50_fpn.eval()             # set to eval mode

model_path = 'C:\\Users\\eddie\\CIS680\\hw3\\code\\train_check_point\\new_best.pth'
# Load the checkpoint from the specified model path
checkpoint = torch.load(model_path, map_location=device)  # map_location ensures the model is loaded to the correct device

# Load the model state dict from the checkpoint
solo_head.load_state_dict(checkpoint['model_state_dict'])

solo_head = solo_head.to(device)
solo_head.eval()

mask_color_list = ["jet", "ocean", "Spectral"]
os.makedirs("infer_result", exist_ok=True)
with torch.no_grad():
    # vbar = tqdm(enumerate(test_loader, 0))
    for iter, data in enumerate(test_loader, 0):   
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        img = img.to(device)  
        label_list = [x.to(device) for x in label_list]
        mask_list = [x.to(device) for x in mask_list]
        bbox_list = [x.to(device) for x in bbox_list]

        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=True) 
        # ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
        #                                                                 bbox_list,
        #                                                                 label_list,
        #                                                                 mask_list)
        #print(np.any(ins_pred_list[0].cpu().numpy()))
        NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = solo_head.PostProcess(ins_pred_list, cate_pred_list, (img.shape[2], img.shape[3]))
        if iter<=5:
            solo_head.PlotInfer(NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list,
                                mask_color_list, img, iter)


