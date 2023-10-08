import numpy as np
import torch
from sklearn.model_selection import train_test_split
from solo_head2 import *
from backbone import *
from dataset import *
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tensorboardX import SummaryWriter
from scipy import ndimage
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import cm
import skimage.transform
import os.path
import tqdm
import shutil
import gc
from sklearn import metrics
gc.enable()
def plot_loss(train_losses, test_losses, name):
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(name)
    plt.savefig(name+'.png')
    plt.close()

if __name__ == '__main__':
    load_model = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("DEVICE" ,device)
    # file path and make a list

    imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '../data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '../data/hw3_mycocodata_bboxes_comp_zlib.npy'

    # set up output dir (for plotGT)

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)
    del paths

    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    #reproductivity
    torch.random.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = test_build_loader.loader()

    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image

    resnet50_fpn = resnet50_fpn.to(device)
    resnet50_fpn.eval()
    solo_head = solo_head.to(device)

    num_epochs = 36
    optimizer = optim.SGD(solo_head.parameters(), lr=0.01/16*batch_size, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27,33], gamma=0.1)


    if load_model:
        checkpoint_path = 'train_check_point/checkpoint.pth'
        checkpoint = torch.load(checkpoint_path)
        solo_head.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        num_epochs = num_epochs - epoch


    train_cate_losses=[]
    train_mask_losses=[]
    train_total_losses=[]

    test_cate_losses=[]
    test_mask_losses=[]
    test_total_losses=[]

    os.makedirs("train_check_point", exist_ok=True)

    # tensorboard
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir="logs")

    min_test_loss = np.inf

    for epoch in range(num_epochs):
        ## fill in your training code
        solo_head.train()
        running_cate_loss = 0.0
        running_mask_loss=0.0
        running_total_loss=0.0
        pbar = tqdm.tqdm(enumerate(train_loader, 0))
        for iter, data in pbar:
            img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]

            img = img.to(device)      
            # label_list = [torch.tensor(x).to(device) for x in label_list]
            # mask_list = [torch.tensor(x).to(device) for x in mask_list]
            # bbox_list = [torch.tensor(x).to(device) for x in bbox_list]
            label_list = [x.to(device) for x in label_list]
            mask_list = [x.to(device) for x in mask_list]
            bbox_list = [x.to(device) for x in bbox_list]

            with torch.no_grad():
                backout = resnet50_fpn(img)
            del img
            fpn_feat_list = list(backout.values())
            optimizer.zero_grad()
            cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False) 
            del fpn_feat_list
            ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                            bbox_list,
                                                                            label_list,
                                                                            mask_list)  
            cate_loss, mask_loss, total_loss=solo_head.loss(cate_pred_list,ins_pred_list,ins_gts_list,ins_ind_gts_list,cate_gts_list)  #batch loss
            del label_list, mask_list, bbox_list
            del ins_gts_list, ins_ind_gts_list, cate_gts_list, cate_pred_list, ins_pred_list      
            total_loss.backward()
            optimizer.step()
            running_cate_loss += cate_loss.item()
            running_mask_loss += mask_loss.item()
            running_total_loss += total_loss.item()
            s = ('%10s' + '%10.4f' * 3) % ('iter: %d/%d| ' % (iter + 1,len(train_loader)), cate_loss.item(), mask_loss.item(), total_loss.item())
            pbar.set_description(s)
            if np.isnan(running_total_loss):
                raise RuntimeError("[ERROR] NaN encountered at iter: {}".format(iter))

      
        epoch_cate_loss = running_cate_loss / len(train_loader)
        epoch_mask_loss = running_mask_loss / len(train_loader)
        epoch_total_loss = running_total_loss / len(train_loader)
        # write to summary writer
        writer.add_scalar('Loss/train/log_cate_loss', epoch_cate_loss, epoch)
        writer.add_scalar('Loss/train/log_mask_loss', epoch_mask_loss, epoch)
        writer.add_scalar('Loss/train/log_total_loss', epoch_total_loss, epoch)
        print('\nEpoch:{} Avg. train loss: {:.4f}\n'.format(epoch + 1, epoch_total_loss))
        # save to list
        train_cate_losses.append(epoch_cate_loss)
        train_mask_losses.append(epoch_mask_loss)
        train_total_losses.append(epoch_total_loss)

        running_cate_loss = 0.0
        running_mask_loss=0.0
        running_total_loss=0.0  

        solo_head.eval()
        test_running_cate_loss = 0.0    
        test_running_mask_loss=0.0
        test_running_total_loss=0.0
        
        
        with torch.no_grad():
            vbar = tqdm.tqdm(enumerate(test_loader, 0))
            for iter, data in vbar:   
                img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
                img = img.to(device)  
                label_list = [x.to(device) for x in label_list]
                mask_list = [x.to(device) for x in mask_list]
                bbox_list = [x.to(device) for x in bbox_list]

                backout = resnet50_fpn(img)
                del img
                fpn_feat_list = list(backout.values())
                cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False) 
                del fpn_feat_list
                ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                        bbox_list,
                                                                        label_list,
                                                                        mask_list)

                cate_loss, mask_loss, total_loss=solo_head.loss(cate_pred_list,ins_pred_list,ins_gts_list,ins_ind_gts_list,cate_gts_list)
                label_list, mask_list, bbox_list
                del ins_gts_list, ins_ind_gts_list, cate_gts_list, cate_pred_list, ins_pred_list
                test_running_cate_loss += cate_loss.item()
                test_running_mask_loss += mask_loss.item()
                test_running_total_loss += total_loss.item()
                s = ('%10s' + '%10.4f' * 3) % ('iter: %d/%d | ' % (iter + 1,len(test_loader)), cate_loss.item(), mask_loss.item(), total_loss.item())
                vbar.set_description(s)
                
            epoch_cate_loss = test_running_cate_loss / len(test_loader)
            epoch_mask_loss = test_running_mask_loss / len(test_loader)
            epoch_total_loss = test_running_total_loss / len(test_loader)

            if min_test_loss > epoch_total_loss:
                min_test_loss = epoch_total_loss
                path = './train_check_point/solo_epoch_'+str(epoch)+'_best'
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': solo_head.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, path)
                
            print('\nEpoch:{} Avg. test loss: {:.4f}\n'.format(epoch + 1, epoch_total_loss))
            # write to summary writer
            writer.add_scalar('Loss/test/cate_loss', epoch_cate_loss, epoch)
            writer.add_scalar('Loss/test/mask_loss', epoch_mask_loss, epoch)
            writer.add_scalar('Loss/test/total_loss', epoch_total_loss, epoch)
            # save to list
            test_cate_losses.append(epoch_cate_loss)
            test_mask_losses.append(epoch_mask_loss)
            test_total_losses.append(epoch_total_loss)
        
        scheduler.step()

    writer.close()

    plot_loss(train_cate_losses, test_cate_losses, 'cate_loss')
    plot_loss(train_mask_losses, test_mask_losses, 'mask_loss')
    plot_loss(train_total_losses, test_total_losses, 'total_loss')

