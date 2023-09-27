## Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # TODO: load dataset, make mask list
        self.imgs_path = path[0]
        self.masks_path = path[1]
        self.labels_path = path[2]
        self.bboxes_path = path[3]

        # self.imgs_data = self.load_h5py(self.imgs_path)
        # self.masks_data = self.load_h5py(self.masks_path)
        self.labels_data = self.load_npy(self.labels_path)
        self.bboxes_data = self.load_npy(self.bboxes_path)
        # self.resize = transforms.Resize((800, 1066))
        # self.totensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # self.pad = transforms.Pad((11,0),fill=0)
        self.transform_img = transforms.Compose([transforms.Resize((800, 1066)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                             transforms.Pad((11,0),fill=0)])
        self.transform_mask = transforms.Compose([transforms.Resize((800, 1066)),
                                             transforms.ToTensor(),
                                             transforms.Pad((11,0),fill=0)])
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
        img = self.load_single_h5py(self.imgs_path, index)
        label = self.labels_data[index]
        mask = self.load_multi_h5py(self.masks_path, index,len(label))
        bbox = self.bboxes_data[index]
        # print(img.shape)
        # print(mask.shape)
        # print(len(label))
        # print(bbox)

        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)
        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox
    
    def __len__(self):
        return len(self.labels_data)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: (n_box, 300, 400)
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        # img = cv2.resize(img, (3,800, 1066))
        # mask = cv2.resize(mask, (800, 1066))
        # img = img/255.0
        # print(img.shape)
        # img = (img - np.array([0.485, 0.456, 0.406]).reshape(3,1,1))/np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        # img = np.pad(img, ((0,8),(0,0),(0,0)), 'constant', constant_values=0)
        # mask = np.pad(mask, ((0,8),(0,0),(0,0)), 'constant', constant_values=0)
        # bbox = bbox * np.array([1, 1066/400, 800/300, 1066/400, 800/300]).reshape(1,5)
        img_ = Image.fromarray((img).astype('uint8').transpose(1, 2, 0))
        # plt.imshow(img_)
        # plt.savefig("img1.png")
        if len(bbox) == 1:
            mask_ = Image.fromarray((mask.squeeze(0)).astype('uint8'))
        else:
            mask_ = Image.fromarray((mask).astype('uint8').transpose(1, 2, 0))
        img_ = self.transform_img(img_)
        mask_ = self.transform_mask(mask_)
        if len(bbox) == 1:
            mask_ = mask_.unsqueeze(0)
        bbox_ = torch.tensor(bbox)
        bbox_ = bbox_ * torch.tensor([  800/300,1066/400 , 800/300,1066/400]).reshape(1,4) + torch.tensor([0, 11, 0, 11]).reshape(1,4) # bbox: x,y,x,y 
        # i = (img_*255).permute(1,2,0).numpy()
        # i = Image.fromarray((i).astype('uint8'))
        # plt.imshow(i)
        # plt.savefig("img2.png")
        # print(bbox_.shape[0])
        # print(mask_.squeeze(0).shape[0])
        # check flag
        assert img_.shape == (3, 800, 1088)
        assert bbox_.shape[0] == mask_.squeeze(0).shape[0]
        return img_, mask_, bbox_
    
    def load_single_h5py(self, h5path, index):
        with h5py.File(h5path, 'r') as f:
            data = f["data"][index]

        return data
    def load_multi_h5py(self, h5path, index, num):
        with h5py.File(h5path, 'r') as f:
            data = f["data"][index:index+num]

        return data
    
    def load_npy(self, npypath):
        data = np.load(npypath, allow_pickle=True)
        return data


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn
        imgs, labels, masks, bboxes = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        label_list = list(labels)
        transed_mask_list = [mask.clone() for mask in masks]
        trans_bbox_list = [bbox.clone() for bbox in bboxes]
        
        return imgs, label_list, transed_mask_list,trans_bbox_list

    def loader(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collect_fn)
        return dataloader




## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = '../data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = '../data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]

    # build the dataset
    dataset = BuildDataset(paths)
    #print(dataset[0])
    
    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    print( full_size)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader
    print(train_dataset[0])
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = test_build_loader.loader()
    print("Finish building dataloader")
    # print(len(train_loader))
    # print(len(test_loader))
    img, label, mask, bbox = next(iter(train_loader))
    print(img.shape)
    print(label)
    print(mask[1].shape)
    print(bbox[1].shape)
    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    print(torch.cuda.is_available())
    exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        print(len(data))
        exit()
        img, label, mask, bbox = [data[i] for i in range(len(data))]
        print(img.shape)
        print(label)
        print(mask[0].shape)
        print(bbox[0].shape)
        exit()
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]


        # plot the origin img
        for i in range(batch_size):
            ## TODO: plot images with annotations
            plt.savefig("./testfig/visualtrainset"+str(iter)+".png")
            plt.show()

        if iter == 10:
            break

