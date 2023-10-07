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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import os


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path, augmentation=True):
        # TODO: load dataset, make mask list
        self.imgs_path = path[0]
        self.masks_path = path[1]
        self.labels_path = path[2]
        self.bboxes_path = path[3]

        self.augmentation = augmentation

        # self.imgs_data = self.load_h5py(self.imgs_path)
        # self.masks_data = self.load_h5py(self.masks_path)
        self.labels_data = self.load_npy(self.labels_path)
        self.bboxes_data = self.load_npy(self.bboxes_path)
        # self.resize = transforms.Resize((800, 1066))
        # self.totensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # self.pad = transforms.Pad((11,0),fill=0)
        # self.transform_img = transforms.Compose([transforms.Resize((800, 1066)),
        #                                      transforms.ToTensor(),
        #                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #                                      transforms.Pad((11,0),fill=0)])
        # self.transform_mask = transforms.Compose([transforms.Resize((800, 1066)),
        #                                      transforms.ToTensor(),
        #                                      transforms.Pad((11,0),fill=0)])
        self.masks__idx = []
        count = 0
        for i in range(len(self.labels_data)):
            self.masks__idx.append(count)
            count += self.labels_data[i].shape[0]
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
        img = self.load_single_h5py(self.imgs_path, index)/ 255.0
        label = torch.tensor(self.labels_data[index], dtype=torch.long)
        mask = self.load_multi_h5py(self.masks_path, index,len(label))
        bbox = self.bboxes_data[index]

        img = torch.tensor(img, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)
        # print(img.shape)
        #print(mask.shape)
        # print(len(label))
        #print("original",bbox)

        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)
        if self.augmentation and (np.random.rand(1).item() > 0.5):
            # perform horizontally flipping (data augmentation)
            assert transed_img.ndim == 3
            assert transed_mask.ndim == 3
            transed_img = torch.flip(transed_img, dims=[2])
            transed_mask = torch.flip(transed_mask, dims=[2])
            # bbox transform
            h, w  = transed_img.size()[-2:]
            transed_bbox_new = transed_bbox.clone()
            transed_bbox_new[:, 0] = w - transed_bbox[:, 2]
            transed_bbox_new[:, 2] = w - transed_bbox[:, 0]
            transed_bbox = transed_bbox_new

            assert torch.all(transed_bbox[:, 0] < transed_bbox[:, 2])

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
        # The input mask has shape (n_obj, 800, 1088), so we need to process each object's mask separately.
        # img_ = Image.fromarray((img).astype('uint8').transpose(1, 2, 0))
        # mask_ = [Image.fromarray(m.astype('uint8')) for m in mask]  # process each object's mask separately.
        # mask_ = torch.stack([self.transform_mask(m) for m in mask_])
        # img_ = self.transform_img(img_)
        img = self.torch_interpolate(img, 800, 1066)  # (3, 800, 1066)
        img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
        img = F.pad(img, [11, 11])

        # mask: (N_obj * 300 * 400)
        mask = self.torch_interpolate(mask, 800, 1066)  # (N_obj, 800, 1066)
        mask = F.pad(mask, [11, 11])  
        bbox_ = torch.tensor(bbox)
        bbox_ = bbox_ * torch.tensor([1066 / 400, 800 / 300,  1066 / 400, 800 / 300]).reshape(1, 4) + torch.tensor(
            [11, 0,11, 0]).reshape(1, 4)
        # bbox_ = torch.tensor(bbox_)
        #print("after",bbox_)
        # print(mask.shape)
        # print(bbox_.shape[0])
        # Check flag
        assert img.shape == (3, 800, 1088)
        assert bbox_.shape[0] == mask.shape[0]
        return img, mask, bbox_


    def load_single_h5py(self, h5path, index):
        with h5py.File(h5path, 'r') as f:
            data = f["data"][index]

        return data
    def load_multi_h5py(self, h5path, index, num):
        idx = self.masks__idx[index]
        with h5py.File(h5path, 'r') as f:
            data = f["data"][idx:idx+num] * 1.0

        return data
    
    def load_npy(self, npypath):
        data = np.load(npypath, allow_pickle=True)
        return data

    @staticmethod
    def torch_interpolate(x, H, W):
        """
        A quick wrapper fucntion for torch interpolate
        :return:
        """
        assert isinstance(x, torch.Tensor)
        C = x.shape[0]
        # require input: mini-batch x channels x [optional depth] x [optional height] x width
        x_interm = torch.unsqueeze(x, 0)
        x_interm = torch.unsqueeze(x_interm, 0)

        tensor_out = F.interpolate(x_interm, (C, H, W))
        tensor_out = tensor_out.squeeze(0)
        tensor_out = tensor_out.squeeze(0)
        return tensor_out

    @staticmethod
    def unnormalize_bbox(bbox):
        """
        Unnormalize one bbox annotation. from 0-1 => 0 - 1088
        x_res = x * 1066 + 11
        y_res = x * 800
        :param bbox: the normalized bounding box (4,)
        :return: the absolute bounding box location (4,)
        """
        bbox_res = torch.tensor(bbox, dtype=torch.float)
        bbox_res[0] = bbox[0] * 1066 + 11
        bbox_res[1] = bbox[1] * 800
        bbox_res[2] = bbox[2] * 1066 + 11
        bbox_res[3] = bbox[3] * 800
        return bbox_res

    @staticmethod
    def unnormalize_img(img):
        """
        Unnormalize image to [0, 1]
        :param img:
        :return:
        """
        assert img.shape == (3, 800, 1088)
        img = transforms.functional.normalize(img, mean=[0.0, 0.0, 0.0],
                                                          std=[1.0/0.229, 1.0/0.224, 1.0/0.225])
        img = transforms.functional.normalize(img, mean=[-0.485, -0.456, -0.406],
                                                          std=[1.0, 1.0, 1.0])
        return img
class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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
        transed_mask_list = []
        # for mask in masks:
        #     # Adjust the shape of each mask tensor to be (n_obj, 800, 1088)
        #     # The exact adjustment will depend on your initial mask shape
        #     transed_mask_list.append(mask)
        transed_mask_list = list(masks)
        # transed_mask_list = [mask.clone() for mask in masks]
        # trans_bbox_list = [bbox.clone() for bbox in bboxes]
        trans_bbox_list = list(bboxes)
       
        # return imgs, label_list, masks, bboxes
        return imgs, label_list, transed_mask_list,trans_bbox_list

    def loader(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collect_fn,pin_memory=self.pin_memory)
        return dataloader


def plot_and_save_batch(batch, save_dir):
    img, label, mask, bbox = batch
    # print(bbox)
    print(label)
    # print(len(mask))
    print(mask[0].shape,mask[1].shape)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    batch_size = img.shape[0]
    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    
    for i in range(batch_size):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        
        img_data = img[i].cpu().numpy().transpose(1, 2, 0) * 255
        img_data = np.clip(img_data, 0, 1)
        ax.imshow(img_data)

        for j in range(len(label[i])):
                # plot the bbox
                x1, y1, x2, y2 = bbox[i][j]
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)

                # plot the mask
                mask_np = mask[i][j].cpu().numpy()
                mask_np = np.ma.masked_where(mask_np == 0, mask_np)
                ax.imshow(mask_np, cmap=mask_color_list[j], alpha=0.5, interpolation='none')
        fig.savefig(os.path.join(save_dir, f"visualtrainset_{i}.png"))
        plt.close(fig)
            
        
        # # Visualize masks
        # for m, l in zip(mask[i], label[i]):
        #     c = mask_color_list[l]
        #     mask_data = m.cpu().numpy()
        #     # print(mask_data.shape)
        #     if mask_data.ndim == 3:  # if the mask is (1, height, width)
        #         mask_data = mask_data[0]  # remove the singleton dimension
        #     elif mask_data.ndim == 1:  # if the mask is (1088,)
        #         height = int(np.sqrt(mask_data.size))  # assuming the mask is square
        #         mask_data = mask_data.reshape((height, height))  # reshape to 2D
        #     ax.imshow(mask_data, alpha=0.5, cmap=c)

        # # Visualize bounding boxes
        # for b in bbox[i]:
        #     rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
            
        # # Saving the plots to the specified folder
        # fig.savefig(os.path.join(save_dir, f"visualtrainset_{i}.png"))
        # plt.close(fig)


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
    device = torch.device("cpu")
    #print(dataset[0])
    
    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    #print(full_size)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader
    #print(train_dataset[11])
    batch_size = 10
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = test_build_loader.loader()
    print("Finish building dataloader")
    print(len(train_loader))
    print(len(test_loader))
    img, label, mask, bbox = next(iter(train_loader))
    print(img[0].shape)
    print(label)
    print(mask[0].shape,mask[1].shape)
    print(bbox)
    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "../plot"

    for iter, data in enumerate(train_loader, 0):
        plot_and_save_batch(data, save_dir)
        if iter >= 0:  # Stopping after 5 batches, you can change this to your preference.
            break


    # for iter, data in enumerate(train_loader, 0):
    #     img, label, mask, bbox = [data[i] for i in range(len(data))]
    #     print(img.shape)
    #     print(label)
    #     print(mask[0].shape,mask[1].shape)
    #     print(bbox[0].shape,bbox[1].shape)
    #     exit()
    #     # check flag
    #     assert img.shape == (batch_size, 3, 800, 1088)
    #     assert len(mask) == batch_size

    #     label = [label_img.to(device) for label_img in label]
    #     mask = [mask_img.to(device) for mask_img in mask]
    #     bbox = [bbox_img.to(device) for bbox_img in bbox]


    #     # plot the origin img
    #     for i in range(batch_size):
    #         ## TODO: plot images with annotations
    #         plt.savefig("./testfig/visualtrainset"+str(iter)+".png")
    #         plt.show()

    #     if iter == 10:
    #         break

