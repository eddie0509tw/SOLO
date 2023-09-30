import torchvision
import torch
import torch.nn.functional as F
import numpy as np


def Resnet50Backbone(checkpoint_file=None, device="cpu", eval=True):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)

    if eval == True:
        model.eval()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    resnet50_fpn = model.backbone

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)

        resnet50_fpn.load_state_dict(checkpoint['backbone'])

    return resnet50_fpn

if __name__ == '__main__':
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    resnet50_fpn = Resnet50Backbone(device=device)
    # backbone = Resnet50Backbone('checkpoint680.pth')
    E = torch.ones([2,3,800,1088], device=device)
    backout = resnet50_fpn(E)
    print(backout.keys())
    print(backout["0"].shape)
    print(backout["1"].shape)
    print(backout["2"].shape)
    print(backout["3"].shape)
    print(backout["pool"].shape)
    # a = F.interpolate(backout["0"], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
    # b = F.interpolate(backout["0"], size=a.size()[-2:], mode='bilinear', align_corners=False)

    # print(a)
    # print(b)
    # x_idx = torch.arange(0, 4, 1)
    # print(x_idx)
    # x_min = x_idx.min().item()
    # x_max = x_idx.max().item()
    # x_idx = -1 + 2 * (x_idx - x_min) / (x_max - x_min)
    # print(x_idx)



    # ins_feat = torch.randn(3, 4, 5)  # Example shape

    # # Create a linspace tensor
    # x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
    # y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
    # print(x_range)
    # print(y_range)
    a = torch.tensor([[1.2,2.54,3],[4,5.3,6]])
    top = torch.tensor([0,1])
    down = torch.tensor([1,2])
    print(int(2.9))
    k = torch.where(a>3,0,a)
    print(k[top:down,:])
    # area = torch.randn(3, 1) > 0.4
    # print(area)
    # print(area.nonzero())
    # print(torch.nonzero(area).squeeze(1))
    # print(area.nonzero().flatten())
    # y, x = torch.meshgrid(y_range, x_range)
    # print(y)
    # y = y.expand([ins_feat.shape[0], 1, -1, -1])
    # print(y)
    

