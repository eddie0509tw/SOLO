import torchvision
import torch


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    ins_feat = torch.randn(3, 4, 5)  # Example shape

    # Create a linspace tensor
    x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
    y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
    print(x_range)
    print(y_range)
    y, x = torch.meshgrid(y_range, x_range)
    print(y)
    y = y.expand([ins_feat.shape[0], 1, -1, -1])
    print(y)
    

