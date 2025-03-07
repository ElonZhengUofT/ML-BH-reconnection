import torch
import torch.nn as nn
import torchvision

ResNet = torchvision.models
ResNet = torchvision.models.resnet50
Vit = torchvision.models.vit_b_16

class PreTrainVitUnet(nn.Module):
    """
    PreTrainVitUnet Model:
    - Pretrained ResNet(Resnet50) as extractor
    - Pretrained Vision Transformer (ViT-b-16) as encoder
    - UNet as decoder

    Issues: The output of the ResNet is not compatible with the input of the ViT.
    And the output of the ViT is not compatible with the input of the UNet.

    Solution:
    """

if __name__ == "__main__":
    # check structure of the resnet
    resnet = ResNet()
    # cut the last layer
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    print(resnet)
    # check the output of the resnet (size of the output)
    x = torch.randn(1, 3, 224, 224)
    print(resnet(x).shape)

    resnet1 = ResNet()
    print(resnet1(x).shape)


