import torch
import torchvision
from torch import nn

# import segmentation_models as smp

# unet = smp.Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=3,                      # model output channels (number of classes in your dataset)
# )

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(torch.nn.Module):
  def __init__(self, ):
    super().__init__()

    self.base_model = torchvision.models.resnet18(pretrained=True)
    self.base_layers = list(self.base_model.children())

    # for i in range(8):
      # print(f"layer {i}: {self.base_layers[i]}")

    self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
    self.layer0_1x1 = convrelu(64, 64, 1, 0)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
    self.layer1_1x1 = convrelu(64, 64, 1, 0)
    self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
    self.layer2_1x1 = convrelu(128, 128, 1, 0)
    self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    self.layer3_1x1 = convrelu(256, 256, 1, 0)
    self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    self.layer4_1x1 = convrelu(512, 512, 1, 0)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
    self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
    self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
    self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

    self.conv_original_size0 = convrelu(3, 64, 3, 1)
    self.conv_original_size1 = convrelu(64, 64, 3, 1)
    self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

    self.conv_last = nn.Sequential(nn.Conv2d(64, 1, 1), nn.ReLU())

  def forward(self, input):
    # print(f"input size = {input.size()}")
    x_original = self.conv_original_size0(input)
    # print(f"x_original size = {x_original.size()}")
    x_original = self.conv_original_size1(x_original)
    # print(f"x_original size = {x_original.size()}")

    layer0 = self.layer0(input)
    # print(f"layer0 size = {layer0.size()}")
    layer1 = self.layer1(layer0)
    # print(f"layer1 size = {layer1.size()}")
    layer2 = self.layer2(layer1)
    # print(f"layer2 size = {layer2.size()}")
    layer3 = self.layer3(layer2)
    # print(f"layer3 size = {layer3.size()}")
    layer4 = self.layer4(layer3)
    # print(f"layer4 size = {layer4.size()}")


    layer4 = self.layer4_1x1(layer4)
    # print(f"layer4 after conv: {layer4.size()}")
    x = self.upsample(layer4)
    # print(f"layer4 upsampled: {x.size()}")
    layer3 = self.layer3_1x1(layer3)
    # print(f"layer3 upsampled: {x.size()}")
    x = torch.cat([x, layer3], dim=1)
    x = self.conv_up3(x)
    # print(f'x size: {x.size()}, layer2 size: {layer2.size()}')
    # print(type(x))

    x = self.upsample(x)
    # print(f'x size: {x.size()}')
    layer2 = self.layer2_1x1(layer2)
    # print(f"layer2 size: {layer2.size()}")

    x = torch.cat([x, layer2], dim=1)
    x = self.conv_up2(x)

    x = self.upsample(x)
    layer1 = self.layer1_1x1(layer1)
    x = torch.cat([x, layer1], dim=1)
    x = self.conv_up1(x)

    x = self.upsample(x)
    layer0 = self.layer0_1x1(layer0)
    x = torch.cat([x, layer0], dim=1)
    x = self.conv_up0(x)

    x = self.upsample(x)
    x = torch.cat([x, x_original], dim=1)
    x = self.conv_original_size2(x)

    out = self.conv_last(x) + 1e-4
    # out = torch.abs(x)
    #out = x
    # print(f"out size: {out.size()}")

    return out