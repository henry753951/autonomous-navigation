import torch
import torch.nn.modules
import torchvision


class VGG16BN(torch.nn.Module):
    """
    A class representing the VGG16 model with batch normalization.
    """

    def __init__(self, pretrained: bool = False) -> None:
        """
        VGG16BN constructor.

        Args:
            pretrained (bool): If True, loads pretrained weights.
        """
        super().__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33] + model[34:43]
        self.model = torch.nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VGG16BN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class ResNet(torch.nn.Module):
    """
    A class representing ResNet and its variants.
    """

    def __init__(self, layers: str, pretrained: bool = False) -> None:
        """
        ResNet constructor.

        Args:
            layers (str): Specify the type of ResNet architecture ('18', '34', '50', etc.).
            pretrained (bool): If True, loads pretrained weights.
        """
        super().__init__()

        if layers == "18":
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == "34":
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == "50":
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == "101":
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == "152":
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == "50next":
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == "101next":
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == "50wide":
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == "101wide":
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            msg = f"ResNet architecture '{layers}' is not implemented."
            raise NotImplementedError(msg)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ResNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Intermediate outputs from ResNet layers 2, 3, and 4.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4