# Adapted from https://github.com/pytorch/vision/blob/v0.11.0/torchvision/models/resnet.py

import torch
from torchvision import transforms
from typing import Type, Union, List, Any, Callable
from torchvision.models.resnet import _resnet, BasicBlock, Bottleneck, ResNet, resnet18
import torch.nn as nn


def load_resnet50(pretrained: bool = False):
    backbone = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrained, progress=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return backbone, normalize

def load_resnet18(pretrained: bool = False):
    backbone = _resnet('resnet18', Bottleneck, [2, 2, 2, 2], pretrained=pretrained, progress=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return backbone, normalize

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNetFeatures(block, layers, **kwargs)
    if pretrained:
        if int(torch.__version__[0]) <= 1:
            from torch.hub import load_state_dict_from_url
            from torchvision.models.resnet import model_urls
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
        else:
            raise NotImplementedError("Pretrained models not supported in PyTorch 2.0+")
    return model


class ResNetFeatures(ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)

    def _forward_impl(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }


# Adapted from Diffusion Policy codebase
def replace_submodules(
        root_module:nn.Module,
        predicate: Callable[[nn.Module],bool],
        func: Callable[[nn.Module],nn.Module]) -> nn.Module:
    
    if predicate(root_module):
        return func(root_module)
    
    bn_list=[k.split('.') for k,m 
                     in root_module.named_modules(remove_duplicate=True)
                     if predicate(m)]
    for *parent,k in bn_list:
        parent_module=root_module
        if(len(parent)>0):
            parent_module=root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module,nn.Sequential):
            src_module=parent_module[int(k)]
        else:
            src_module=getattr(parent_module,k)
        tgt_module=func(src_module)
        if isinstance(parent_module,nn.Sequential):
            parent_module[int(k)]=tgt_module
        else:
            setattr(parent_module,k,tgt_module)
    bn_list=[k.split('.') for k,m 
                     in root_module.named_modules(remove_duplicate=True)
                     if predicate(m)]
    assert len(bn_list)==0
    return root_module


class SpatialSoftmax(nn.Module):
    def __init__(self,height,width,channel):
        super().__init__()
        self.height=height
        self.width=width
        self.channel=channel
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,x):
        x=x.view(-1,self.channel,self.height*self.width)
        x=self.softmax(x)
        x=x.view(-1,self.channel,self.height,self.width)
        return x

        
class ResNet18_custom(nn.Module):
    def __init__(self, out_channels=60, spatial_softmax_size=(32,32,512),pretrained=True):
        super().__init__()
        self.out_channels=out_channels
        net=resnet18(pretrained=pretrained)
        self.normalize=nn.functional.normalize
        self.backbone=nn.Sequential(*list(net.children())[:-2])
        # upsample from 8*8 to 32*32
        self.upsample=nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # dimensionality stays the same
        self.spatial_softmax=SpatialSoftmax(*spatial_softmax_size)
        self.fc=nn.Linear(512,out_channels)

    def forward(self,x):
        x=self.normalize(x)
        x=self.backbone(x)
        x=self.upsample(x)
        x=self.spatial_softmax(x)
        x=x.view(-1,512)
        x=self.fc(x)
        x=x.view(-1,32,32,self.out_channels)
        return x

        