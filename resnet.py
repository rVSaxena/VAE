import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    """
    Implements ResidualBlock for
    rectangular feature maps with input shape == output shape
    OR input shape == (output shape)*2 ie in all dimensions (except the batch dimension).
    In the latter case, input_shape must be even in all dimensions (expect the batch dimension)
    """

    def __init__(self, in_channels, out_channels, downsample, kernel_dim=3, **kwargs):
        
        """
        in_channels: int
        out_channels: int
        kernel_dim: int
        
        """
        assert kernel_dim%2==1, "Only odd kernel dimensions supported. Received {}".format(kernel_dim)

        super(ResidualBlock, self).__init__()
        self.downsample=downsample

        self.Normalizer1=nn.BatchNorm2d(out_channels)
        self.Normalizer2=nn.BatchNorm2d(out_channels)

        stride=2 if self.downsample else 1
        if self.in_feat_size!=self.out_feat_shape:
            self.SAK=nn.Conv2d(in_channels, out_channels, 1, stride=2)

        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_dim, stride=stride, padding=(kernel_dim-1)/2)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_dim, padding=(kernel_dim-1)/2) # this one maintains shape. So stride 1 and padd=(k-1)/2 work
        self.activation1=nn.ReLU(inplace=True)
        self.activation2=nn.ReLU(inplace=True)

        return

    def forward(self, x):
        """
        The shape of x must be accd to (n,c,h,w)
        """

        # compute the first block
    
        out=self.conv1(x)
        out=self.Normalizer1(out)
        out=self.activation1(out)

        # ready the input for addition
        if self.downsample:
            x=self.SAK(x)

        # compute the output
        out=self.conv2(out)+x
        out=self.Normalizer2(out)
        out=self.activation2(out)
    
        return out

