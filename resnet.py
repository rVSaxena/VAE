import torch
import torch.nn as nn


class ResidualEncoderBlock(nn.Module):

    """
    Implements ResidualBlock for
    rectangular feature maps with input shape == output shape
    OR input shape == (output shape)*2 ie in all dimensions (except the batch dimension).
    In the latter case, input_shape must be even in all dimensions (expect the batch dimension)
    """

    def __init__(self, in_channels, out_channels, downsample, kernel_dim=3, normalizer=nn.BatchNorm2d, **kwargs):
        
        """
        in_channels: int
        out_channels: int
        downsample: Boolean
        kernel_dim: int
        
        use nn.Identity to skip normalization
        """
        assert kernel_dim%2==1, "Only odd kernel dimensions supported. Received {}".format(kernel_dim)

        super(ResidualEncoderBlock, self).__init__()
        self.downsample=downsample
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.Normalizer1=normalizer(out_channels)
        self.Normalizer2=normalizer(out_channels)

        # SAK is shape adjusting kernel 
        if self.downsample or (in_channels!=out_channels):
            if not self.downsample:
                self.SAK=nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
            else:
                self.SAK=nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)


        stride=2 if self.downsample else 1
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_dim, stride=stride, padding=int((kernel_dim-1)/2))
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_dim, padding=int((kernel_dim-1)/2)) # this one maintains shape. So stride 1 and padd=(k-1)/2 work
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
        if self.downsample or (self.in_channels!=self.out_channels):
            x=self.SAK(x)

        # compute the output
        out=self.conv2(out)
        out=self.Normalizer2(out)+x
        out=self.activation2(out)
        
        return out


class ResidualDecoderBlock(nn.Module):

    """
    Implements ResidualBlock for
    rectangular feature maps with input shape == output shape
    OR 2*input shape == output shape ie in all dimensions (except the batch dimension).
    In the latter case, input_shape must be even in all dimensions (expect the batch dimension)
    """

    def __init__(self, in_channels, out_channels, upsample, kernel_dim=3, normalizer=nn.BatchNorm2d, **kwargs):
        
        """
        in_channels: int
        out_channels: int
        upsample: Boolean
        kernel_dim: int
        
        use nn.Identity to skip normalization
        """
        
        assert kernel_dim%2==1, "Only odd dimension supported, got {}".format(kernel_dim)

        super(ResidualDecoderBlock, self).__init__()
        self.upsample=upsample
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.Normalizer1=normalizer(out_channels)
        self.Normalizer2=normalizer(out_channels)

        # SAK is shape adjusting kernel 
        if self.upsample or (in_channels!=out_channels):
            if not self.upsample:
                self.SAK=nn.ConvTranspose2d(in_channels, out_channels, 1, stride=1, bias=False, padding=0)
            else:
                self.SAK=nn.ConvTranspose2d(in_channels, out_channels, 1, stride=2, bias=False, padding=0, output_padding=1)


        stride=2 if self.upsample else 1
        output_pad=1 if self.upsample else 0
        self.conv1=nn.ConvTranspose2d(in_channels, out_channels, kernel_dim, stride=stride, padding=int((kernel_dim-1)/2), output_padding=output_pad)
        self.conv2=nn.ConvTranspose2d(out_channels, out_channels, kernel_dim, padding=int((kernel_dim-1)/2)) # this one maintains shape. So stride 1 and padd=(k-2)/2 work
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
        if self.upsample or (self.in_channels!=self.out_channels):
            x=self.SAK(x)

        # compute the output
        out=self.conv2(out)
        out=self.Normalizer2(out)+x
        out=self.activation2(out)
        
        return out

