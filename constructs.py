import torch
import torch.nn as nn
from resnet import ResidualBlock

class Reshape(nn.Module):

    def __init__(self, out_shape):
    
        super(Reshape, self).__init__()
        self.out_shape=out_shape
        return

    def forward(self, x):

        return x.view(self.out_shape)


args={}
args["epochs"]=50
args["data_dim"]=28*28
args["latent_dim"]=4
args["num_expectation_samples"]=200
args["logging_dir"]

# made sure encoder returns (batch_size, 2*latent_dim)
encoder=nn.Sequential(
    ResidualBlock(1, 64, False, 3),
    ResidualBlock(64, 64, False, 3),
    ResidualBlock(64, 32, False, 3),
    ResidualBlock(32, 4, True, 3),
    ResidualBlock(4, 1, True, 3),
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.Linear(49, 8)
    )

# decoder will be given (batch_size, num_expectation_samples, latent_dim)

decoder=nn.Sequential(
    Reshape((-1, args["latent_dim"])),
    nn.Linear(args["latent_dim"], 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 784),
    nn.Sigmoid(),
    Reshape((-1, 1, 28, 28))
    )

args["encoder_net"]=encoder
args["decoder_net"]=decoder

params=list(args["encoder_net"].parameters())+list(args["decoder_net"].parameters())
args["optimizer"]=torch.optim.Adam(params)

args["lr_scheduler"]=torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
