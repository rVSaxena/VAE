import torch
import torch.nn as nn
import pandas as pd
from resnet import ResidualBlock

class Reshape(nn.Module):

    def __init__(self, out_shape):
    
        super(Reshape, self).__init__()
        self.out_shape=out_shape
        return

    def forward(self, x):

        return x.reshape(self.out_shape)

class mnist_dataset(torch.utils.data.Dataset):

    def __init__(self):
        return

    def __len__(self):
        return 70000

    def __getitem__(self, index):
        x=torch.from_numpy((pd.read_csv("data/mnist.csv", skiprows=index, nrows=1).values[0, 1:].astype('double')/255).reshape((1, 28, 28)))
        return x


args={}
args["device"]="cuda"

dataset=mnist_dataset()
args["trainLoader"]=torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

args["epochs"]=50
args["data_dim"]=28*28
args["latent_dim"]=4
args["num_expectation_samples"]=5
args["logging_dir"]="mnist_run/"

# made sure encoder returns (batch_size, 2*latent_dim)
encoder=nn.Sequential(
    ResidualBlock(1, 64, False, 3),
    ResidualBlock(64, 64, False, 3),
    ResidualBlock(64, 32, False, 3),
    ResidualBlock(32, 4, True, 3),
    ResidualBlock(4, 1, True, 3),
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.Linear(49, 8)
    ).double().to(args["device"])

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
    ).double().to(args["device"])

args["encoder_net"]=encoder
args["decoder_net"]=decoder

params=list(args["encoder_net"].parameters())+list(args["decoder_net"].parameters())
args["optimizers"]=[torch.optim.Adam(params)]

args["lr_schedulers"]=[torch.optim.lr_scheduler.StepLR(args["optimizers"][0], 10, gamma=0.1)]
