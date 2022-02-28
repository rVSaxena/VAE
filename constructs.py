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
        self.all_data=torch.from_numpy((pd.read_csv("data/mnist.csv").values[:, 1:].astype('double')/255).reshape((-1, 1, 28, 28)))
        return

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        return self.all_data[index]


args={}
args["device"]="cuda"

dataset=mnist_dataset()
args["trainLoader"]=torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

args["epochs"]=80
args["data_dim"]=28*28
args["latent_dim"]=3
args["num_expectation_samples"]=20
args["logging_dir"]="mnist_run/"

# made sure encoder returns (batch_size, 2*latent_dim)
encoder=nn.Sequential(
    ResidualBlock(1, 8, False, 3),
    ResidualBlock(8, 16, False, 3),
    ResidualBlock(16, 16, False, 3),
    ResidualBlock(16, 4, True, 3),
    ResidualBlock(4, 1, True, 3),
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.Linear(49, 2*args["latent_dim"])
    ).double().to(args["device"])

# encoder=nn.Sequential(
#     nn.Linear(784, 16),
#     nn.Sigmoid(),
#     nn.Linear(16, 16),
#     nn.Sigmoid(),
#     nn.Linear(16, 2*args["latent_dim"])
#     ).double().to(args["device"])

# decoder will be given (batch_size, num_expectation_samples, latent_dim)

decoder=nn.Sequential(
    Reshape((-1, args["latent_dim"])),
    nn.Linear(args["latent_dim"], 16),
    nn.ReLU(inplace=True),
    nn.Linear(16, 16), 
    nn.ReLU(inplace=True),
    nn.Linear(16, 784),
    nn.ReLU(inplace=True),
    Reshape((-1, args["num_expectation_samples"], 1, 28, 28))
    ).double().to(args["device"])

args["encoder_net"]=encoder
args["decoder_net"]=decoder

params=list(args["encoder_net"].parameters())+list(args["decoder_net"].parameters())
args["optimizers"]=[torch.optim.Adam(params, lr=1e-3)]

def f1(epoch):
    if epoch>12:
        return 1.0
    if epoch%3==0:
        return 0.6**(epoch/3)
    return 1


args["lr_schedulers"]=[
torch.optim.lr_scheduler.StepLR(args["optimizers"][0], 9, gamma=0.6)
# torch.optim.lr_scheduler.LambdaLR(args["optimizers"][0], lr_lambda=f1)
]
