import torch
import torch.nn as nn
import pandas as pd
from resnet import ResidualEncoderBlock, ResidualDecoderBlock

class Reshape(nn.Module):

    def __init__(self, out_shape):
    
        super(Reshape, self).__init__()
        self.out_shape=out_shape
        return

    def forward(self, x):

        return x.reshape(self.out_shape)

class mnist_dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.all_data=torch.from_numpy((pd.read_csv("data/mnist.csv").values[:, 1:].astype('float32')/255).reshape((-1, 1, 28, 28)))
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
args["latent_dim"]=49
args["num_expectation_samples"]=2
args["logging_dir"]="mnist_run1/"

# made sure encoder returns (batch_size, 2*latent_dim)
encoder=nn.Sequential(
    ResidualEncoderBlock(1, 32, False, 3),
    ResidualEncoderBlock(32, 32, False, 3),
    ResidualEncoderBlock(32, 16, True, 3),
    ResidualEncoderBlock(16, 16, True, 3),
    ResidualEncoderBlock(16, 1, False, 3),
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.Linear(49, 2*args["latent_dim"])
    ).type(torch.float32).to(args["device"])

# encoder=nn.Sequential(
#     nn.Linear(784, 16),
#     nn.Sigmoid(),
#     nn.Linear(16, 16),
#     nn.Sigmoid(),
#     nn.Linear(16, 2*args["latent_dim"])
#     ).double().to(args["device"])

# decoder will be given (batch_size, num_expectation_samples, latent_dim)

# decoder=nn.Sequential(
#     Reshape((-1, args["latent_dim"])),
#     nn.Linear(args["latent_dim"], 16),
#     nn.ReLU(inplace=True),
#     nn.Linear(16, 16), 
#     nn.ReLU(inplace=True),
#     nn.Linear(16, 784),
#     nn.ReLU(inplace=True),
#     Reshape((-1, args["num_expectation_samples"], 1, 28, 28))
#     ).double().to(args["device"])

decoder=nn.Sequential(
    Reshape((-1, 1, 7, 7)),
    ResidualDecoderBlock(1, 32, False, 3),
    ResidualDecoderBlock(32, 32, False, 3),
    ResidualDecoderBlock(32, 16, True, 3),
    ResidualDecoderBlock(16, 16, True, 3),
    ResidualDecoderBlock(16, 4, False, 3),
    ResidualDecoderBlock(4, 1, False, 3),
    Reshape((-1, args["num_expectation_samples"], 1, 28, 28))
    ).type(torch.float32).to(args["device"])

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
torch.optim.lr_scheduler.StepLR(args["optimizers"][0], 9, gamma=0.8)
# torch.optim.lr_scheduler.LambdaLR(args["optimizers"][0], lr_lambda=f1)
]
