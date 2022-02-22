import torch
import numpy as np
from tqdm import tqdm
from os.path import join as pathjoin
from constructs import args
from vae import VAE
from os import makedirs

def elbo_loss(x, reconstruction_means, q_means, q_covs):

	# do stuff

	# if n is batch size, then
	# shape of 
	# x: n, shape_of_1_sample
	# reconstruction_means: n, num_expectation_samples, shape_of_1_sample
	# q_means: n, latent_dimension
	# q_covs: n, latent_dimension, latent_dimension
	
	a=torch.mean(torch.square((reconstruction_means-x[:, None, :]).flatten()))
	b=torch.mean(
		0.5*torch.sum(
			torch.diagonal(q_covs, dim1=1, dim2=2)+torch.square(q_means)-1-torch.log(torch.diagonal(q_covs, dim1=1, dim2=2)),
			axis=1
			)
		)

	return a+b

device=args["device"]
trainLoader=args["trainLoader"] # a torch.utils.data.DataLoader object
epochs=args["epochs"]
optimizers=args["optimizers"]
lr_schedulers=args["lr_schedulers"]

model=VAE(
	args["device"],
	args["data_dim"],
	args["latent_dim"],
	args["encoder_net"],
	args["decoder_net"],
	args["num_expectation_samples"]
	)

model.train()
makedirs(pathjoin(args["logging_dir"], "models"), exist_ok=True)
makedirs(pathjoin(args["logging_dir"], "loss_values"), exist_ok=True)

if __name__=='__main__':
	
	for epoch in range(epochs):

		lossarr=[]
		
		with tqdm(trainLoader) as t:

			t.set_description("Epoch: {}".format(epoch))
		
			for x in t:

				x=x.to(device)
				reconstruction_means, q_means, q_covs=model(x)
				loss=elbo_loss(x, reconstruction_means, q_means, q_covs)
				loss.backward()
				lossarr.append(loss.item())
				
				for opts in optimizers:
					opts.step()

				t.set_postfix(elbo_loss=loss.item())

			for schs in lr_schedulers:
				schs.step()

		torch.save(model.state_dict(), pathjoin(args["logging_dir"], "models", "{}.pth".format(epoch)))
		np.savetxt(pathjoin(args["logging_dir"], "loss_values", "epoch_{}.csv".format(epoch)), lossarr, delimiter=",")

	torch.save(model.state_dict(), pathjoin(args["logging_dir"], "final_model.pth"))
	print("Done! Model saved at {}".format(pathjoin(args["logging_dir"], "model.pth")))