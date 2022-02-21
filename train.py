import torch
from tqdm import tqdm
from os.path import join as pathjoin
from constructs import args
from vae import VAE

def elbo_loss(x, reconstruction_means, q_means, q_covs):

	# do stuff
	return loss

trainLoader=args["trainLoader"] # a torch.utils.data.DataLoader object
epochs=args["epochs"]
optimizers=args["optimizer"]
lr_schedulers=args["lr_scheduler"]

model=VAE(
	args["data_dim"],
	args["latent_dim"],
	args["encoder_net"],
	args["decoder_net"],
	args["num_expectation_samples"]
	)

model.train()

for epoch in range(epochs):

	with tqdm(trainLoader) as t:

		t.set_description("Epoch: {}".format(epoch))
	
		for x in t:

			reconstruction_means, q_means, q_covs=model(x)
			loss=elbo_loss(x, reconstruction_means, q_means, q_covs)
			loss.backward()

			for opts in optimizers:
				opts.step()

			t.set_postfix(elbo_loss=loss.item())

		for schs in lr_schedulers:
			schs.step()


torch.save(model.state_dict(), pathjoin(args["logging_dir"], "model.pth"))
print("Done!")