import torch
import torch.nn as nn

class VAE(nn.Module):

	def __init__(self, device, data_dimension, latent_dimension, encoder_net, decoder_net, num_expectation_samples=500):

		"""
		Encoder should output (num_x, 2*latent_dimension)
		Decoder should expect an input of shape (num_x, num_exepectation_samples, latent_dimension)
		"""

		# TODO data_dimension is probably not needed so remove that

		super(VAE, self).__init__()
		self.device=device
		self.data_dim=data_dimension
		self.latent_dim=latent_dimension
		self.encoder=encoder_net
		self.decoder=decoder_net
		self.num_expectation_samples=num_expectation_samples

		return

	def forward(self, x):
		# The shape of x is supposed to be 
		# whatever is expected by encoder

		g_stats=self.encoder(x)
		
		# Expected shape is (N, 2*latent_dimension) -> mean and the diag element of the cov_mat
		# g_stats contains the numbers needed to construct the mean and cov matrix,
		# for use in re-parameterization. cov mat is chosen to be diagonal

		mean, cov=g_stats[:, :self.latent_dim], torch.diag_embed(torch.exp(g_stats[:, self.latent_dim:]))
		
		with torch.no_grad():
			epsilons=torch.normal(0, 1, size=(mean.shape[0], self.num_expectation_samples, self.latent_dim)).type(torch.float32).to(self.device)
	
		all_z=(torch.sqrt(cov)@epsilons.swapaxes(1, 2)).swapaxes(1, 2)+mean[:, None, :]

		output=self.decoder(all_z)

		return output, mean, cov


