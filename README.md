# VAE

This is a generic and fast implementation of variational auto-encoders in SP float32 (GTX and RTX series Nvidia GPUs <br> have significantly higher SP performance vs DP) in PyTorch.

For a different usage (dataset), just a new constructs file needs to be written.

-----
An example usage on the MNIST dataset is provided.

### 2-D latent space:
![](pictures/2dimLat.jpg)

### 49D latent space:
![](pictures/49dimLat.jpg)