import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_images(dataset_type, name, device):
    sample_batch = next(iter(dataset_type))
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(
        sample_batch[0].to(device)[: 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('Training Images {}'.format(name))
    plt.close('all')

def generate_latent_code(Generator, noise, feature):
    with torch.no_grad():
        generated = Generator(noise).detach().cpu()
    # Display the generated image.
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(generated, nrow=10, padding=2, normalize=True), (1, 2, 0)))
    # plt.show()
    plt.savefig('Manipulating_Latent_code_{}'.format(feature))
    plt.close('all')

def weights_initialization(model):
    if (type(model) == nn.ConvTranspose2d or type(model) == nn.Conv2d):
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif (type(model) == nn.BatchNorm2d):
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


""" Below is learned from https://github.com/Natsu6767/InfoGAN-PyTorch"""
def noise_sample(num_discrete_code, dim_discrete_code, num_continuous_code, num_z, batch_size, device):

    z = torch.randn(batch_size, num_z, 1, 1, device=device)
    index = np.zeros((num_discrete_code, batch_size))
    if (num_discrete_code != 0):
        discrete_code = torch.zeros(batch_size, num_discrete_code, dim_discrete_code, device=device)

        for i in range(num_discrete_code):
            index[i] = np.random.randint(dim_discrete_code, size=batch_size)
            discrete_code[torch.arange(0, batch_size), i, index[i]] = 1.0
        discrete_code = discrete_code.view(batch_size, -1, 1, 1)

    # Random uniform between -1 and 1.
    if (num_continuous_code != 0):
        continuous_code = torch.rand(batch_size, num_continuous_code, 1, 1, device=device) * 2 - 1

    noise = z
    if (num_discrete_code != 0):
        noise = torch.cat((z, discrete_code), dim=1)
    if (num_continuous_code != 0):
        noise = torch.cat((noise, continuous_code), dim=1)

    return noise, index

class nllloss:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())
        return nll

