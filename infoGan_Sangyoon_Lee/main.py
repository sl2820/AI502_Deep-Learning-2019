import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from trainer import*
from Manipulate_Latent_Code import *
from utils import*

#Necessary Variables
file_path = 'data/'
batch_size = 128

#Random Seeds for reproducibility
seed = 1234
random.seed(seed)
torch.manual_seed(seed)

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#MNIST & FashionMNIST (same 28x28)
transform_mnist_type = transforms.Compose([transforms.Resize(28),
                                      transforms.CenterCrop(28),
                                      transforms.ToTensor()])
mnist_dataset = datasets.MNIST(file_path+'mnist/', train='train', download=True, transform=transform_mnist_type)
fashion_dataset = datasets.FashionMNIST(file_path+'fashion/', train='train', download=True, transform=transform_mnist_type)

dataloader_mnist = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
dataloader_fashionmnist = torch.utils.data.DataLoader(fashion_dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    mnist = Runner(batch_size, dataloader_mnist, "MNIST", device)
    mnist.run()
    mnist_latent_code_manipulation = manipulate_latent_code("MNIST", device)

    fashionmnist = Runner(batch_size, dataloader_fashionmnist, "FASHION_MNIST", device)
    fashionmnist.run()
    fashionmnist_latent_code_manipulation = manipulate_latent_code("FASHION_MNIST", device)

