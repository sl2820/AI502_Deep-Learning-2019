import torch
import torch.nn as nn

# from C.1MNIST of the Paper

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.fc1 = nn.ConvTranspose2d(74, 1024, 1, 1) #conv with stride 1 == fc
        self.batch_norm1 = nn.BatchNorm2d(1024)

        self.fc2 = nn.ConvTranspose2d(1024, 128, 7, 1)
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, padding = 1)
        self.batch_norm3 = nn.BatchNorm2d(64)

        self.upconv2 = nn.ConvTranspose2d(64, 1, 4, 2, padding= 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = input

        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.upconv1(x)
        x = self.batch_norm3(x)
        x = self.relu(x)

        x = self.upconv2(x)
        x = self.sigmoid(x)

        return x

class Discriminator (nn.Module):
    def __init__(self):
        super().__init__()

        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        #no batchnorm

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Conv2d(128, 1024, 7)
        self.batch_norm3 = nn.BatchNorm2d(1024)

    def forward(self, input):
        x = input

        x = self.conv1(x)
        x = self.leakyrelu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leakyrelu(x)

        x = self.fc1(x)
        x = self.batch_norm3(x)
        x = self.leakyrelu(x)

        return x



class D_last(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Conv2d(1024, 1, 1)

    def forward(self, input):
        x = input
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

class Q_last(nn.Module):
    def __init__(self):
        super().__init__()

        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.batch_norm = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, input):
        x = input

        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.leakyrelu(x)

        disc_logits = self.conv_disc(x).squeeze()
        mu = self.conv_mu(x).squeeze()
        var_temp = self.conv_var(x).squeeze()
        var = torch.exp(var_temp)

        return disc_logits, mu, var

