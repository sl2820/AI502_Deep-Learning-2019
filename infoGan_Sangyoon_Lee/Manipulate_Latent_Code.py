from model import Generator
from utils import*
""" Help from https://github.com/Natsu6767/InfoGAN-PyTorch """

class manipulate_latent_code(object):
    def __init__(self, dataset_name, device):
        self.load_path="../checkpoint/model_final_{}".format(dataset_name)
        # Load model
        self.state_dict = torch.load(self.load_path)
        self.device = device

        # Create the generator network.
        self.Gen = Generator().to(self.device)
        # Load trained weights
        self.Gen.load_state_dict(self.state_dict['Gen'])

        self.c = np.linspace(-2, 2, 10).reshape(1, -1)
        self.c = np.repeat(self.c, 10, 0).reshape(-1, 1)
        self.c = torch.from_numpy(self.c).float().to(self.device)
        self.c = self.c.view(-1, 1, 1, 1)

        self.zeros = torch.zeros(100, 1, 1, 1, device=self.device)

        # Continuous latent code (continouos features --> like rotation, width)
        self.c2 = torch.cat((self.c, self.zeros), dim=1)
        self.c3 = torch.cat((self.zeros, self.c), dim=1)

        self.idx = np.arange(10).repeat(10)
        self.dis_c = torch.zeros(100, 10, 1, 1, device=self.self.device)
        self.dis_c[torch.arange(0, 100), self.idx] = 1.0

        # Discrite latent code (categorical --> like type of clothing or number)
        self.c1 = self.dis_c.view(100, -1, 1, 1)

        self.z = torch.randn(100, 62, 1, 1, device=self.device)

        # c2 (columns) vs. c1 (rows)
        self.noise_c2 = torch.cat((self.z, self.c1, self.c2), dim=1)
        # c3 (columns) vs. c1 (rows)
        self.noise_c3 = torch.cat((self.z, self.c1, self.c3), dim=1)

        generate_latent_code(self.Gen, self.noise_c2, "c2")
        generate_latent_code(self.Gen, self.noise_c3, "c3")

