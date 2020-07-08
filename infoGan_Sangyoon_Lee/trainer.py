import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets11
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn.functional as F

import random
import time
import itertools
from model import Generator, Discriminator, D_last, Q_last
from utils import*


class Runner(object):
    def __init__(self, batch_size_given, dataloader, dataset_name, device):
        self.data = dataloader
        self.dataset_name = dataset_name
        self.batch_size = batch_size_given
        self.total_epoch = 100
        self.learning_rate_d = 2e-4
        self.learning_rate_g = 1e-3
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.save_epoch = 25

        # Random Seeds for reproducibility
        self.seed = 1234
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Device
        self.device = device

        #Hyperparameters from C.1MNIST of the Paper
        self.num_z = 62
        self.num_discrete_latent_code = 1
        self.dim_discrete_latent_code = 10
        self.num_continuous_latent_code = 2


        #Create Network Architecture
        self.Gen = Generator().to(self.device)
        self.Gen.apply(weights_initialization)

        self.Dis = Discriminator().to(self.device)
        self.Dis.apply(weights_initialization)

        self.D_l = D_last().to(self.device)
        self.D_l.apply(weights_initialization)

        self.Q_l = Q_last().to(self.device)
        self.Q_l.apply(weights_initialization)

        #Loss for Discriminator
        self.Dis_criterion = nn.BCELoss()

        #Loss for Discrete latent code and Continuous latent code
        self.Q_discrete_latent_criterion = nn.CrossEntropyLoss()
        self.Q_continuous_latent_criterion = nllloss()

        #Optimizer
        self.Gen_optim = torch.optim.Adam(itertools.chain(self.Gen.parameters(), self.Q_l.parameters()), lr=self.learning_rate_g, betas=(self.beta1, self.beta2))
        self.Dis_optim = torch.optim.Adam(itertools.chain(self.Dis.parameters(), self.D_l.parameters()), lr=self.learning_rate_d, betas=(self.beta1, self.beta2))

        #Add noise
        self.fixed_noise = torch.randn(100, self.num_z, 1,1, device = self.device)
        self.index = np.arange(self.dim_discrete_latent_code).repeat(10)
        self.discrete_latent_code = torch.zeros(100, self.num_discrete_latent_code, self.dim_discrete_latent_code, device = self.device)
        for i in range(self.num_discrete_latent_code):
            self.discrete_latent_code[torch.arange(0,100), i, self.index] = 1.0

        self.discrete_latent_code = self.discrete_latent_code.view(100,-1,1,1)
        self.fixed_noise = torch.cat((self.fixed_noise, self.discrete_latent_code), dim=1)

        self.continuous_latent_code = torch.randn(100, self.num_continuous_latent_code,1,1,device=self.device) * 2 - 1
        self.fixed_noise = torch.cat((self.fixed_noise, self.continuous_latent_code), dim=1)




    def run(self):
        #label
        label_real = 1.0
        label_fake = 0.0

        #Save results
        Gen_losses = []
        Dis_losses = []
        Total_Entropy = []
        #Print about the final setting of the experiment
        print("----------------------------------------")
        print("Training starts...")
        print("Dataset: {}".format(self.dataset_name)) #or Fashion MNIST
        print("Epochs: {} | Batch size: {}".format(self.total_epoch, self.batch_size))
        print("----------------------------------------")


        begin_time = time.time()
        iterations = 0

        #Training
        for epoch in range(self.total_epoch):

            for batch_num, (x, y) in enumerate(self.data, 0):
                iterations +=1
                curr_b_size =x.size(0)

                real_data = x.to(self.device)
                discrete_noise = torch.randint(0, self.dim_discrete_latent_code, (curr_b_size,)).to(self.device)
                discrete_one_hot = F.one_hot(discrete_noise, self.dim_discrete_latent_code).to(self.device).float()

                #Update Discriminator and D_l
                self.Dis_optim.zero_grad()

                #Real Data learning
                label = torch.full((curr_b_size, ), label_real, device = self.device)
                output_real = self.Dis(real_data)
                probability_real = self.D_l(output_real).view(-1)
                loss_real = self.Dis_criterion(probability_real, label)
                loss_real.backward()

                #Fake Data learning
                label.fill_(label_fake)
                noise, index = noise_sample(self.num_discrete_latent_code, self.dim_discrete_latent_code, self.num_continuous_latent_code, self.num_z, curr_b_size, self.device)
                fake_data = self.Gen(noise)
                output_fake = self.Dis(fake_data.detach())
                probability_fake = self.D_l(output_fake).view(-1)
                loss_fake = self.Dis_criterion(probability_fake, label)
                loss_fake.backward()

                #Net Loss for the discriminator
                Net_Loss_Discriminator = loss_real + loss_fake
                self.Dis_optim.step()


                #Update Generator and Q_l
                self.Gen_optim.zero_grad()

                #Fake data to Real data
                output_gen = self.Dis(fake_data)
                label.fill_(label_real)
                probability_gen = self.D_l(output_gen).view(-1)
                gen_loss = self.Dis_criterion(probability_gen, label)

                Q_logits, Q_mu, Q_var = self.Q_l(output_gen)
                target = torch.LongTensor(index).to(self.device)

                discrete_loss = 0
                for j in range(self.num_discrete_latent_code):
                    discrete_loss+=self.Q_discrete_latent_criterion(Q_logits[:,j*10: j*10 + 10], target[j])

                continuous_loss = self.Q_continuous_latent_criterion(noise[:, self.num_z+self.num_discrete_latent_code*self.dim_discrete_latent_code : ].view(-1,self.num_continuous_latent_code), Q_mu, Q_var )*0.1


                #Experiment 1. gen_loss alone vs. gen_loss+discrete_loss
                #Net loss for generator

                #Net_Loss_Generator = gen_loss
                #Net_Loss_Generator = gen_loss + discrete_loss

                #Experiment 2. exploring latent code
                #Net loss for generator
                Net_Loss_Generator = gen_loss+discrete_loss+continuous_loss
                Net_Loss_Generator.backward()
                self.Gen_optim.step()

                #Progress check bar
                if batch_num!=0 and batch_num%100 == 0:
                    print('Epoch: [{}/{}] | Batch: [{}/{}] | Discriminator Loss: {:.2f} | Generator Loss: {:.2f}'.format(epoch+1, self.total_epoch, batch_num, len(self.data), Net_Loss_Discriminator.item(), Net_Loss_Generator.item()))

                Dis_losses.append(Net_Loss_Discriminator.item())
                Gen_losses.append(Net_Loss_Generator.item())

                #Calculate entropy
                # to calculate L1
                entropy_value = -1 * (discrete_one_hot.sum(0).div(curr_b_size).log() * discrete_one_hot.sum(0).div(curr_b_size)).sum()
                Total_Entropy.append(-discrete_loss.item() + entropy_value.cpu().detach().item())

            #Save Generated results for epoch = 1, epoch = 50, epoch = 100(max)
            if((epoch+1)==1 or (epoch+1)== (self.total_epoch/2) or (epoch+1)==self.total_epoch):
                with torch.no_grad():
                    generated_data = self.Gen(self.fixed_noise).detach().cpu()
                plt.figure(figsize=(10,10))
                plt.axis("off")
                plt.imshow(np.transpose(vutils.make_grid(generated_data, nrow = 10, padding = 2, normalize = True), (1,2,0)))
                plt.savefig("Epoch_{}_{}".format((epoch+1), self.dataset_name))
                plt.close('all')


        training_time = time.time() - begin_time
        print("----------------------------------------")
        print("Training Done (%.2fm)"%(training_time/60))
        print("----------------------------------------")

        # Save weights for quality evaluation later
        torch.save({
            'Gen': self.Gen.state_dict(),
            'Dis': self.Dis.state_dict(),
            'D_l': self.D_l.state_dict(),
            'Q_l': self.Q_l.state_dict(),
            'Dis_optim': self.Dis_optim.state_dict(),
            'Gen_optim': self.Gen_optim.state_dict(),
        }, 'checkpoint/model_final_{}'.format(self.dataset_name))

        #Save loss curve
        plt.figure(figsize=(10,5))
        plt.title("Loss Curves for Generator and Discriminator")
        plt.plot(Gen_losses, label = "Gen")
        plt.plot(Dis_losses, label = "Dis")
        plt.xlabel("Training Iterations")
        plt.ylabel("Loss")
        plt.ylim([0,8])
        plt.legend()
        plt.savefig("loss_curve_{}".format(self.dataset_name))

        # Save l1 curve --> try to use Epoch less than 5 (saturates before 1000 iterations)
        plt.figure(figsize=(10, 5))
        plt.title("L1 over iteration")
        plt.plot(Total_Entropy, label="GAN") # or InfoGAN
        plt.xlabel("Training Iterations")
        plt.ylabel("L1")
        plt.legend()
        plt.ylim([-0.2, 2.5])

        plt.savefig("l1_{}".format("MNIST"))


