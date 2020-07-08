import torch
import torch.nn as nn
import torch.optim as optimizer
import statistics
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from resnet_model import resnet_cifar, deep_resnet_cifar, deeper_resnet_cifar, shallow_resnet_cifar
from plain_model import plain_cifar, deeper_plain_cifar

from tensorboardX import SummaryWriter

# In order to have 5 runs at single running code, I created it as a Runner class and initialize hyperparameters for each new run
class Runner(object):
    def __init__(self,model_name):
        self.learning_rate = 0.1
        self.batch_size = 128
        self.num_worker = 4
        self.log_directory = model_name+"_logs"   # for tensorboard
        self.model = model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transforms_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # given pixel mean and std
        ])
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.dataset_train = CIFAR10(root='../ResNet/data', train=True, download=True, transform=transforms_train)
        self.dataset_test = CIFAR10(root='../ResNet/data', train=False, download=True, transform=transforms_test)
        self.train_loader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
        self.test_loader = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)

        if self.model == "resnet":
            self.net = resnet_cifar()
        elif self.model == "plain":
            self.net = plain_cifar()
        elif self.model == "deeper_plain":
            self.net = deeper_plain_cifar()
        elif self.model == "deeper":
            self.net = deeper_resnet_cifar()
        elif self.model == "deep":
            self.net = deep_resnet_cifar()
        elif self.model == "shallow":
            self.net = shallow_resnet_cifar()

        else:
            self.net = None

        self.net = self.net.to(self.device)
        self.parameters= sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001)

        self.decay_iteration = [32000, 48000]  # divide by 10s
        self.step_lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay_iteration,
                                                     gamma=0.1)  # gamma is the learning rate decay --> divide by 10

        self.writer = SummaryWriter(self.log_directory)

    def learning(self, epoch, best_acc, iteration, type):
        curr_loss = 0
        correct = 0
        total = 0
        best_loss = 0
        if type == "train":
            self.net.train()
            for batch_num, (x, y) in enumerate(self.train_loader):
                iteration += 1
                self.step_lr_scheduler.step()
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.net(x)
                loss = self.criterion(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                curr_loss += loss.item()
                _, prediction = output.max(1)
                total += y.size(0)
                correct += prediction.eq(y).sum().item()


            acc = 100 * correct / total
            error = 100 - acc
            print('train epoch : {} | batch size : [{}/{}] | loss: {:.3f} | accuracy: {:.3f}'.format(
                epoch, batch_num + 1, len(self.train_loader), curr_loss / (batch_num + 1), acc))

            self.writer.add_scalar('log/train error', error, iteration)
            self.writer.add_scalar('log/train loss', curr_loss / (batch_num + 1), iteration)
            if acc > best_acc:
                best_acc = acc
                best_loss = curr_loss / (batch_num + 1)
            best_loss = curr_loss / (batch_num + 1)

        else:
            self.net.eval()
            with torch.no_grad():
                for batch_num, (x, y) in enumerate(self.test_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.net(x)
                    loss = self.criterion(output, y)

                    curr_loss += loss.item()
                    _, prediction = output.max(1)
                    total += y.size(0)
                    correct += prediction.eq(y).sum().item()

            acc = 100 * correct / total
            error = 100 - acc
            print('test epoch : {} | batch size : [{}/{}] | loss: {:.3f} | accuracy: {:.3f}'.format(
                epoch, batch_num + 1, len(self.test_loader), curr_loss / (batch_num + 1), acc))

            self.writer.add_scalar('log/test error', error, iteration)
            self.writer.add_scalar('log/test loss', curr_loss / (batch_num + 1), iteration)

            if acc > best_acc:
                best_acc = acc
                best_loss = curr_loss / (batch_num + 1)
            best_loss = curr_loss / (batch_num + 1)

        return iteration, best_acc, best_loss

    def run(self):
        test_best_acc = 0
        train_best_acc = 0
        epoch = 0
        iteration = 0
        train_best_loss = 0
        test_best_loss = 0

        # Because paper states that the learning stops by the iteration, not the specific epoch
        while iteration < 64000:
            epoch += 1
            iteration, train_best_acc, train_best_loss = self.learning(epoch, train_best_acc, iteration, "train")
            iteration, test_best_acc, test_best_loss = self.learning(epoch, test_best_acc, iteration, "test")
            print('best test accuracy is: ', test_best_acc,' | Current Iterations: ', iteration)

        self.writer.close()
        return test_best_acc, test_best_loss, train_best_acc, train_best_loss


if __name__ == "__main__":
    total_run = 5
    test_acc = []
    test_loss = []
    train_acc = []
    train_loss = []
    model_name = ""
    num_param = 0
    for i in range (total_run):
        run = Runner("resnet")# POSSIBLE MODEL ENTRY: plain, deeper_plain, shallow, resnet, deeper, deep
        model_name = run.model
        num_param = run.parameters
        test_best_acc, test_best_loss, train_best_acc, train_best_loss = run.run()

        test_acc.append(test_best_acc)
        test_loss.append(test_best_loss)
        train_acc.append(train_best_acc)
        train_loss.append(train_best_loss)




    print("")
    print(">>> MODEL: ",model_name," OUTPUT SUMMARY","<<<")
    print("Number of Parameters: ", num_param)
    print("====== Train Results ======")
    print(train_acc)
    print("Best Train Accuracy: %.4f | Avg Train Accuracy: %.4f | Standard Deviation: %.4f" % (max(train_acc), statistics.mean(train_acc), statistics.stdev(train_acc)))
    print(train_loss)
    print("Best Train Loss: %.4f | Avg Train Loss: %.4f | Standard Deviation: %.4f" % (min(train_loss), statistics.mean(train_loss), statistics.stdev(train_loss)))
    print("")
    print("====== Test Results ======")
    print(test_acc)
    print("Best Test Accuracy: %.4f | Avg Test Accuracy: %.4f | Standard Deviation: %.4f" % (max(test_acc), statistics.mean(test_acc), statistics.stdev(test_acc)))
    print(test_loss)
    print("Best Test Loss: %.4f | Avg Test Loss: %.4f | Standard Deviation: %.4f" % (min(test_loss), statistics.mean(test_loss), statistics.stdev(test_loss)))
