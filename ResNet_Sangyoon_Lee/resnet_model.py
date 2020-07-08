import torch.nn as nn
import torch.nn.functional as functions

class cifar_model(nn.Module):
    def __init__(self, n_value, num_class = 10): #num_class = 10 for CIFAR 10
        super(cifar_model, self).__init__()
        self.n_value = n_value

        # Initial layer - first layer where input passes through
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,stride=1,padding=1,bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.ReLu = nn.ReLU(inplace=True)

        #Three kinds of Output Feature Map
        #1 - 32 x 32 x channel (16)
        self.first_map_2n = self.stack(n_value, residual_block, 16, 16, stride=1)
        #2 - 16 x 16 x channel (32)
        self.second_map_2n = self.stack(n_value, residual_block, 16, 32, stride=2)
        #3 - 8 x 8 x channel (64)
        self.third_map_2n = self.stack(n_value, residual_block, 32, 64, stride=2)

        # Final Output with a Global Average Pooling, a 10-way fully-connected layers, and softmax
        self.average_pooling = nn.AvgPool2d(8, stride = 1)
        self.fully_connected = nn.Linear(64, num_class) # num_class = 10 categories

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode = 'fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    # repetition of same modules
    def stack(self, n_value, block, input_ch, output_ch, stride):
        if stride == 2:
            down_sample = True
        else:
            down_sample = False

        layers = nn.ModuleList(
            [block(input_ch, output_ch, stride, down_sample)])

        for i in range(n_value - 1):
            layers.append(block(output_ch, output_ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial
        x = self.conv_layer_1(x)
        x = self.batch_norm_1(x)
        x = self.ReLu(x)

        # Feature Map Output
        x = self.first_map_2n(x)
        x = self.second_map_2n(x)
        x = self.third_map_2n(x)

        # Final Output layers
        x = self.average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        return x

# 2n layer one (
class residual_block(nn.Module):
    def __init__(self, input_ch, output_ch, stride = 1, down_sample = False):
        super(residual_block, self).__init__()
        # First conv
        self.conv_layer_1 = nn.Conv2d(input_ch, output_ch, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.batch_norm_1 = nn.BatchNorm2d(output_ch)
        self.ReLu = nn.ReLU(inplace = True)

        # Second conv
        self.conv_layer_2 = nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(output_ch)
        self.stride = stride

        self.max_pool = nn.MaxPool2d(1, stride = stride)
        # Case for reducing size or not
        if down_sample:
            self.reduce = down_sample
            self.channel = output_ch - input_ch # for zero padding
        else:
            self.reduce = down_sample

    def forward(self, x):
        # Keep initial input as it is, in order to make a shortcut later
        identity_mapping= x

        # First layer
        output = self.conv_layer_1(x)
        output = self.batch_norm_1(output)
        output = self.ReLu(output)

        # Second layer
        output = self.conv_layer_2(output)
        output = self.batch_norm_2(output)

        # reduce x with zero identity mapping to add with output properly for stride = 2
        if self.reduce:
            temp = functions.pad(x, (0,0,0,0,0, self.channel))
            identity_mapping = self.max_pool(temp)

        output += identity_mapping
        output = self.ReLu(output)

        return output


def deeper_resnet_cifar():
    return cifar_model(18) # Depth of 110 (6n + 2)

def deep_resnet_cifar():
    return cifar_model(9) # Depth of 56 (6n + 2)

def resnet_cifar():
    return cifar_model(5) # Depth of 32 (6n + 2)

def shallow_resnet_cifar():
    return cifar_model(3) # Depth of 20 (6n + 2)