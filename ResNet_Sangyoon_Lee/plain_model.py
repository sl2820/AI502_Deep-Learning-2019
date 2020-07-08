import torch.nn as nn

class cifar_model(nn.Module):
    def __init__(self, n_value, num_class = 10): #num_class = 10 for CIFAR 10
        super(cifar_model, self).__init__()
        self.n_value = n_value

        # Initial layer - first layer where input passes through
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,stride=1,padding=1,bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.ReLu = nn.ReLU(inplace=True)

        #Three kinds of Output Feature Map (without residual connection) simple repeated blocks
        #1 - 32 x 32 x channel (16)
        self.first_map_2n = self.stack(n_value, plain_block, 16, 16, stride=1)
        #2 - 16 x 16 x channel (32)
        self.second_map_2n = self.stack(n_value, plain_block, 16, 32, stride=2)
        #3 - 8 x 8 x channel (64)
        self.third_map_2n = self.stack(n_value, plain_block, 32, 64, stride=2)

        # Final Output with a Global Average Pooling, a 10-way fully-connected layers, and softmax
        self.average_pooling = nn.AvgPool2d(8, stride = 1)
        self.fully_connected = nn.Linear(64, num_class) # num_class = 10 categories

        for layers in self.modules():
            if isinstance(layers, nn.Conv2d):
                nn.init.kaiming_normal_(layers.weight, mode = 'fan_out', nonlinearity='relu')
            elif isinstance(layers, nn.BatchNorm2d):
                nn.init.constant_(layers.weight, 1)
                nn.init.constant_(layers.bias, 0)

    def stack(self, n_value, block, input_ch, output_ch, stride):
        layers = nn.ModuleList(
            [block(input_ch, output_ch, stride)])
        for i in range(n_value - 1):
            layers.append(block(output_ch, output_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial
        x = self.conv_layer_1(x)
        x = self.batch_norm_1(x)
        x = self.ReLu(x)

        # Feature Map Output 2n repeated blocks
        x = self.first_map_2n(x)
        x = self.second_map_2n(x)
        x = self.third_map_2n(x)


        # Final Output layers
        x = self.average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        return x

# 2 layer one (
class plain_block(nn.Module):
    def __init__(self, input_ch, output_ch, stride = 1):
        super(plain_block, self).__init__()
        # First conv (convolutional downsampling with stride = 2 if necessary)
        self.conv_layer_1 = nn.Conv2d(input_ch, output_ch, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.batch_norm_1 = nn.BatchNorm2d(output_ch)
        self.ReLu = nn.ReLU(inplace = True)

        # Second conv
        self.conv_layer_2 = nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(output_ch)
        # self.stride = stride

    def forward(self, x):
        # Keep initial input as it is, in order to make a shortcut later

        # First layer
        output = self.conv_layer_1(x)
        output = self.batch_norm_1(output)
        output = self.ReLu(output)

        # Second layer
        output = self.conv_layer_2(output)
        output = self.batch_norm_2(output)
        output = self.ReLu(output)

        return output


def plain_cifar():
    return cifar_model(5) # Depth of 32 (6n + 2)

def deeper_plain_cifar():
    return cifar_model(9) # Depth of 56 (6n + 2)

