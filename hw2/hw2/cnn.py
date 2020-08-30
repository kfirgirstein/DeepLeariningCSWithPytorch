import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        for i in range(len(self.channels)):
            layers.append(nn.Conv2d(in_channels, self.channels[i], 3, padding = 1))
            in_channels = self.channels[i]
            layers.append(nn.ReLU())
            if ((i+1)%self.pool_every == 0):
                layers.append(nn.MaxPool2d(2))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        div_param = 2 ** (len(self.channels) // self.pool_every)
        layers.append(nn.Linear(int((in_h/div_param) * (in_w/div_param) * self.channels[-1]), self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        #print("channels:" + str(channels))
        #print("kernel_sizes:" + str(kernel_sizes))
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order). Should end with a
        #    final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use. This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        main_path_layers, shortcut_path_layers = [], []
        
        in_c = in_channels
        for i in range(len(channels) - 1):
            main_path_layers.append(nn.Conv2d(in_c, channels[i], kernel_sizes[i], padding = (kernel_sizes[i] // 2)))
            in_c = channels[i]
            
            if (batchnorm):
                main_path_layers.append(nn.BatchNorm2d(in_c))
            if (dropout):
                main_path_layers.append(nn.Dropout(dropout))
                
            main_path_layers.append(nn.ReLU())
        main_path_layers.append(nn.Conv2d(in_c, channels[-1], kernel_sizes[-1], padding = (kernel_sizes[-1] // 2)))
        
        if(in_channels != channels[-1]):
            shortcut_path_layers.append(nn.Conv2d(in_channels, channels[-1], 1, bias = False))
        
        self.main_path = nn.Sequential(*main_path_layers)
        self.shortcut_path = nn.Sequential(*shortcut_path_layers)    
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ReLUs (with a skip over them) should exist at the end,
        #    without a MaxPool after them.
        #  - Use your ResidualBlock implemetation.
        # ====== YOUR CODE: ======
        for i in range(len(self.channels) // self.pool_every):
            layers.append(ResidualBlock(in_channels, self.channels[i*self.pool_every:(i+1)*self.pool_every],[3] * self.pool_every))
            in_channels = self.channels[(i+1) * self.pool_every - 1]
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
        if(len(self.channels) % self.pool_every != 0):
            layers.append(ResidualBlock(in_channels, self.channels[-(len(self.channels)%self.pool_every):],[3] * (len(self.channels)%self.pool_every)))
        layers.append(nn.ReLU())
            
        
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)
        
    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        self.dropout = 0.3
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []
        for i in range(len(self.channels) // self.pool_every):
            filters = self.channels[i*self.pool_every:(i+1)*self.pool_every]
            layers.append(ResidualBlock(in_channels, filters,[3] * self.pool_every,batchnorm=True, dropout=self.dropout))
            in_channels = self.channels[(i+1) * self.pool_every - 1]
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool2d(2))
        if(len(self.channels) % self.pool_every != 0):
            layers.append(ResidualBlock(in_channels, self.channels[-(len(self.channels)%self.pool_every):],[3] * (len(self.channels)%self.pool_every),batchnorm=True, dropout=self.dropout))
        layers.append(nn.ReLU())
        seq = nn.Sequential(*layers)
        return seq
    # ========================
