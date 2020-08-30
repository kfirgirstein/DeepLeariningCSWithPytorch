import abc
import torch


class Block(abc.ABC):
    """
    A block is some computation element in a network architecture which
    supports automatic differentiation using forward and backward functions.
    """

    def __init__(self):
        # Store intermediate values needed to compute gradients in this hash
        self.grad_cache = {}
        self.training_mode = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the forward pass of the block.
        :param args: The computation arguments (implementation specific).
        :return: The result of the computation.
        """
        pass

    @abc.abstractmethod
    def backward(self, dout):
        """
        Computes the backward pass of the block, i.e. the gradient
        calculation of the final network output with respect to each of the
        parameters of the forward function.
        :param dout: The gradient of the network with respect to the
        output of this block.
        :return: A tuple with the same number of elements as the parameters of
        the forward function. Each element will be the gradient of the
        network output with respect to that parameter.
        """
        pass

    @abc.abstractmethod
    def params(self):
        """
        :return: Block's trainable parameters and their gradients as a list
        of tuples, each tuple containing a tensor and it's corresponding
        gradient tensor.
        """
        pass

    def train(self, training_mode=True):
        """
        Changes the mode of this block between training and evaluation (test)
        mode. Some blocks have different behaviour depending on mode.
        :param training_mode: True: set the model in training mode. False: set
        evaluation mode.
        """
        self.training_mode = training_mode


class Linear(Block):
    """
    Fully-connected linear layer.
    """

    def __init__(self, in_features, out_features, wstd=0.1):
        """
        :param in_features: Number of input features (Din)
        :param out_features: Number of output features (Dout)
        :wstd: standard deviation of the initial weights matrix
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # TODO: Create the weight matrix (w) and bias vector (b).
        # ====== YOUR CODE: ======
        self.b = torch.ones(out_features)
        self.w = torch.zeros(out_features, in_features).normal_(mean = 0, std = wstd)
        # ========================

        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def params(self):
        return [
            (self.w, self.dw), (self.b, self.db)
        ]

    def forward(self, x, **kw):
        """
        Computes an affine transform, y = x W^T + b.
        :param x: Input tensor of shape (N,Din) where N is the batch
        dimension, and Din is the number of input features, or of shape
        (N,d1,d2,...,dN) where Din = d1*d2*...*dN.
        :return: Affine transform of each sample in x.
        """

        x = x.reshape((x.shape[0], -1))

        # TODO: Compute the affine transform
        # ====== YOUR CODE: ======
        out = x.matmul(self.w.t()) + self.b
        # ========================

        self.grad_cache['x'] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, Dout).
        :return: Gradient with respect to block input, shape (N, Din)
        """
        x = self.grad_cache['x']

        # TODO: Compute
        #   - dx, the gradient of the loss with respect to x
        #   - dw, the gradient of the loss with respect to w
        #   - db, the gradient of the loss with respect to b
        #  You should accumulate gradients in dw and db.
        # ====== YOUR CODE: ======
        self.db += dout.sum(dim=0) #sumarize each row
        self.dw +=dout.t().mm(x)
        dx = dout.mm(self.w)
        # ========================

        return dx

    def __repr__(self):
        return f'Linear({self.in_features}, {self.out_features})'


class ReLU(Block):
    """
    Rectified linear unit.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes max(0, x).
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: ReLU of each sample in x.
        """

        # TODO: Implement the ReLU operation.
        # ====== YOUR CODE: ======
        out = x.clamp(min=0)
        # ========================

        self.grad_cache['x'] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, *).
        :return: Gradient with respect to block input, shape (N, *)
        """
        x = self.grad_cache['x']

        # TODO: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        dx = dout
        dx[x<0] = 0
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return 'ReLU'


class Sigmoid(Block):
    """
    Sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes s(x) = 1/(1+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """

        # TODO: Implement the Sigmoid function.
        #  Save whatever you need into
        #  grad_cache.
        # ====== YOUR CODE: ======
        out = 1/(1+torch.exp(-x))
        self.grad_cache['x'] = x
        # ========================
        
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, *).
        :return: Gradient with respect to block input, shape (N, *)
        """

        # TODO: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        x = self.grad_cache['x']
        sig =  1/(1+torch.exp(-x))
        dx = sig*(1-sig)*dout
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return 'Sigmoid'


class CrossEntropyLoss(Block):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Computes cross-entropy loss directly from class scores.
        Given class scores x, and a 1-hot encoding of the correct class yh,
        the cross entropy loss is defined as: -yh^T * log(softmax(x)).

        This implementation works directly with class scores (x) and labels
        (y), not softmax outputs or 1-hot encodings.

        :param x: Tensor of shape (N,D) where N is the batch
        dimension, and D is the number of features. Should contain class
        scores, NOT PROBABILITIES.
        :param y: Tensor of shape (N,) containing the ground truth label of
        each sample.
        :return: Cross entropy loss, as if we computed the softmax of the
        scores, encoded y as 1-hot and calculated cross-entropy by
        definition above. A scalar.
        """

        N = x.shape[0]
        xmax, _ = torch.max(x, dim=1, keepdim=True)
        x = x - xmax  # for numerical stability

        # TODO: Compute the cross entropy loss using the last formula from the
        #  notebook (i.e. directly using the class scores).
        #  Tip: to get a different column from each row of a matrix tensor m,
        #  you can index it with m[range(num_rows), list_of_cols].
        # ====== YOUR CODE: ======
        loss = (-x[range(N), y] + torch.log(torch.sum(x.exp(), dim = 1))).sum()/N       
        # ========================

        self.grad_cache['x'] = x
        self.grad_cache['y'] = y
        return loss

    def backward(self, dout=1.0):
        """
        :param dout: Gradient with respect to block output, a scalar which
        defaults to 1 since the output of forward is scalar.
        :return: Gradient with respect to block input (only x), shape (N,D)
        """
        x = self.grad_cache['x']
        y = self.grad_cache['y']
        N = x.shape[0]

        # TODO: Calculate the gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        exp_sum = x.exp().sum(dim = 1)
        q_mat = x.exp() / (exp_sum.reshape(-1,1).repeat( 1, x.shape[1]))
        q_mat[range(N), y] -= 1
        dx = dout * q_mat / N
        # ========================

        return dx

    def params(self):
        return []

class Dropout(Block):
    def __init__(self, p=0.5):
        """
        Initializes a Dropout block.
        :param p: Probability to drop an activation.
        """
        super().__init__()
        assert 0. <= p <= 1.
        self.p = p

    def forward(self, x, **kw):
        # TODO: Implement the dropout forward pass.
        #  Notice that contrary to previous blocks, this block behaves
        #  differently a according to the current training_mode (train/test).
        # ====== YOUR CODE: ======
        indexs = torch.multinomial(torch.tensor(list(range(x.shape[1])), dtype=torch.float), int(self.p * x.shape[1]))
        out = x
        if(self.training_mode):
            out[:, indexs] = 0
                    
        self.indexs_cache = indexs
        # ========================

        return out

    def backward(self, dout):
        # TODO: Implement the dropout backward pass.
        # ====== YOUR CODE: ======
        dx = dout
        if(self.training_mode):
            dx[:, self.indexs_cache] = 0
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return f'Dropout(p={self.p})'


class Sequential(Block):
    """
    A Block that passes input through a sequence of other blocks.
    """

    def __init__(self, *blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, **kw):
        out = None

        # TODO: Implement the forward pass by passing each block's output
        #  as the input of the next.
        # ====== YOUR CODE: ======
        out = x
        for block in self.blocks:
            out = block.forward(out,**kw)
        # ========================

        return out

    def backward(self, dout):
        din = None

        # TODO: Implement the backward pass.
        #  Each block's input gradient should be the previous block's output
        #  gradient. Behold the backpropagation algorithm in action!
        # ====== YOUR CODE: ======
        din = dout
        for block in reversed(self.blocks):
            din = block.backward(din)
        # ========================

        return din

    def params(self):
        params = []

        # TODO: Return the parameter tuples from all blocks.
        # ====== YOUR CODE: ======
        params = [param 
                  for block in self.blocks
                  for param in block.params()]
        # ========================

        return params

    def train(self, training_mode=True):
        for block in self.blocks:
            block.train(training_mode)

    def __repr__(self):
        res = 'Sequential\n'
        for i, block in enumerate(self.blocks):
            res += f'\t[{i}] {block}\n'
        return res

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, item):
        return self.blocks[item]


class MLP(Block):
    """
    A simple multilayer perceptron based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """

    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        NLS = {'relu': ReLU,'sigmoid': Sigmoid,'FC':Linear}
#        if(all(hidden_features)):
#            blocks.append(Linear(in_features,num_classes))
#        else:
        prev = in_features
        for h in hidden_features:
            blocks.append(Linear(prev,h,**kw))
            if(dropout>0 and activation == 'relu'):
                blocks.append(Dropout(dropout))
            blocks.append(NLS[activation]())
            prev = h
        blocks.append(NLS['FC'](hidden_features[-1],num_classes,**kw))
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'
