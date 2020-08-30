import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        i_yi = range(x_scores.size(0)),y
        S_yi = x_scores[i_yi]        
        margin_loss_matrix = x_scores - S_yi.reshape(-1,1) + self.delta
        margin_loss_matrix[i_yi] -= self.delta #remove the equals times
        hinge_matrix =  torch.clamp(margin_loss_matrix,min=0)
        loss = hinge_matrix.sum()/x.size(0)
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["x"] = x
        self.grad_ctx["y"] = y
        self.grad_ctx["loss"] = hinge_matrix
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        N = self.grad_ctx["x"].size(0)
        G = torch.zeros(self.grad_ctx["loss"].shape)
        G[self.grad_ctx["loss"]>0] = 1 #W_j
        i_yi = range(N),self.grad_ctx["y"]
        G[i_yi] = -G.sum(dim=1) #get the indector of (M[i,j]>0)
        
        grad = torch.mm(self.grad_ctx["x"].t(),G)/N

        # ========================

        return grad
