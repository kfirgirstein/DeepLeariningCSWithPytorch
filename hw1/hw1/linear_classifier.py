import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.empty(n_features ,n_classes).normal_(mean=0,std=weight_std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.mm(x,self.weights)
        y_pred = torch.argmax(class_scores,dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = ((y == y_pred).sum()).item() / y.numel()
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            loss_avg = 0 
            accuracy_avg = 0
            batch_count = len(dl_train)/dl_train.batch_size
            if (len(dl_train)%dl_train.batch_size!=0): batch_count+=1 #batch numers
            #trains epoch step, update weights          
            for X, y in dl_train:
                y_pred, class_scorces = self.predict(X)
                accuracy_avg+=self.evaluate_accuracy(y,y_pred)
                #calculate loss with regularization
                loss = loss_fn(X,y,class_scorces,y_pred)
                regularization = 0.5*weight_decay*torch.norm(self.weights,p=2)
                loss_avg+=loss_fn(X,y,class_scorces,y_pred)+regularization
                #update weights
                grad = loss_fn.grad()
                self.weights -= learn_rate*(grad + weight_decay*self.weights) 
            
            train_res.accuracy.append(accuracy_avg/batch_count)
            train_res.loss.append(loss_avg/batch_count)
            
            #validation epoch step            
            loss_avg = 0 
            accuracy_avg = 0
            batch_count = len(dl_valid)/dl_valid.batch_size
            if (len(dl_valid)%dl_valid.batch_size!=0): batch_count+=1 #batch numers
            
            for X,y in dl_valid:
                y_pred, class_scorces = self.predict(X)
                accuracy_avg+=self.evaluate_accuracy(y,y_pred)
                #calculate loss with regularization
                regularization = 0.5*weight_decay*torch.norm(self.weights,p=2)
                loss_avg+=loss_fn(X,y,class_scorces,y_pred)+regularization
                
            valid_res.accuracy.append(accuracy_avg/batch_count)
            valid_res.loss.append(loss_avg/batch_count)
            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        images = self.weights
        if(has_bias):
            images = self.weights[1:,:]
        w_images = images.t().reshape(self.n_classes,*img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0., learn_rate=0., weight_decay=0.)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp["learn_rate"]=0.01
    hp["weight_decay"] = 0.01
    hp["weight_std"] = 0.002
    # ========================

    return hp
