import time
import torch
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from model import Model, Phi
from utils import *
from data import *
from BaseClass import BaseClass

class Classic(BaseClass):
    def __init__(self,
                 save_path=None,
                 n_epochs = 50,
                 batch_size=64,
                 lr=0.001,
                 checkpoint_save = 1,
                 checkpoint_test=5,
                 training_data_ratio=0.8,
                 labeled_data_ratio=1,
                 ):
        '''
        :param batch_size: batch_size for training data
        :param training_data_ratio: training data ratio
        '''
        BaseClass.__init__(
            self,
            save_path=save_path,
            n_epochs = n_epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_save = checkpoint_save,
            checkpoint_test= checkpoint_test,
        )
        self.data_loader(labeled_data_ratio, training_data_ratio)

    def data_loader(self, labeled_data_ratio, training_data_ratio):
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.train_val,
        ) = data_loaders(self.batch_size,
                         labeled_data_ratio =labeled_data_ratio,
                         training_data_ratio=training_data_ratio,
                         without_unlabeled=True)
        pass

    def training(self):
        t0 = time.time()
        if self.on_cuda:
            print('training on GPU')
        else:
            print('training on CPU')
        optimizer = optim.Adam(self.Net.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            training_loss = 0
            for local_X, local_y in iter(self.train_loader):
                if self.on_cuda:
                    local_X = local_X.to('cuda')
                    local_y = local_y.to('cuda')

                prediction = self.Net(local_X)
                loss = cross_entropy(prediction, local_y, one_hot=False)
                training_loss += loss.data.item()

                # gradient descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            training_loss /= len(self.train_loader)
            self.training_losses.append(training_loss)
            # evaluation
            self.evaluate()

            current_time = get_duration(t0, time.time())
            print(f'epoch {epoch+1} --- training_loss = {training_loss} --- val_loss = {self.val_losses[-1]} -- val_accuracy = {self.val_accuracies[-1]}%'
                  f'--- time: {current_time}')
            if self.save_path is not None and (epoch+1)%self.checkpoint_save==0:
                torch.save(self.Net.state_dict(), self.save_path)
            if (epoch+1)%self.checkpoint_test==0:
                self.testing()
            torch.cuda.empty_cache()

        self.save_losses(Classic=True)
        self.plot_results(Classic=True)
