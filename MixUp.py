import time
import torch.optim as optim
from data import *
from utils import *
from BaseClass import BaseClass

class MixUp(BaseClass):
    def __init__(
            self,
            save_path=None,
            n_classes = 10,
            dataset='mnist',
            n_epochs = 50,
            batch_size=64,
            lr=0.001,
            checkpoint_save = 1,
            checkpoint_test=5,
            alpha = 0.8,
            labeled_data_ratio=1,
            training_data_ratio=0.8
    ):
        '''
        :param batch_size: batch_size for training data
        :param alpha: Beta distribution parameter, used for MatchUp training
        :param training_data_ratio: training data ratio
        '''
        BaseClass.__init__(
            self,
            save_path=save_path,
            n_classes = n_classes,
            dataset=dataset,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_save=checkpoint_save,
            checkpoint_test=checkpoint_test,
        )
        self.alpha = alpha
        self.data_loader(labeled_data_ratio, training_data_ratio, dataset=dataset)

    def data_loader(self, labeled_data_ratio, training_data_ratio, dataset='mnist'):
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.train_val,
        ) = data_loaders(self.batch_size,
                         dataset=dataset,
                         K=1,
                         labeled_data_ratio=labeled_data_ratio,
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
            for (local_X1, local_y1), (local_X2, local_y2) in zip(iter(self.train_loader), iter(self.train_loader)):
                local_y1 = self.make_one_hot(local_y1)
                local_y2 = self.make_one_hot(local_y2)

                lmbda = np.long(np.random.beta(self.alpha, self.alpha))
                local_X = lmbda * local_X1 + (1 - lmbda) * local_X2
                local_y = lmbda * local_y1 + (1 - lmbda) * local_y2
                if self.on_cuda:
                    local_X = local_X.to('cuda')
                    local_y = local_y.to('cuda')

                prediction = self.Net(local_X)
                loss = cross_entropy(prediction, local_y, )
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

            if self.save_path is not None and (epoch+1)%self.checkpoint_save == 0:
                model_location = os.path.join(self.save_path, f'MixUp_{epoch+1}.pth')
                torch.save(self.Net.state_dict(), model_location)

            if (epoch+1)%self.checkpoint_test==0:
                self.testing(epoch)
            torch.cuda.empty_cache()

        self.testing(epoch)
        self.save_losses(MixUp=True)
        self.plot_results(MixUp=True)
