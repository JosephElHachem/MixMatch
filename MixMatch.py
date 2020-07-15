import time
import copy
import torch.optim as optim
from GPUtil import showUtilization as gpu_usage
from data import *
from utils import *
from BaseClass import BaseClass



class MixMatch(BaseClass):
    def __init__(self,
        save_path=None,
        n_epochs = 50,
        lr=0.001,
        batch_size_l=64,
        batch_size_u=None,
        K=2,
        temperature=0.5,
        ULoss_factor=100,
        checkpoint_save = 1,
        checkpoint_test=5,
        alpha = 0.75,
        labeled_data_ratio=0.1,
        training_data_ratio=0.8
        ):
        '''
        :param batch_size_l: batch_size for labeled data
        :param batch_size_u: batch_size for unlabeled data
        :param K: number of augmentations for unlabeled data
        :param temperature: parameter for distribution sharpening for unlabeled labels generation
        :param ULoss_factor: factor for unlabeled loss in total loss. Labeled loss factor is set to 1
        :param alpha: Beta distribution parameter, used for MatchUp training
        :param labeled_data_ratio: labeled data ratio
        :param training_data_ratio: training data ratio
        '''
        BaseClass.__init__(
            self,
            save_path=save_path,
            n_epochs=n_epochs,
            batch_size=batch_size_l,
            lr=lr,
            checkpoint_save=checkpoint_save,
            checkpoint_test=checkpoint_test
        )
        self.batch_size_l = batch_size_l
        self.batch_size_u = batch_size_u
        self.K = K
        self.temperature = temperature
        self.ULoss_factor = ULoss_factor
        self.alpha = alpha
        self.data_loader(labeled_data_ratio, training_data_ratio)

    def sharpen(self, predictions_Us):
        predictions_Us = predictions_Us**(1./self.temperature)
        predictions_Us /= torch.sum(predictions_Us, dim=1).view(-1,1)
        return predictions_Us

    def concatenate_shuffle(self, local_Us, local_X, one_hot_y, predictions_Us):
        Ws = copy.deepcopy(local_X)
        Labels = copy.deepcopy(one_hot_y)
        for i in range(self.K):
            Ws = torch.cat((Ws, local_Us[i]), axis=0)
            Labels = torch.cat((Labels, predictions_Us), axis=0)
        shuffled_idx = torch.randperm(Ws.shape[0])
        Ws, Labels = Ws[shuffled_idx], Labels[shuffled_idx]  # random shuffle
        return Labels, Ws

    def prediction_unlabeled(self, local_Us):
        predictions_Us = torch.zeros((local_Us[0].shape[0], self.n_classes), dtype=torch.float)
        if self.on_cuda:
            predictions_Us = predictions_Us.to('cuda')
        for i in range(self.K):
            predictions_Us += self.softmax(self.Net(local_Us[i]), dim=1)
        predictions_Us /= self.K
        predictions_Us = self.sharpen(predictions_Us)
        return predictions_Us


    def data_loader(self, labeled_data_ratio, training_data_ratio):
        (
            self.train_loader,
            self.unlabeled_loaders,
            self.val_loader,
            self.test_loader,
            self.train_val,
            self.batch_size_u
        ) = data_loaders(self.batch_size_l,
                         self.K,
                         batch_size_u=self.batch_size_u,
                         labeled_data_ratio=labeled_data_ratio,
                         training_data_ratio=training_data_ratio)
        pass

    def training(self):
        if self.on_cuda:
            print('training on GPU')
        else:
            print('training on CPU')
        optimizer = optim.Adam(self.Net.parameters(), lr=self.lr)

        t0 = time.time()

        idx = 0
        print('begining training')
        for epoch in range(self.n_epochs):
            self.ULoss_factor = min(self.ULoss_factor+3, 400)
            labeled_loss_epoch = 0 # avg cross entropy loss
            unlabeled_loss_epoch = 0 # avg L2**2 loss

            iter_unlabeled_loaders = [iter(loader) for loader in self.unlabeled_loaders]
            for local_X, local_y in self.train_loader:
                idx+=1
                one_hot_y = self.make_one_hot(local_y)
                if self.on_cuda:
                    local_X = local_X.to('cuda')
                    one_hot_y = one_hot_y.to('cuda')
                    local_Us = [next(loader)[0].to('cuda') for loader in iter_unlabeled_loaders]
                else:
                    local_Us = [next(loader)[0] for loader in iter_unlabeled_loaders]
                predictions_Us = self.prediction_unlabeled(local_Us)
                Labels, Ws = self.concatenate_shuffle(local_Us, local_X, one_hot_y, predictions_Us)

                # MixUp on local_X and random batch from Ws
                lmbda = np.random.beta(self.alpha, self.alpha)
                local_X_W = lmbda * local_X + (1 - lmbda) * Ws[:len(local_X)]
                local_y_W = lmbda * one_hot_y + (1 - lmbda) * Labels[:len(local_X)]
                # prediction and gradient step
                prediction = self.Net(local_X_W)
                loss_X = cross_entropy(prediction, local_y_W) # mean
                labeled_loss_epoch += float(loss_X)
                # MixUp on local_Us and remaining random batches from Ws
                loss_U = 0
                for i in range(self.K):
                    lmbda = np.long(np.random.beta(self.alpha, self.alpha))
                    local_U_W = lmbda * local_Us[i] + (1 - lmbda) * Ws[len(local_X):][i*self.batch_size_u:(i+1)*self.batch_size_u]
                    local_y_W = lmbda * predictions_Us + (1 - lmbda) * Labels[len(local_X):][i*self.batch_size_u:(i+1)*self.batch_size_u]
                    prediction = self.softmax(self.Net(local_U_W), dim=1)
                    loss_U += self.MSELoss(prediction, local_y_W)
                loss_U /= (Ws.shape[0] - self.batch_size_l) * self.n_classes
                unlabeled_loss_epoch += float(loss_U)
                # gradient descent
                batch_loss = loss_X + self.ULoss_factor * loss_U
                if idx % 50 == 0:
                    print(f"batch_loss: {batch_loss} -- loss_X: {int(100*(loss_X/batch_loss).item())}% -- loss_U: {int(100*(loss_U*self.ULoss_factor/batch_loss).item())}%")
                    gpu_usage()
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                del loss_X, loss_U, prediction, batch_loss
                if self.on_cuda:
                    torch.cuda.empty_cache()
            if self.save_path is not None and (epoch+1)%self.checkpoint_save == 0:
                torch.save(self.Net.state_dict(), f'MixMatch_{epoch+1}.pth')
            labeled_loss_epoch /= len(self.train_loader)
            self.l_training_losses.append(float(labeled_loss_epoch))
            self.u_training_losses.append(float(self.ULoss_factor * unlabeled_loss_epoch))
            accuracy, val_loss = self.evaluate()
            # timing
            current_time = get_duration(t0, time.time())
            print(f'epoch {epoch+1} --- l_train_loss = {labeled_loss_epoch}  -- u_train_loss = {self.ULoss_factor * unlabeled_loss_epoch} --- val_loss = {val_loss} -- val_accuracy = {accuracy}%'
                  f'--- time: {current_time}')
            del predictions_Us, local_X, local_y, local_Us, Ws, Labels, local_X_W, local_y_W, local_U_W, val_loss
            if self.on_cuda:
                torch.cuda.empty_cache()
            gpu_usage()
            # testing
            if (epoch+1) % self.checkpoint_test == 0:
                self.testing(epoch)
        # accuracy
        self.testing(epoch)
        self.save_losses(MixMatch=True)
        self.plot_results(MixMatch=True)
