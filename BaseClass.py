import pickle
import matplotlib.pyplot as plt
from model import Model, Phi
from utils import *
from data import *


class BaseClass:
    def __init__(self,
        save_path=None,
        n_epochs = 50,
        batch_size=64,
        lr=0.001,
        checkpoint_save = 1,
        checkpoint_test=5,
        ):
        self.save_path = save_path
        make_path(self.save_path)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.checkpoint_save = checkpoint_save
        self.checkpoint_test = checkpoint_test
        self.on_cuda = torch.cuda.is_available()
        self.n_classes = 10
        self.init_model()
        self.init_losses()
        self.softmax = torch.nn.functional.softmax
        self.MSELoss = torch.nn.MSELoss(reduction='sum')

    def init_model(self):
        Phi_instance = Phi()
        self.Net = Model(Phi_instance)
        if self.on_cuda:
          self.Net.to('cuda')

    def init_losses(self):
        self.l_training_losses = []
        self.u_training_losses = []
        self.training_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.test_accuracies = []
        self.test_losses = []
        self.test_epochs = []
        self.all_epochs = []

    def make_one_hot(self, local_y):
        one_hot_y = torch.zeros((len(local_y), self.n_classes), dtype=torch.float)
        one_hot_y[torch.arange(len(local_y)), local_y] = 1.
        return one_hot_y

    def evaluate(self):
        val_images, val_labels = next(iter(self.val_loader))
        if self.on_cuda:
            val_images, val_labels = val_images.to('cuda'), val_labels.to('cuda')
        # evaluation on validation set
        with torch.no_grad():
            prediction = self.Net(val_images)
        val_loss = cross_entropy(prediction, val_labels, one_hot=False)
        predicted_labels = torch.argmax(prediction, dim=1)
        accuracy = np.round(100.0 * (predicted_labels == val_labels).sum().item() / self.train_val[1], 2)
        del val_images, val_labels
        try:
            self.val_accuracies.append(float(accuracy))
            self.val_losses.append(float(val_loss))
        except:
            pass
        return accuracy, val_loss

    def testing(self, epoch=0, model_path=None):
        if model_path is not None:
            self.Net.load_state_dict(torch.load(model_path))
        # accuracy
        test_images, test_labels = next(iter(self.test_loader))
        if self.on_cuda:
            self.Net = self.Net.to('cuda')
            test_images, test_labels = test_images.to('cuda'), test_labels.to('cuda')
        with torch.no_grad():
            prediction = self.Net(test_images)
            predicted_labels = torch.argmax(prediction, dim=1)
            accuracy = 100.0 * (predicted_labels == test_labels).sum().item() / len(test_labels)
            test_loss = cross_entropy(prediction, test_labels, one_hot=False)
            print(f'Test set loss: {test_loss} -- accuracy: {accuracy}%')
            self.test_accuracies.append(accuracy)
            self.test_epochs.append(epoch+1)
            self.test_losses.append(test_loss)
            del test_images, test_labels, prediction, test_loss
            if self.on_cuda:
                torch.cuda.empty_cache()

    def plot_results(self, Classic=False, MixUp=False, MixMatch=False):
        if not Classic and not MixUp and not MixMatch:
            raise ValueError('One of Classic, MixUp or MixMatch must be set True')
        if (Classic and MixUp) or (Classic and MixMatch) or (MixUp and MixMatch):
            raise ValueError('Only one of Classic, MixUp or MixMatch must be set True')


        plt.figure()
        plt.title('unlabeled losses')
        plt.xlabel('epochs')
        plt.plot(self.u_training_losses, label='unlabeled')
        if self.save_path is not None:
            path = os.path.join(self.save_path, 'unlabeled_losses.png')
        else:
            path = 'unlabeled_losses.png'
        plt.legend()
        plt.savefig(path)

        plt.figure()
        plt.title('losses')
        plt.xlabel('epochs')
        if MixMatch:
            plt.plot(self.l_training_losses, label='labeled')
        else:
            plt.plot(self.training_losses, label='training')
        plt.plot(self.val_losses, label='val')
        plt.plot(self.test_epochs, self.test_losses, label='test')
        plt.legend()
        if self.save_path is not None:
            path = os.path.join(self.save_path, 'losses.png')
        else:
            path = 'losses.png'
        plt.savefig(path)

        plt.figure()
        plt.xlabel('epochs')
        plt.plot(self.val_accuracies, label='val')
        plt.plot(self.test_epochs, self.test_accuracies, label='test')
        plt.legend()
        if self.save_path is not None:
            path = os.path.join(self.save_path, 'accuracies.png')
        else:
            path = 'accuracies.png'
        plt.savefig(path)

    def save_losses(self, Classic=False, MixUp=False, MixMatch=False):
        if not Classic and not MixUp and not MixMatch:
            raise ValueError('One of Classic, MixUp or MixMatch must be set True')
        if (Classic and MixUp) or (Classic and MixMatch) or (MixUp and MixMatch):
            raise ValueError('Only one of Classic, MixUp or MixMatch must be set True')
        if self.save_path is not None:
            current_dir = os.getcwd()
            os.chdir(self.save_path)

        if MixMatch:
            with open('l_training_losses.pkl', 'wb') as f:
                pickle.dump(self.l_training_losses, f)
            with open('u_training_losses.pkl', 'wb') as f:
                pickle.dump(self.u_training_losses, f)
        else:
            with open('training_losses.pkl', 'wb') as f:
                pickle.dump(self.training_losses, f)
        with open('val_losses.pkl', 'wb') as f:
            pickle.dump(self.val_losses, f)
        with open('test_losses.pkl', 'wb') as f:
            pickle.dump(self.test_losses, f)
        with open('val_accuracies.pkl', 'wb') as f:
            pickle.dump(self.val_accuracies, f)
        with open('test_accuracies.pkl', 'wb') as f:
            pickle.dump(self.test_accuracies, f)
        with open('test_epochs.pkl', 'wb') as f:
            pickle.dump(self.test_epochs, f)

        if self.save_path is not None:
            os.chdir(current_dir)

    def load_losses(self, Classic=False, MixUp=False, MixMatch=False, load_path=None):
        if not Classic and not MixUp and not MixMatch:
            raise ValueError('One of Classic, MixUp or MixMatch must be set True')
        if (Classic and MixUp) or (Classic and MixMatch) or (MixUp and MixMatch):
            raise ValueError('Only one of Classic, MixUp or MixMatch must be set True')
        if load_path is not None:
            current_dir = os.getcwd()
            os.chdir(load_path)

        if MixMatch:
            with open('l_training_losses.pkl', 'rb') as f:
                self.l_training_losses = pickle.load(f)
            with open('u_training_losses.pkl', 'rb') as f:
                self.u_training_losses = pickle.load(f)
        else:
            with open('training_losses.pkl', 'rb') as f:
                self.training_losses = pickle.load(f)

        with open('val_losses.pkl', 'rb') as f:
            self.val_losses = pickle.load(f)
        with open('test_losses.pkl', 'rb') as f:
            self.test_losses = pickle.load(f)
        with open('val_accuracies.pkl', 'rb') as f:
            self.val_accuracies = pickle.load(f)
        with open('test_accuracies.pkl', 'rb') as f:
            self.test_accuracies = pickle.load(f)
        with open('test_epochs.pkl', 'rb') as f:
            self.test_epochs = pickle.load(f)

        if load_path is not None:
            os.chdir(current_dir)



    def data_loader(self, labeled_data_ratio, training_data_ratio):
        raise NotImplementedError
        pass

    def training(self):
        raise NotImplementedError
        pass
