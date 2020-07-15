import os
from Classic import Classic
from MixUp import MixUp
from MixMatch import MixMatch

'''
The goal of this experiment is to compare the performance (test_losses and test_accuracies)
of the three models with little labeled data. We only choose to use 100 labeled images because
our network is small and does not have a large number of parameters.
Hyper-parameters:
n_epochs = 50
lr = 0.001
batch_size = 34
alpha = 0.75
batch_size_u = 34
K = 2
temperature = 0.5
ULoss_factor = 75
labeled_data_ratio = 0.016668
training_data_ratio = 0.25  # to have 250 labeled images and 750 validation images
'''
if __name__ == '__main__':
    n_epochs = 50
    lr = 0.001
    batch_size = 34
    alpha = 0.75
    batch_size_u = 34
    K = 2
    temperature = 0.5
    ULoss_factor = 75
    labeled_data_ratio = 0.016668
    training_data_ratio = 0.25  # to have 250 labeled images and 750 validation images

    # 250 labeled images
    names = ['250', '500', '1000']
    factor = [1, 2, 4]
    for name, factor in zip(names, factor):
        classic_model = Classic(
            save_path=os.path.join('results1', 'classic', name),
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_save=10,
            checkpoint_test=10,
            labeled_data_ratio=0.016668*factor,
            training_data_ratio=0.25,
        )

        classic_model.training()

        mixup_model = MixUp(
            save_path=os.path.join('results1', 'mixup', name),
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_save=10,
            checkpoint_test=10,
            labeled_data_ratio=0.016668 * factor,
            training_data_ratio=0.25
        )
        mixup_model.training()

        mixmatch_model = MixMatch(
            save_path=os.path.join('results1', 'mixmatch', name),
            n_epochs=n_epochs,
            lr=lr,
            batch_size_l=batch_size,
            batch_size_u=batch_size,
            K=K,
            temperature=temperature,
            ULoss_factor=ULoss_factor,
            checkpoint_save=10,
            checkpoint_test=10,
            alpha=alpha,
            labeled_data_ratio=0.016668 * factor,
            training_data_ratio=0.25
        )
        mixmatch_model.training()
