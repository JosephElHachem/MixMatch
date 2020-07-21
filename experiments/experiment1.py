import os, time
import pickle as pkl
import matplotlib.pyplot as plt
from utils import *
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
    start = time.time()
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
    names = ['250', '500', '1000', '2000']
    names = ['1000']

    factor = [1,2,4,8]
    factor = [4]
    for name, factor in zip(names, factor):
        print(f'doing {name}')
        classic_model = Classic(
            save_path=os.path.join('results2', name, 'classic'),
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_save=10,
            checkpoint_test=4,
            labeled_data_ratio=0.08334*factor,
            training_data_ratio=0.05,
        )
        classic_model.training()
        print('classic done')

        mixup_model = MixUp(
            save_path=os.path.join('results2', name, 'mixup'),
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_save=10,
            checkpoint_test=4,
            labeled_data_ratio=0.08334 * factor,
            training_data_ratio=0.05
        )
        mixup_model.training()
        print('mixup done')

        mixmatch_model = MixMatch(
            save_path=os.path.join('results2', name, 'mixmatch'),
            n_epochs=n_epochs,
            lr=lr,
            batch_size_l=batch_size,
            batch_size_u=batch_size,
            K=K,
            temperature=temperature,
            ULoss_factor=ULoss_factor,
            checkpoint_save=10,
            checkpoint_test=4,
            alpha=alpha,
            labeled_data_ratio=0.08334 * factor,
            training_data_ratio=0.05
        )
        mixmatch_model.training()
        print('mixmatch done\n\n')
    # plotting results
    models = ['classic', 'mixup', 'mixmatch']
    colors = ['b', 'r', 'k']
    folders = ['250', '500', '1000']#, '2000']
    results = {}
    for metric in ['accuracies', 'losses']:
        results[metric] = {}
        for folder in folders:
            results[metric][folder] = {}
            for model in models:
                results[metric][folder][model] = {}
                for experiment in ['val', 'test']:
                    path = os.path.join('results2', folder, model, experiment+'_'+metric+'.pkl')
                    with open(path, 'rb') as f:
                        results[metric][folder][model][experiment] = pkl.load(f)
                try:
                    path = os.path.join('results2', folder, model, 'test_epochs.pkl')
                    with open(path, 'rb') as f:
                        results[metric][folder][model]['test_epochs'] = pkl.load(f)
                except:
                    print('test_epochs not loaded')
    for folder in folders:
        make_path(os.path.join('results2', folder, 'results'))
        for metric in ['accuracies', 'losses']:
            plt.figure()
            plt.title(f'training with {folder} labels')
            plt.xlabel('epochs')
            for idx, model in enumerate(models):
                test_epochs = results[metric][folder][model]['test_epochs']
                if metric is 'accuracies':
                    plt.ylim(-0.2,100.2)
                plt.plot(results[metric][folder][model]['val'],label=f'{model}_val', c=colors[idx])
                plt.plot(test_epochs, results[metric][folder][model]['test'], '--', label=f'{model}_test', c=colors[idx])
                plt.legend()
            save_path = os.path.join('results2', folder, 'results', metric+'.png')
            plt.savefig(save_path)
            plt.close()
    # end = time.time()
    # print(f'total time: {get_duration(start, end)}')
