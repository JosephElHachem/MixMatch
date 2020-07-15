import argparse
from Classic import Classic
from MixUp import MixUp
from MixMatch import MixMatch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', help='saving directory. If not provided, saving will be in current directory', required=False)
    parser.add_argument('--n_epochs', help='number of epochs on training data', default=50, type=int)
    parser.add_argument('--checkpoint_save', help='interval of epochs in for saving model', default=5)
    parser.add_argument('--checkpoint_test', help='interval of epochs in for testing model', default=5)
    parser.add_argument('--batch_size',
                        help='batch size used for training data',
                        default=64)
    parser.add_argument('--batch_size_l',
                        help='batch size used for labeled data. Only used for MixMatch',
                        default=64)
    parser.add_argument('--batch_size_u',
                        help='batch size used for unlabeled data. Only used for MixMatch',
                        default=None)
    parser.add_argument('--lr',
                        help='learning rate',
                        default=0.001)
    parser.add_argument('--K',
                        help='number of unlabeled augmentations to do',
                        default=2)
    parser.add_argument('--temperature',
                        type=float,
                        help='parameter for sharpening distribution on unlabeled predictions',
                        default=0.5)
    parser.add_argument('--ULoss_factor',
                        help='factor for unlabeled loss in total loss. Labaled loss factor is 1.',
                        default=100)
    parser.add_argument('--alpha',
                        type=float,
                        help='parameter for Beta distribution when sampling for MixUp training',
                        default=0.75)
    parser.add_argument('--labeled_data_ratio',
                        type=float,
                        help='labeled data ratio vs unlabeled data ratio. Only for MixMatch training.',
                        default=0.1)
    parser.add_argument('--training_data_ratio',
                        type=float,
                        help='training data vs validation data ratio',
                        default=0.8)
    parser.add_argument('--experiment',
                        help='Experiment number to run. Will neglect all other parameters',
                        default=None)

    parser.add_argument('--model', help='choose one of: Classic, MatchUp or MixMatch ', type=str, required=True)
    opts = parser.parse_args()

    if opts.experiment is None:
        if opts.model == 'Classic':
            Classic_instance = Classic(
                save_path=opts.save_path,
                n_epochs=opts.n_epochs,
                batch_size=opts.batch_size,
                lr=opts.lr,
                checkpoint_save=opts.checkpoint_save,
                checkpoint_test=opts.checkpoint_test,
                training_data_ratio=opts.training_data_ratio)
            Classic_instance.training()
        elif opts.model == 'MatchUp':
            MixUp_instance = MixUp(
                save_path=opts.save_path,
                n_epochs=opts.n_epochs,
                batch_size=opts.batch_size,
                lr=opts.lr,
                alpha=opts.alpha,
                checkpoint_save=opts.checkpoint_save,
                checkpoint_test=opts.checkpoint_test,
                training_data_ratio=opts.training_data_ratio)
            MixUp_instance.training()
        elif opts.model == 'MixMatch':
            MixMatch_instance = MixMatch(
                n_epochs=opts.n_epochs,
                lr=opts.lr,
                batch_size_l=opts.batch_size_l,
                batch_size_u=opts.batch_size_u,
                K=opts.K,
                temperature=opts.temperature,
                ULoss_factor=opts.ULoss_factor,
                checkpoint_save=opts.checkpoint_save,
                checkpoint_test=opts.checkpoint_test,
                alpha=opts.alpha,
                labeled_data_ratio=opts.labeled_data_ratio,
                training_data_ratio=opts.training_data_ratio)
            MixMatch_instance.training()
        else:
            pass

    elif opts.experiment == 1:
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
        exec(open('experiments/experiment1.py').read())

    else:
        raise ValueError('experiment should be 1 or None')
