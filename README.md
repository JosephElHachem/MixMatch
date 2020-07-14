# ABSTRACT
This code contains an implementation of the MixUp and MixMatch papers.

MixUp is a model training method that aims to obtain a smooth mapping function from the training set to the label set.
To do so, the mode is trained on interpolations of two input samples, the
new label being the interpolations of their one-hot-labels.

MixMatch is a semi-supervised training method that combines classic strategies of
semi-supervised training methods incorporating the MixUp training loop.

As a reference model, we aslo train the models in a classic way, implemented inside the Classic class.

**To train:**

# PARAMETERS
- n_epochs=20    		    -> number of epochs for training.
- save_path=None		    -> path to save model and and losses. If None, saved in current directory
- batch_size=64             -> batch size for training data
- batch_size_l=64           -> batch size for labeled training data (MixMatch only)
- batch_size_u=None         -> batch size for unlabeled training data (MixMatch only)
- checkpoint_save=1	        -> save model parameters each "checkpoint_save" epochs, inside the "save_path"
- learning_rate=1e-3    	-> learning rate for learning 
- checkpoint_save=1000      -> interval of epochs in for saving model
- checkpoint_test=1000      -> interval of epochs in for testing model
- K=7                       -> number of unlabeled augmentations to do
- alpha=0.2                 -> parameter for Beta distribution when sampling for MixUp training
- temperature=0.8           -> parameter for sharpening distribution on unlabeled predictions
- ULoss_factor=750          -> factor for unlabeled loss in total loss. Labaled loss factor is 1
- training_data_ratio=0.8   -> training data ratio vs validation data
- labeled_data_ratio=0.1    -> labeled data ratio vs unlabeled data ratio (MixMatch only)

### LOSS FUNCTION
Cross entropy loss is used for labeled data, while MSE loss is used for classification on unlabeled data, comparing the one hot vectors

### EVALUATION

### HYPERPARAMETERS CHOICE

# REFERENCES

* MixMatch: A Holistic Approach to Semi-Supervised Learning; arXiv:1905.02249v2
* MixUp: Beyond Empirical Risk Minimization; arXiv:1710.09412v2
