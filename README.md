# ABSTRACT
This code contains an implementation of the MatchUp and MixMatch papers.

MatchUp is a model training method that aims to obtain a smooth mapping function from the training set to the label set.
To do so, the mode is trained on interpolations of two input samples, the
new label being the interpolations of their one-hot-labels.

MixMatch is a semi-supervised training method that combines classic strategies of
semi-supervised training methods incorporating the MatchUp training loop.

As a reference model, we aslo train the models in a classic way, implemented inside the Classic class.

**To train:**

# PARAMETERS
- n_epochs=20    		-> number of epochs for training
- save_path=None		-> path to save model at the end of training and each checkpoint_save. Saves: hidden_layer.npy, output_layer.npy, w2id.pkl, id2w.pkl, vocab.pkl
- checkpoint_save=1	-> save model parameters each "checkpoint_save" epochs, inside the "save_path"
- learning_rate=1e-2	-> learning rate for learning
- checkpoint_save=1000		-> print average loss each "checkpoint" sentences
- checkpoint_test=1000

# ARCHITECTURE

### GENERAL ARCHITECTURE

### PREPROCESSING

 
### LOSS FUNCTION
Loss function used is the logsoftmax

### EVALUATION
For evaluation, we observe the convergence of the loss.
We also define a function called most_similar that takes as input a word and a number k and returns the k most similar words in the voca.
We also define a function called test_model that takes as input a path for a ground truth and computes the correlation between results of similarities.

### HYPERPARAMETERS CHOICE
learning rate: after a few quick experiences, learning_rate = 1e-2
For the rest of parameters, as we did not have time to experiment since the model takes a considerable time to train, we used standard choices:
embedding dimension: 100
window_size=5
negativeRate=5 
minCount=5


# REFERENCES
* Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributedrepresentations of words and phrases and their compositionality, 2013
* Eric Kim. Optimize Computational Efficiency of Skip-Gram with Negative Sampling. 26 May 2019. https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling
* Chris McCormick. Word2vec Tutorial Part 2: Negative Sampling. 11 Jan 2017. http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
* Yoav Goldberg and Omer Levy. word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method, 2014
