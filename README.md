# Computer Vision Demo

The goals of this project were to:

1. refresh on some machine learning concepts and implementation
2. extend my knowledge about computer vision
3. learn pytorch

## The Project

### Process

#### Selecting a Dataset

A dataset should be appropriate for the project goals. The dataset must be easily available and ideally designed for computer vision classification.

The following criteria were designed to guide dataset selection.

##### 1. Popularity

Selecting a popular dataset already used for academic research means availability of papers and other resources discussing the dataset. The literature communicates how complex the dataset is and the degree of sophistication of models appropriate to use with it.

A popular dataset will also have performance benchmarking to measure progress against.

##### 2. Modest Scope

Executing training runs efficiently (on an NVIDIA RTX3060 12GB GPU) ensures tight feedback loops to maximise learning.

##### 3. Sufficiently Deep

An overly simple dataset with obvious signal (eg. MNIST) is less useful, because it is easy to achieve maximal outcomes with even very simple models. This limits the scope for exploration compared with a dataset against which improvements in performance can be discovered by applying better techniques and models.

#### Selected Dataset

The Fashion-MNIST dataset was selected.

Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)

Two sets of benchmarked results are available for this dataset.

The original set is tested by the contributors of the dataset:  
http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/

A secondary set is contributed by the community and shared on the github page (scroll down a little from this link):  
https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file#benchmark

#### Building a Pipeline

A training pipeline was built using a standard machine learning toolkit (Python, Jupyter, Pytorch, TensorBoard).

Experimentation is performed using a Jupyter Notebook and TensorBoard for exploration and observation.

Project code is encapsulated in classes contained a source directory so that each concept is easy to focus on in isolation, facilitating efficient experimentation.

#### Experimentation

Using this pipeline, experimentation was performed on the dataset with concepts implemented as changes in model, loss function, optimizer, data transformations, and hyperparameters.

Improvements are considered to be (in order of descending importance):

* higher accuracy of prediction against the test set (which is not trained on)
* reduction in test set loss
* reduction in training set loss
* faster descent of training & test loss (reduction in training time)
* low divergence of training & test loss

These are observed via Jupyter Notebook logging and TensorBoard visualisations.

A combination of theoretical (eg. changing the model based on hypotheses) and empirical experimentation (eg. grid search over hyperparameters) were employed.

### Experimental Results

#### Model

The first model built was a simple 3 linear layer stack, separated by rectified linear unit activation functions, outputting to a 10 neuron layer (matching the 10 output classes of the dataset).

This model was sufficient to develop and test the training pipeline. In its original incarnation (and in combination with initial loss function, optimizer, data transforms and hyperparameters) a baseline accuracy of ~65% was achieved on predictions against the test set - a poor perfomance against benchmarks.

The second model was initialized with a modestly more complex structure:

* Convolutional layer
* ReLU activation function
* Max Pooling layer
* Convolutional layer
* ReLU activation function
* Max Pooling layer
* Random Dropout
* Linear layer
* ReLU activation function
* Linear layer outputting classes

Two blocks of convolutional layers extract initial, and then more abstract features from the input image. ReLU activation functions zero any negative weights at each stage and Max Pooling layers reduce dimensionality so semantic features are maintained, facilitating generalisation.

A Random Dropout is introduced during training to reduce overreliance on specific nodes and improve model robustness. 

The final dense Linear layers map the features captured by the convolutional layers through to the final set of classes.

Additional experimentation:

A LogSoftMax layer was added in order to normalize the output of the final layer into a percentage prediction for each class. This did not materially change metrics of success, but can be useful for interpreting results.

The Max Pooling layers were substituted for Average Pooling layers to see if average feature representation contained more discriminative power than maximum feature representation. However, this marginally worsened results.

#### Loss Function

Cross-entropy loss was selected as a standard logistic regression loss function.

When the LogSoftMax layer was added, this was changed to the negative log-likelihood loss function. The combined effect of these changes were not significant for accuracy or loss.

#### Optimizer

Stochastic gradient descent was selected as a common optimizer.

Additional optimizers were tested experimentally.

Improvement was found using the Adam optimizer. From some research, it seems likely that this improvement results from overcoming suboptimal hyperparameter initialization and naive application of a static learning rate.

#### Data Preparation

Initially, no functional transformations were performed on the data apart from loading into batches of tensors for ingestion into the pipeline.

Later, to introduce additional robustness and generalisation into the model, random horizontal flips and minor rotational pertubation were introduced. Together these provided a small boost to performance.

#### Hyperparameters

After each significant change to the abovementioned components, grid search was run over hyperparameters (eg. batch size, learning rate, number of training epocs) to select optimal values.

#### Outcome

With the final model, loss function and optimizer choice, and hyperparameter values, accuracy of roughly ≥93.0% is recorded against the test set. This represents a better score than models tested by the original dataset contributors, and a score competetive with the community contributed models.

#### Improvements

If I were to spend more time on this project I would try out some more sophisticated model structures and concepts such as residual networks and visual transformers.

## Running locally

[This notebook](./cv_demo.ipynb) is the starting point.

Additional project code can be found in [the src folder](./src).

### Dependencies

* Python 3.12.3
* (optional) CUDA 12.4 - if you want to run on NVIDIA GPU

If you are using `uv` (a modern python package and project management tool written in rust):

```sh
uv python install
uv python sync
```

If you are using `pip`, with an appropriate version of python installed and referenced by `python`:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

### Usage

Using VS Code, you can execute the notebook and visit TensorBoard in your IDE:

* Install the [jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
* Install the [tensorboard extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.tensorboard)

Alternatively, you can use jupyter lab or jupyter notebook, depending on your environment.
