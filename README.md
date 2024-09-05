# Computer Vision Demo

A pytorch computer vision demo project that walks through setting up and training a computer vision model on a fashion image dataset.

The purpose of this project was to spend a bit of time learning pytorch and refresh on some computer vision (and general ML) concepts.

[This notebook](./cv_demo.ipynb) is the starting point.

Additional project code can be found in [the src folder](./src).

## Dependencies

* Python 3.12.3
* (optional) CUDA 12.4 - if you want to run on NVIDIA GPU

If you are using `uv` (a modern python package and project management tool written in rust):

```sh
TODO
```

If you are using `pip`

```sh
TODO
```

conda? anything else?

## Running locally

Using VS Code, you can execute the notebook and visit TensorBoard in your IDE:

* Install the [jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
* Install the [tensorboard extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.tensorboard)

Alternatively, you can use jupyter lab or jupyter notebook, depending on your environment.

## Notes

The model...

With the current set of hyperparameter values, accuracy of approximately 93.0% is recorded against the test set.

## References

* Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)
* https://github.com/kefth/fashion-mnist

## TODO

* flesh out the readme
  * get some theory
  * document and explain why the model is good
