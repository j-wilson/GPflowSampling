# GPflowSampling
Companion code for [Efficiently Sampling Functions from Gaussian Process Posteriors](https://arxiv.org/abs/2002.09309) and [Pathwise Conditioning of Gaussian processes](https://arxiv.org/abs/2011.04026).
## Overview
Software provided here revolves around Matheron's update rule

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;(f&space;\mid&space;\mathbf{y})(\cdot)&space;=&space;f(\cdot)&space;&plus;&space;k(\cdot,&space;\mathbf{X})\mathbf{K}^{-1}\big(\mathbf{y}&space;-&space;f(\mathbf{X})\big)," target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;(f&space;\mid&space;\mathbf{y})(\cdot)&space;=&space;f(\cdot)&space;&plus;&space;k(\cdot,&space;\mathbf{X})\mathbf{K}^{-1}\big(\mathbf{y}&space;-&space;f(\mathbf{X})\big)," title="matherons_update_rule" /></a>

which allows us to represent a GP posterior as the sum of a prior random function and a data-driven update term. Thinking about conditioning at the level of random function (rather than marginal distributions) enables us to accurately sample GP posteriors in linear time.

Please see `examples` for tutorials and (hopefully) illustrative use cases.

## Installation
```
git clone git@github.com:j-wilson/GPflowSampling.git
cd GPflowSampling
pip install -e .
```
To install the dependencies needed to run `examples`, use `pip install -e .[examples]`.


## Citing Us
If our work helps you in a way that you feel warrants reference, please cite the following paper:
```
@inproceedings{wilson2020efficiently,
    title={Efficiently sampling functions from Gaussian process posteriors},
    author={James T. Wilson
            and Viacheslav Borovitskiy
            and Alexander Terenin
            and Peter Mostowsky
            and Marc Peter Deisenroth},
    booktitle={International Conference on Machine Learning},
    year={2020},
    url={https://arxiv.org/abs/2002.09309}
}
```
