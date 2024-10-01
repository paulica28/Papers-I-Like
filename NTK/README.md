# Neural Tangent Kernel: Convergence and Generalization in Neural Networks

This repository contains the code for some short experiments on neural tangent kernels (NTK) as introduced in the blogpost below.

## Paper Details

- **Title**: Some Math behind Neural Tangent Kernel

You can access the full blogpost [here](https://lilianweng.github.io/posts/2022-09-08-ntk/).

## Summary

Neural networks are widely recognized for being over-parameterized, often capable of fitting data with an almost zero training loss while still achieving good generalization on test datasets. Despite the fact that all these parameters start with random initialization, the optimization process tends to yield consistently strong results. This holds true even when the model has more parameters than training data points.
