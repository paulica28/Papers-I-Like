# DENOISING DIFFUSION IMPLICIT MODELS

This repository contains the code for the short experiment on Flow Matching using the Sprites by ElvGames dataset.

## Paper Details

- **Title**: FLOW MATCHING FOR GENERATIVE MODELING

You can access the full paper [here](https://arxiv.org/pdf/2210.02747).

## Summary

This paper introduces *Flow Matching* (FM), a new method for training Continuous Normalizing Flows (CNFs) that enhances their scalability and performance. Unlike previous methods, FM doesn’t rely on simulations but instead focuses on regressing vector fields of fixed conditional probability paths. This approach is compatible with a wide variety of Gaussian paths, including diffusion paths, which are often used in generative modeling. The authors find that FM not only makes diffusion-based models more robust and stable but also enables CNFs to be trained with alternative paths, such as Optimal Transport (OT) displacement interpolation. These OT paths improve efficiency, allowing for faster training and sampling, and lead to better generalization. Experiments on ImageNet show that FM-trained CNFs outperform diffusion-based models in both likelihood and sample quality, with faster, more reliable sample generation using standard numerical ODE solvers.
