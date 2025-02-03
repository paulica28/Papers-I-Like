# RECTIFIED FLOW

This repository contains the code for a minimal implementation of **Rectified Flow** as presented in the paper "Flow Straight and Fast:
Learning to Generate and Transfer Data with Rectified Flow". The implementation demonstrates a simple yet effective method for learning neural ODE models for distribution transport.

## Paper Details

**Title:** Rectified Flow  
**Paper:** [https://arxiv.org/pdf/2209.03003](#)  
*(Replace the link above with the actual URL if available.)*

## Summary

We present **Rectified Flow**, a surprisingly simple approach to learning (neural) ordinary differential equation (ODE) models to transport between two empirically observed distributions, π₀ and π₁. This method provides a unified solution to generative modeling, domain transfer, and various other tasks involving distribution transport.

The core idea behind Rectified Flow is to learn an ODE that follows the straight paths connecting points drawn from π₀ and π₁ as closely as possible. This is achieved by solving a straightforward nonlinear least squares optimization problem, which scales to large models without introducing extra parameters beyond standard supervised learning.

Key points include:
- **Efficiency:** The straight paths are the shortest routes between two points and can be simulated exactly without time discretization, yielding computationally efficient models.
- **Deterministic Coupling:** The process of rectification transforms an arbitrary coupling between π₀ and π₁ into a deterministic coupling with provably non-increasing convex transport costs.
- **Recursive Improvement:** Recursively applying rectification produces a sequence of flows with increasingly straight paths, allowing accurate simulation even with coarse time discretization during inference.
- **Empirical Success:** Experiments demonstrate that Rectified Flow achieves superb performance on tasks such as image generation, image-to-image translation, and domain adaptation—even with a single Euler discretization step.

