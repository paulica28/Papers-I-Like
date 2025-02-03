# RECTIFIED FLOW

This repository contains the code for a minimal implementation of **Rectified Flow** as presented in the paper "Flow Straight and Fast:
Learning to Generate and Transfer Data with Rectified Flow". The implementation demonstrates a simple yet effective method for learning neural ODE models for distribution transport.

## Paper Details
  
**Paper:** [https://arxiv.org/pdf/2209.03003](#)  

## Summary

Rectified Flow is a surprisingly simple approach to learning neural ordinary differential equation (ODE) models for transporting between two empirically observed distributions, π₀ and π₁. The method provides a unified solution for tasks such as generative modeling, domain transfer, and various other applications involving distribution transport.

The core idea behind Rectified Flow is to learn an ODE that follows nearly straight paths connecting points drawn from π₀ and π₁. This is achieved by solving a nonlinear least squares optimization problem that scales to large models without introducing extra parameters beyond standard supervised learning.

Key features include:
- **Efficiency:** The straight paths represent the shortest routes between two points and can be simulated exactly without time discretization, yielding computationally efficient models.
- **Deterministic Coupling:** The rectification process transforms an arbitrary coupling between π₀ and π₁ into a deterministic coupling with provably non-increasing convex transport costs.
- **Recursive Improvement:** Recursively applying rectification results in a sequence of flows with increasingly straight paths, which can be simulated accurately even with coarse time discretization during inference.
- **Empirical Performance:** Experimental results demonstrate that Rectified Flow achieves excellent performance on tasks such as image generation, image-to-image translation, and domain adaptation—even with a single Euler discretization step.

