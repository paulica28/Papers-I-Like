
# Replicating a Theoretical Exploration of Diffusion Models' Generalization Capabilities

This repository contains code that replicates the experiment presented in the paper titled **"On the Generalization Properties of Diffusion Models"** This paper provides a comprehensive theoretical analysis of the generalization attributes of diffusion models, establishing theoretical estimates for the generalization gap that evolves during the training dynamics of score-based diffusion models.

The implementation draws inspiration from the code presented in **"Score-Based Generative Modeling through Stochastic Differential Equations."**

## Paper Abstract

Diffusion models are a class of generative models that establish a stochastic transport map between an empirically observed, yet unknown, target distribution and a known prior. Despite their remarkable success in real-world applications, a theoretical understanding of their generalization capabilities remains underdeveloped.

This work embarks on a comprehensive theoretical exploration of the generalization attributes of diffusion models. We establish theoretical estimates of the generalization gap that evolves in tandem with the training dynamics of score-based diffusion models, suggesting a polynomially small generalization error \(O(n^{-2/5} + m^{-4/5})\) on both the sample size \(n\) and the model capacity \(m\), evading the curse of dimensionality when early-stopped. Furthermore, the authors extend their quantitative analysis to a data-dependent scenario, wherein target distributions are portrayed as a succession of densities with progressively increasing distances between modes.

The findings have been validated through numerical simulations, contributing to a rigorous understanding of diffusion models' generalization properties and offering insights for practical applications.

## Training Code

The main training loop for score-based diffusion models is implemented using the following loss function:

```python
def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a
          time-dependent score-based model.
        x: A mini-batch of training data.
        marginal_prob_std: A function that gives the standard deviation of
          the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss
```

### Explanation of the Loss Function
The loss function is designed for training score-based generative models:

- `model`: The score-based model learns to estimate the gradient of the log probability density function with respect to the input data.
- `x`: A mini-batch of training data.
- `marginal_prob_std`: A function that provides the standard deviation of the perturbation kernel at a given time.
- The model learns to match the perturbed data distribution to the underlying data distribution by minimizing the difference between the estimated score and the true perturbations.


## Key Findings

- The paper establishes that the generalization error of score-based diffusion models is polynomially small in both the sample size \(n\) and the model capacity \(m\), avoiding the curse of dimensionality when early-stopped.
- The analysis extends to data-dependent scenarios, providing insights into how "mode shifts" in ground truths affect model generalization.
- Numerical simulations confirm the theoretical estimates, validating the model's generalization capabilities.

## References

- **Main Paper**: [https://proceedings.neurips.cc/paper_files/paper/2023/file/06abed94583030dd50abe6767bd643b1-Paper-Conference.pdf]
- **Inspired by**: "Score-Based Generative Modeling through Stochastic Differential Equations" [https://arxiv.org/pdf/2011.13456]
