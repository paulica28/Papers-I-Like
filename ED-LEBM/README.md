
# Energy Discrepancy (ED) for Training Energy-Based Models

This repository contains an implementation of the training process for the paper titled **"Energy Discrepancies: A Score-Independent Loss for Energy-Based Models"** The paper introduces a novel loss function called **Energy Discrepancy (ED)**, designed to improve the training efficiency and accuracy of energy-based models (EBMs) without relying on traditional, computationally expensive methods.

## Paper Abstract

Energy-based models are a simple yet powerful class of probabilistic models, but their widespread adoption has been limited by the computational burden of training them. In a recent paper, the authors propose a novel loss function called Energy Discrepancy (ED), which does not rely on the computation of scores or expensive Markov chain Monte Carlo (MCMC).

The authors show that energy discrepancy approaches the explicit score matching and negative log-likelihood loss under different limits, effectively interpolating between both. Consequently, minimum energy discrepancy estimation overcomes the problem of nearsightedness encountered in score-based estimation methods, while also enjoying theoretical guarantees.

Through numerical experiments, they demonstrate that ED learns low-dimensional data distributions faster and more accurately than explicit score matching or contrastive divergence. For high-dimensional image data, the paper describes how the manifold hypothesis puts limitations on their approach and demonstrates the effectiveness of energy discrepancy by training the energy-based model as a prior of a variational decoder model.

## Training Process

The training process is designed to minimize the Energy Discrepancy (ED) loss for energy-based models (EBMs). The model is trained using a combination of a generator network (`netG`) and an energy-based prior model (`netE`). Here's an overview of the training steps:

1. **Generator Training:**
   - Sample initial latent vectors \( z_g_0 \) from a prior distribution.
   - Apply Langevin dynamics to obtain updated latent vectors \( z_g_k \).
   - Train the generator network `netG` to minimize the mean squared error (MSE) loss between generated samples and real data.

2. **Energy-Based Model (EBM) Training:**
   - Compute the energy values of positive and negative samples using `netE`.
   - Use the ED loss to update `netE`'s parameters by applying backpropagation.

3. **Exponential Moving Average (EMA) Updates:**
   - The parameters of `netE` are updated using an EMA technique to stabilize training.

### Training Code Example

The core training loop looks like this:

```python
for epoch in range(10):
    pbar = tqdm(dataloader_train)
    for x, _ in pbar:
        x = x.to(device) * 2 - 1

        z_g_0 = sample_p_0(x.size(0))
        z_g_k = sample_langevin_post_z(Variable(z_g_0), x, netG, netE)

        # Learn generator
        optG.zero_grad()
        x_hat = netG(z_g_k.detach())
        loss_g = mse(x_hat, x) / x.size(0)
        loss_g.backward()
        optG.step()

        # Learn prior EBM
        optE.zero_grad()
        z_g_k = z_g_k.detach().squeeze()    
        pert_z_g_k = z_g_k + t_sqrt * torch.randn_like(z_g_k)
        pert_z_g_k = pert_z_g_k.unsqueeze(1) + t_sqrt *                 torch.randn(z_g_k.size(0), m_particles, *z_g_k.shape[1:]).to(z_g_k.device)
        pert_z_g_k = pert_z_g_k.detach().view(-1, *pert_z_g_k.shape[2:])
        
        en_pos = netE(z_g_k).view(z_g_k.size(0))                     + 1.0 / (2 * e_prior_sig * e_prior_sig) * (z_g_k**2).sum(dim=-1)
        en_neg = netE(pert_z_g_k).view(z_g_k.size(0), -1)                     + 1.0 / (2 * e_prior_sig * e_prior_sig) * (pert_z_g_k**2).sum(dim=-1).view(z_g_k.size(0), -1)
        en_ctr = en_pos.unsqueeze(-1) - en_neg
        en_ctr = torch.cat([torch.zeros_like(en_ctr[:, :1]), en_ctr], dim=1)
        loss_e = en_ctr.logsumexp(dim=-1).mean()
        loss_e.backward()
        optE.step()

        with torch.no_grad():
            for p, ema_p in zip(netE.parameters(), ema_netE.parameters()):
                ema_p.mul_(0.999).add_(p, alpha=0.001)

        pbar.set_description('Epoch: {:4d}, loss_g: {:.4f}, loss_e: {:.4f}'.format(epoch, loss_g.item(), loss_e.item()))
    
    lr_scheduleE.step()
    lr_scheduleG.step()
```

## Key Features of Energy Discrepancy (ED)

- **Efficient Training**: The Energy Discrepancy loss allows training without expensive MCMC sampling, significantly reducing computational costs.
- **Interpolates Between Loss Functions**: ED smoothly transitions between score matching and negative log-likelihood under different limits, offering flexibility in training.
- **Handles Low-Dimensional Data Well**: Numerical experiments show that ED learns low-dimensional data distributions faster and more accurately than traditional methods like score matching or contrastive divergence.
- **Effective for High-Dimensional Data**: Although the manifold hypothesis imposes some limitations, ED has been shown to be effective when used as a prior for training a variational decoder model.

## References

- Paper Title: "Energy Discrepancies: A Score-Independent Loss for Energy-Based Models"
- For more details, refer to the full [paper](https://arxiv.org/pdf/2307.06431).
