# DENOISING DIFFUSION IMPLICIT MODELS

This repository contains the code for the short experiment on fast inference for the Sprites by ElvGames as presented in the paper **"DENOISING DIFFUSION IMPLICIT MODELS"**.

## Paper Details

- **Title**: DENOISING DIFFUSION IMPLICIT MODELS

You can access the full paper [here](https://arxiv.org/pdf/2010.02502).

## Summary

The authors propose Denoising Diffusion Implicit Models (DDIMs) as a more efficient alternative to Denoising Diffusion Probabilistic Models (DDPMs) for image generation. While DDPMs require many steps in a Markov chain to generate samples, DDIMs introduce non-Markovian diffusion processes that maintain the same training objective but allow faster, deterministic sampling. The authors demonstrate that DDIMs can produce high-quality images 10 to 50 times faster than DDPMs, enable a trade-off between computation and sample quality, support image interpolation in latent space, and reconstruct observations with low error.
