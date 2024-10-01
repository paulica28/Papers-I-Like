
# Restricted Boltzmann Machine (RBM)

This repository contains a Python implementation of a Restricted Boltzmann Machine (RBM) using PyTorch. The RBM is an energy-based, generative stochastic neural network that can learn a probability distribution over its set of inputs.

## What is an RBM?

A Restricted Boltzmann Machine (RBM) is a two-layer neural network consisting of a **visible layer** and a **hidden layer**. These layers are fully connected, but there are no connections between the neurons within a layer (i.e., no visible-visible or hidden-hidden connections). This structure makes RBMs "restricted" compared to general Boltzmann Machines, enabling efficient training.

RBMs are primarily used for:
- Feature learning
- Dimensionality reduction
- Collaborative filtering
- Pretraining deep neural networks

### Energy Function

RBMs define an energy function that assigns a scalar energy value to every configuration of visible and hidden units. The model learns by adjusting its weights to minimize the energy of the training data.

The joint probability distribution over the visible (\(v\)) and hidden (\(h\)) layers is defined as:
\[
P(v, h) = \frac{1}{Z} \exp(-E(v, h))
\]
where \(E(v, h)\) is the energy function and \(Z\) is a normalizing constant.

## Training RBMs with Contrastive Divergence (CD)

Training an RBM involves maximizing the likelihood of the observed data, which is typically done using the Contrastive Divergence (CD) algorithm. CD is an efficient approximation technique for training RBMs, and it works as follows:

1. **Positive Phase**: 
   - Compute the hidden activations given the input data (visible layer).
   - This step captures the correlation between visible and hidden layers based on the input data.

2. **Negative Phase**:
   - Use Gibbs sampling to reconstruct the visible layer from the hidden activations, then recompute the hidden activations.
   - This step refines the model's learned distribution by contrasting the reconstructed data against the original input.

3. **Parameter Update**:
   - Update the weights and biases using the difference between the positive and negative associations.

CD can be run for one step (CD-1) or multiple steps (CD-k), where more steps improve the approximation but increase computational cost.


## References

- Geoffrey Hinton, "A Practical Guide to Training Restricted Boltzmann Machines" (2012)
- [RBMs Explained](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
