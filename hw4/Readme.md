# GAN Variants Implementation on CIFAR-10

Implementation and comparison of three GAN architectures (DCGAN, WGAN, ACGAN) trained on the CIFAR-10 dataset.

## Models Performance 
| Model  | Initial FID | Final FID  | 
|--------|-------------|------------|
| DCGAN  | 472.2073    | 207.7875   |
| WGAN   | 367.7482    | 189.3365   |
| ACGAN  | 379.9142    | 268.8299   |

## Repository Structure
Each results directory contains:
- GeneratedImages: Generated image grids per epoch
- RealImages: Real image grids for comparison
- Metrics: Training metrics and FID scores
- ComparedImages (WGAN only): Side-by-side comparisons

## Training Environment
- Platform: Palmetto Cluster
- GPU: NVIDIA V100 
- CPU: 10 cores
- Memory: 30GB
- GPU Memory: 8GB

## Configuration

### Common Parameters
- Epochs: 50
- Batch Size: 128  
- Learning Rate: 0.0002
- Adam Optimizer:
  - β1: 0.5
  - β2: 0.999
- Image Size: 64x64x3
- Latent Dimension: 100

### WGAN Specific
- Critic Iterations: 3
- Gradient Penalty Weight: 10

### ACGAN Specific
- Number of Classes: 10 (CIFAR-10 classes)

## Dataset
CIFAR-10:
- 60,000 32x32 RGB images
- 10 classes
- Split: 50,000 training, 10,000 test
- Preprocessing: Resized to 64x64, normalized to [-1, 1]

## Results Summary
- WGAN achieves the best FID score (189.34) with the most stable training
- DCGAN baseline shows competitive performance (FID: 207.79)
- ACGAN provides class conditioning with 97.05% classification accuracy

For detailed analysis and full training progression, refer to model-specific results directories.
