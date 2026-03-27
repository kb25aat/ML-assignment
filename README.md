# Kolmogorov-Arnold Networks (KANs)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

## Overview

MLPs place **fixed** activation functions on nodes. KANs (Liu et al., April 2024) invert this: every **edge** has its own learnable activation function represented as a B-spline. Grounded in the Kolmogorov-Arnold Representation Theorem (1957), KANs offer higher interpretability and excel at smooth function approximation.

**Technique:** KANs B-spline activations, Kolmogorov-Arnold theorem, symbolic regression  
**Year:** 2024 (brand new not in any standard ML curriculum)  
**Difficulty:** Advanced / beyond-course

## What You Will Learn
- Why KANs invert the MLP design activations on edges, not nodes
- The Kolmogorov-Arnold theorem and why it justifies this design
- B-spline basis functions piecewise polynomial learnable curves
- Complete KAN implementation in PyTorch
- Learnable edge activation visualisation
- KAN vs MLP on function approximation
- Interpretability and symbolic regression

## Quick Start
```bash
git clone https://github.com/yourusername/kan-tutorial.git
cd kan-tutorial
pip install torch numpy matplotlib scikit-learn notebook
jupyter notebook kan_tutorial.ipynb
```

## References
1. Liu et al. (2024)  KAN. https://arxiv.org/abs/2404.19756
2. Kolmogorov (1957)  Superposition theorem.
3. Arnold (1963)  Functions of three variables.
4. Liu et al. (2024)  KAN 2.0. https://arxiv.org/abs/2408.10205

