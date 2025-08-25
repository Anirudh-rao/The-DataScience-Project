# Fundamentals of JAX

In this module we will cover the basics of JAX Perfomance computing library.


**JAX is a library for array-oriented numerical computation (à la NumPy), with automatic differentiation and JIT compilation to enable high-performance machine learning research**.

This document provides a quick overview of essential JAX features, so you can get started with JAX:

1. JAX provides a unified NumPy-like interface to computations that run on CPU, GPU, or TPU, in local or distributed settings.

2. JAX features built-in Just-In-Time (JIT) compilation via Open XLA, an open-source machine learning compiler ecosystem.

3. JAX functions support efficient evaluation of gradients via its automatic differentiation transformations.

4. JAX functions can be automatically vectorized to efficiently map them over arrays representing batches of inputs.