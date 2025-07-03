Spike Bench is a spiking neural network prototyping framework built from the ground up for
maximum control over network dynamics.
It provides a basic UI to configure the networks and train them. This is alpha software.
To build it, you need a copy of `imgui` and `rlImGui` in a directory called `ext` in the 
root of the project.
This builds with some modifications on RHEL 8.

This library simulates the following dynamics:
- STDP
- STP
- Backpropogation
- Random Feedback Backprop
- Metaplasticity with consolidation

The following types of neurons are modelled:
- Izhikevich (Izh)
- Izhikevich with dedicated dendritic compartments (In Progress)
- Stochastic (RNG)
- Leaky-Integrate-and-Fire (LIF)
- Resonate-and-Fire (RAF)

There are too many surrogate gradients to list them all, but here are some:
- Fast Sigmoid
- Super Spike
- Exponential
- Soft Relu

The system currently supports fully connected and convolutional layers in arbitrary
arrangements with any number of layers in any layout.

Right now it runs on the CPU, though it has been ported to CUDA before.
On the GPU, the performance is only better than the CPU with sufficiently large networks.

There are a few key parts of this code:
- `src/math/` includes a small expression template system for array and tensor operations
along with a hardware and software implementation of SIMD functions for optimization. It also
contains a number of key algorithms and some generic functions.
- `src/snn5` the core network simulation code and the UIs for configuring it.
- `src/util` misc utility and data loading functionality.
- `src/test` a number of test applications, the main one being `snn5_ui.cpp`

The project automatically compiles optimized release binaries along with debug and valgrind
versions. The valgrind version requires a special software implementation of AVX512 since it
does not support those instructions yet. This is what the `sw.h` file is for in the math folder.
