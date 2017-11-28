# PyStrem
A Python library for feedback control using finite step response models.

PyStrem is a library providing a framework for the handling of finite step response models used for systems analysis, and feedforward as well as feedback control. The main characteristics of finite step response models is the fact that the system dynamics is not represented by an ordinary differential equation but by the step response data itself. They are especially used for linear systems the dynamics of which can only be described by high order transfer functions.

The library enables the user to import data collected from step response experiments and build objects representing the dynamic behavior of the system by means of this data. Different step response models can be connected in series, in parallel or by feedback and the response of the models to arbitrary input signals can be calculated. The MIMO case will be also covered.

Currently, PyStrem is under heavy development.
