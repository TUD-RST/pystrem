import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
import pystrem as ps

# TODO: move this to doc as maybe a quickstart guide or simply as examples

def creation_example():
    """Example for creation of a FSR model with python-control."""

    # creating a step response for our model
    sys = ctrl.tf([1], [100, 1])
    sim_duration = 2000
    time = np.arange(sim_duration)
    _, output = ctrl.step_response(sys, time)   
    # creating the model
    model = ps.FsrModel(output, t=time)
    # creating a test input signal
    u = np.ones(sim_duration)
    for i in range(len(u)):
        if i < 500:
            u[i] = 0.001*i
    # getting response from our model
    t1, y1 = ps.forced_response(model, time, u)
    # simulating a reference with control
    t2, y2, _ = ctrl.forced_response(sys, time, u)
    # show everything
    ax = plt.subplot(111)
    plt.plot(t1, y1, label="Model response")
    plt.plot(t2, y2, label="Reference response")
    plt.plot(t1, u, label="Input signal")
    ax.legend()
    plt.title("pystrem example")
    plt.show()

def mpc_example():
    """Example of using a model-predictive-controller with pystrem."""
    
    # Simulation times
    t = np.arange(0, 15, 0.1)
    t_horizon = np.arange(0, 3, 0.1)
    
     # Creating our plants   
    sys_11 = ctrl.tf([1], [1, 1, 1])
    _, y = ctrl.step_response(sys_11, t)
    model_11 = ps.FsrModel(y, t)
    
    sys_12 = ctrl.tf([1], [1, 1])
    _, y = ctrl.step_response(sys_12, t)
    model_12 = ps.FsrModel(y, t)
    
    sys_22 = ctrl.tf([2], [1, 2, 1])
    _, y = ctrl.step_response(sys_22, t)
    model_22 = ps.FsrModel(y, t)
    
    sys_21 = ctrl.tf([1], [1, 1])
    _, y = ctrl.step_response(sys_21, t)
    model_21 = ps.FsrModel(y, t)
    
    # creating our system and MPC
    sys = np.ndarray((2, 2), ps.FsrModel)
    sys[0][0] = model_11  # output 1 from input 1
    sys[0][1] = model_12  # output 1 from input 2
    sys[1][1] = model_22  # output 2 from input 2
    sys[1][0] = model_21  # output 2 from input 1
    mpc = ps.Mpc()
    mpc.set_minimizer_args(method='Nelder-Mead')
    
    # creating our desired outputs
    y_d = np.ndarray((2, len(t)), float)
    y_d[0] = np.ones(len(t))   # desired step for output 1
    y_d[1] = np.zeros(len(t))  # and desired zero for output 2
    
    # simulate!
    t, y, u = mpc.simulate(sys, y_d, t_horizon, t)
    
    # plots
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Plant Outputs")
    ax1.plot(t, y[0], label="plant output 1")
    ax1.plot(t, y[1], label="plant output 2")
    ax1.legend()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Controller Outputs")
    ax2.plot(t, u[0], label="controller 1 output")
    ax2.plot(t, u[1], label="controller 2 output")
    ax2.legend()
    
    # some reference to see if we actually did something
    _, y1 = ps.step_response(model_11, t)
    _, y2 = ps.step_response(model_21, t)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title("Reference Outputs")
    ax3.plot(t, y1, label="plant output 1")
    ax3.plot(t, y2, label="plant output 2")
    ax3.legend()
    
    plt.show()

if __name__ == '__main__':
   
    mpc_example()
    
    