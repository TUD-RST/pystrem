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
    # make it faster
    model.crop_to_dynamic_range(0.01)
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

if __name__ == '__main__':
   
    creation_example()
    
    