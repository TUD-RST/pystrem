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
    plt.plot(t1, y1, t2, y2, t1, u)
    plt.show()
    
def external_sim_example():
    """Example for use in a simulation:
    This creates a system with a discrete controller and a FSR model in a
    feedback loop. Looks like this:
                           
        w--*-e->|R|--u->|P|----> y
           ^-               |
           |                |
           ------------------
    
    To keep it simple R = 2 and P is an unstable PT1 system with
        
               1
       P =  -------
             s - 1
             
    """
    # as in above example, create our model from python-control step response
    sys = ctrl.tf([1], [1, -1])
    # create simulation time array
    time = np.arange(0, 5., 0.01)
    _, output = ctrl.step_response(sys, time)   
    # When creating models from unstable systems it is important to create the 
    # model with a step response which is as long as the simulation time. 
    # Otherwise we will very likely make mistakes in simulation. 
    model = ps.FsrModel(output, t=time)

    # create input signal, a step here
    w = np.ones(len(time))
    y = np.zeros(len(time))
    for i in range(len(time)):
        if i != 0:
            e = w[i] - y[i-1]
        else:
            e = w[i]
        u = 2 * e  # here we would normally call a function which calculates
        # the output of our discrete system, here it is simply 2 * e
        y[i] = model.simulate_step(u)
    plt.plot(time, y)
    plt.show()
        
    

if __name__ == '__main__':
   
    creation_example()
    external_sim_example()
    
    