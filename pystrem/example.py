import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
import pystrem as fsr

if __name__ == '__main__':
    # creating a step response for our model
    sys = ctrl.tf([1], [100, 1])
    sim_duration = 2000
    time = np.arange(sim_duration)
    _, output = ctrl.step_response(sys, time)   
    # creating the model
    model = fsr.FsrModel(output, t=time)
    # creating a test input signal
    u = np.ones(sim_duration)
    for i in range(len(u)):
        if i < 500:
            u[i] = 0.001*i
    # getting response from our model
    t1, y1 = model.forced_response(time, u)
    # simulating a reference with control
    t2, y2, _ = ctrl.forced_response(sys, time, u)
    # show everything
    plt.plot(t1, y1, t1, y2)
    plt.show()
