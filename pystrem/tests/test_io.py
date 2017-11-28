'''
Created on Nov 7, 2017

@author: christoph
'''
import unittest
import control as ctrl
import pystrem
import numpy as np
import os


class IOTest(unittest.TestCase):


    def test_import_export(self):
    
        sys = ctrl.tf([1], [100, 1])
        sim_duration = 2000
        time = np.zeros(sim_duration)
        i = 0
        while (i < sim_duration):
            time[i] = i
            i += 1
        _, output = ctrl.step_response(sys, time)   
        m = pystrem.FsrModel(output, t=time)
        # testing our model with a test signal
        test_sgnl_len = int(2500)
        u = np.zeros(test_sgnl_len)
        for i in range(test_sgnl_len):
            u[i] = 1
        time = np.zeros(test_sgnl_len)
        for i in range(test_sgnl_len):
            time[i] = i
        _, comp, _ = ctrl.forced_response(sys, time, u)
        # "rb" and "wb" to avoid problems under Windows!
        with open("temp.csv", "w") as fh:
            pystrem.export_csv(m, fh) 
        with open("temp.csv", "r") as fh:
            m = pystrem.import_csv(fh, show_warnings=False)
        os.remove("temp.csv")
        _, y = m.forced_response(time, u)
        self.assertTrue(np.allclose(y, comp, rtol=1e-2), "import/export broke")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()