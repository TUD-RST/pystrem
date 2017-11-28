import unittest
import control as ctrl
import pystrem as ps
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
        m = ps.FsrModel(output, t=time)
        m.crop_to_dynamic_range()
        # testing our model with a test signal
        test_sgnl_len = int(2500)
        u = np.zeros(test_sgnl_len)
        for i in range(test_sgnl_len):
            u[i] = 1
        time = np.zeros(test_sgnl_len)
        for i in range(test_sgnl_len):
            time[i] = i
        _, comp, _ = ctrl.forced_response(sys, time, u)
        # 'rb' and 'wb' to avoid problems under Windows!
        with open("temp", "w") as fh:
            ps.export_csv(m, fh) 
        with open("temp", "r") as fh:
            m = ps.import_csv(fh)
        os.remove("temp")
        _, y = ps.forced_response(m, time, u)
        self.assertTrue(np.allclose(y, comp, rtol=1e-2), "import/export broke")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()