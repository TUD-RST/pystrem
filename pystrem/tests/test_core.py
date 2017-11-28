'''
Created on Nov 1, 2017

@author: christoph
'''
import unittest
import numpy as np
import control as ctrl
import pystrem


class CoreTest(unittest.TestCase):

    sys = None
    sys = None
    m = None
    m = None
    time = None

    @classmethod
    def setUpClass(cls):

        super(CoreTest, cls).setUpClass()
        cls.sim_duration = 2000
        cls.time = np.zeros(cls.sim_duration)
        i = 0
        while (i < cls.sim_duration):
            cls.time[i] = i
            i += 1
        cls.u = np.zeros(cls.sim_duration)
        cls.t = np.arange(0, cls.sim_duration, 1)
        for i in range(cls.sim_duration):
            if i > 1000:
                cls.u[i] = 1
            else:
                cls.u[i] = 0.5

    def test_forced_response(self):
        
        sys = ctrl.tf([1], [100, 1])  # PT1
        _, o = ctrl.step_response(sys, self.time)
        m = pystrem.FsrModel(o, t=self.time, max_delta=0.01)
        _, y = m.forced_response(self.t, self.u)
        _, o, _ = ctrl.forced_response(sys, self.t, self.u)
        self.assertTrue(np.allclose(y, o, rtol=1e-2), "forced response broke")

    def test_PT2_forced_response(self):
        
        sys = ctrl.tf([1], [1000, 10, 1])  # PT2 with overshooting
        _, o = ctrl.step_response(sys, self.time)
        m = pystrem.FsrModel(o, t=self.time, max_delta=0.01)
        _, y = m.forced_response(self.t, self.u)
        _, o, _ = ctrl.forced_response(sys, self.t, self.u)
        self.assertTrue(np.allclose(y, o, rtol=1e-2), "pt2 response broke")

    def test_all_pass_forced_response(self):
        
        sys = ctrl.tf([-50, 1], [1000, 10, 1])  # all-pass
        _, o = ctrl.step_response(sys, self.time)
        m = pystrem.FsrModel(o, t=self.time, max_delta=0.01)
        _, y = m.forced_response(self.t, self.u)
        _, o, _ = ctrl.forced_response(sys, self.t, self.u)
        self.assertTrue(np.allclose(y, o, rtol=1e-1),
                        "all-pass response broke")

    def test_IT2_forced_response(self):
        
        sys = ctrl.tf([100], [50, 1000, 150, 0])  # IT2
        _, o = ctrl.step_response(sys, self.time)
        m = pystrem.FsrModel(o, t=self.time, max_delta=0.01,
                                show_warnings=False)
        _, y = m.forced_response(self.t, self.u)
        _, o, _ = ctrl.forced_response(sys, self.t, self.u)
        self.assertTrue(np.allclose(y, o, rtol=1e-2), "it2 response broke")
   
    def test_crazy_system_forced_response(self):
        # numerator: (s+50)(s-2)((s+8)²+45)(s²+100))
        # denominator: (s+2)(s+3)(s+3)(s+4)((s+1.5)²+5)((s+3)²+2)((s+0.5)²+3)
        #              ((s+4)²+30)
        sys = ctrl.tf(
            [1, 64, 877, 10032, 66800, 363200, -1090000], 
            [1, 30, 443.5, 4140, 26677.6, 124311, 430648, 1.12577e6, 2.22475e6,
             3.28564e6, 3.49963e6, 2.45298e6, 858429])
        time = np.arange(0, 60, 0.01)
        u = np.ones(len(time))
        for i in range(len(u)):
            if i < 25/0.01:
                u[i] = 0.5
        _, o = ctrl.step_response(sys, time)
        m = pystrem.FsrModel(o, time, max_delta=0.01,
                                show_warnings=False)
        _, y = m.forced_response(time, u)
        _, o, _ = ctrl.forced_response(sys, time, u)
        self.assertTrue(np.allclose(y, o, rtol=1e-2), "crazy response broke")

    def test_u_crop(self):
        
        sys = ctrl.tf([1], [100, 1])  # PT1
        _, o = ctrl.step_response(sys, self.time)
        u = np.zeros(self.sim_duration)
        for i in range(60, len(u)):
            u[i] = 1
        m = pystrem.FsrModel(o, t=self.time, u=u, max_delta=0.01)
        u = np.ones(len(m._u))
        self.assertTrue(np.array_equal(u, m._u), "u cropping broke")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
