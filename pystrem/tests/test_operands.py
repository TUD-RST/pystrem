import unittest
import control as ctrl
import numpy as np
import pystrem as ps


class OperandTest(unittest.TestCase):

    # Class attributes
    sys1 = None
    sys2 = None
    m1 = None
    m2 = None
    time = None

    @classmethod
    def setUpClass(cls):

        # See https://stackoverflow.com/questions/14044474/python-unittest-setupclass-is-giving-me-trouble-why-cant-i-inherit-like-t
        super(OperandTest, cls).setUpClass()
        # Creating our systems is slow, so we do it once at startup.
        cls.sys1 = ctrl.tf([1], [10, 1])
        cls.sys2 = ctrl.tf([2], [10, 10, 1])
        sim_duration = 100
        cls.time = np.zeros(sim_duration)
        i = 0
        while (i < sim_duration):
            cls.time[i] = i
            i += 1
        _, o1 = ctrl.step_response(cls.sys1, cls.time)
        _, o2 = ctrl.step_response(cls.sys2, cls.time)
        cls.m1 = ps.FsrModel(o1, t=cls.time)
        cls.m2 = ps.FsrModel(o2, t=cls.time)

    def test_add(self):

        sum_sys = self.sys1 + self.sys2
        _, y1 = ctrl.step_response(sum_sys, self.time)
        sum_m = self.m1 + self.m2
        _, y2 = ps.step_response(sum_m, self.time)
        self.assertTrue(np.allclose(y1, y2, rtol=1e-2), "add is broken")
        y3, _, _ = (3 + self.m1).get_model_info()
        y4 = self.m1.get_model_info()[0] + 3
        self.assertTrue(np.allclose(y3, y4), "scalar addition is broken")

    def test_sub(self):
        diff_sys = self.sys1 - self.sys2
        _, y1 = ctrl.step_response(diff_sys, self.time)
        diff_m = self.m1 - self.m2
        _, y2 = ps.step_response(diff_m, self.time)
        self.assertTrue(np.allclose(y1, y2, rtol=1e-2), "subtract is broken")
        y3, _, _ = (3 - self.m1).get_model_info()
        y4 = 3 - self.m1.get_model_info()[0]
        self.assertTrue(np.allclose(y3, y4), "scalar subtraction is broken")

    def test_mul(self):

        prod_sys = self.sys1 * self.sys2
        prod_m = self.m1 * self.m2
        t, y2 = ps.step_response(prod_m)
        _, y1 = ctrl.step_response(prod_sys, t)
        self.assertTrue(np.allclose(y1, y2, atol=1e-1), "multiply is broken")
        y3, _, _ = (3 * self.m1).get_model_info()
        y4 = self.m1.get_model_info()[0] * 3
        self.assertTrue(np.allclose(y3, y4), "scalar multiply is broken")

    def test_div(self):

        fb_sys = self.sys1 / (1 + self.sys1 * self.sys2)
        fb_m = self.m1 / self.m2
        t, y2 = ps.step_response(fb_m)
        _, y1 = ctrl.step_response(fb_sys, t)
        self.assertTrue(np.allclose(y1, y2, rtol=1e-2), "feedback is broken")
        fb_sys = ctrl.feedback(self.sys1, 1)
        fb_m = self.m1 / 1
        t, y2 = ps.step_response(fb_m)
        _, y1 = ctrl.step_response(fb_sys, t)
        self.assertTrue(np.allclose(y1, y2, rtol=1e1), "scalar feedback is"
                                    "broken")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
