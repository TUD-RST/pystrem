#
#    Copyright (C) 2017
#    by Christoph Steiger, christoph.steiger@mailbox.tu-dresden.de
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import scipy.optimize as spop
import collections
import math
import warnings
from typing import Iterable, IO, Union, Tuple, Callable, Dict
import csv
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import optimisations



class FsrModel(object):
    """Represents a Finite-Step-Response model.

    A FsrModel represents a dynamic system just with its step response.
    That means to create an instance of this class, a step response of a dynamic
    system is needed. This can either be done by using for example the
    control module, or by importing a step response from a real system.
    
    You can then connect FsrModels like you would do with other systems. For
    that you can use :func:`serial`, :func:`parallel`,
    :func:`feedback` or its corresponding operator overloads.
    
    When you have created a FsrModel either combined from other models or 
    otherwise, you can simulate with it. Do that by calling either
    :func:`step_response` or :func:`forced_response`.
    """
    _unstable_allowed = False

    def __init__(self, y: Iterable[float], t: Iterable[float],
                 u: Iterable[float] = None, optimize: bool = True) -> None:
        """Creates FsrModel from given parameters.

        u should be a step, otherwise simulation accuracy will be bad. y is
        the resulting step response of the system to input u. Internally, the
        step response is normalized to an input of amplitude 1.
        
        Args:
            y: Step response of the system.
            t: Time vector.
            u (Optional): Input which was given to the system to create y. 
                Defaults to a step of amplitude one starting at time zero.
            optimize (Optional): Indicates whether model should be optimized
                for fast simulation speed. Defaults to True.

        Raises:
            TypeError: if argument is of wrong type
            ValueError: if invalid argument is given
        """
        try:
            self._y = np.array(y, dtype=float)
        except Exception as e:
            msg = "Argument y cannot be converted to numpy array."
            raise type(e)(msg) from None
        try:
            self._t = np.array(t, dtype=float)
        except Exception as e:
            msg = "Argument t cannot be converted to numpy array."
            raise type(e)(msg) from None
        if u is not None:
            try:
                self._u = np.array(u, dtype=float)
            except Exception as e:
                msg = "Argument u cannot be converted to numpy array."
                raise type(e)(msg) from None
        else:
            # default: step with amplitude 1
            self._u = np.ones(len(self._t))
        idx = self._find_step_in_u(self._u)
        self._y = self._y[idx:]
        self._u = self._u[idx:]
        self._t = self._t[idx:]
        self._y = self._y * self._u[-1]  # normalize to step of amplitude 1
        self._u = self._u * self._u[-1]  # normalize to step of amplitude 1
        self._dt = self._t[1] - self._t[0]
        # model step size
        for i in range(1, len(self._t)):
            dt = self._t[i] - self._t[i - 1]
            if not math.isclose(self._dt, dt, rel_tol=1e-3):
                msg = "t is not equidistant."
                raise ValueError(msg)
        if len(self._u) != len(self._t):
            msg = "t and u don't have the same dimensions. t: %d, u: %d" % (
                len(self._t), len(self._u))
            raise ValueError(msg)
        self._sim_u = []
        # used in method simulate_step, stores input
        self._sim_du = []
        # used in method simulate_step, stores input delta
        # detect if system is unstable
        if not math.isclose((y[-1] - y[-2]) / self._dt, 0., abs_tol=1e-2):
            if FsrModel._unstable_allowed:
                msg = ("Your system appears to be unstable. Simulating with "
                       "unstable systems is very likely to cause simulation "
                       "errors.")
                warnings.warn(UnstableSystemWarning(msg))
            else:
                msg = ("Your system appears to be unstable. If you still want "
                       "to simulate you can call FsrModel.allow_unstable(True)"
                       " to simulate anyways.")
                raise UnstableSystemException(msg)
        self._is_optimized = False
        if optimize:
            self.optimize()

    @staticmethod
    def allow_unstable(allow: bool) -> None:
        """Allow or disallow unstable systems.
        
        Unstable systems are currently not supported and will in most cases 
        cause errors in simulation. Normally an exception is raised whenever 
        an unstable system is created, by allowing them this is changed to a 
        warning.
        
        Args:
            allow: Setting this to True will warn you when an unstable system 
                is created, setting this to False will raise an exception.
        """
        FsrModel._unstable_allowed = allow

    def _find_step_in_u(self, u: Iterable[float]) -> int:
        """ Looks for the index where the jump in _u occurs."""
        u_max = abs(max(u))
        for i in range(len(u)):
            if abs(u[i]) > (0.1 * u_max):
                return i
        return 0

    def optimize(self, delta: float = 0.01) -> None:
        """Crops the underlying response according to delta.
        
        This method crops y, t and u to show only the dynamics of the system.
        Should speed up simulation, at the cost of a slight accuracy loss.
        Will do nothing if model has already been optimized.
        
        Args:
            delta: Used to determine at what point model step response is to 
                be considered static. At the point where response last differs 
                more than delta in % relative to static value, it is
                considered static. Defaults to 0.01.
        """
        if not self._is_optimized:
            self._is_optimized = True
            static_val = self._y[-1]
            if math.isclose(static_val, 0):
                static_val = max(self._y)
            for i in range(1, len(self._t)):
                if i == (len(self._t) - 1):
                    # No dynamic detected, leave it be
                    return
                val = self._y[-i]
                current_delta = (abs((val - static_val) / static_val)) * 100.
                if current_delta > delta:
                    if (i <= 1) or (i < (0.01 * len(self._t))):
                        # only a small amount would be cut, better not cut it
                        return
                    crop_idx = len(self._y) - i + 1
                    self._y = self._y[:crop_idx]
                    self._u = self._u[:crop_idx]
                    self._t = self._t[:crop_idx]
                    return
        else:
            return

    def validate_input(
            self, t: Iterable[float], u: Iterable[float]) -> (bool, str):
        """ Validates t and u input."""
        if len(t) != len(u):
            msg = ("Given t and u dimensions do not match: len(t)=%d, "
                   "len(u)=%d") % (len(t), len(u))
            return False, msg
        for i in range(1, len(t)):
            if not math.isclose(t[i] - t[i - 1], self._dt, rel_tol=1e-3):
                msg = ("Given 't' step size does not match model step size at "
                       "index %d." % i)
                return False, msg
        return True, ""

    def _validate_model_compatibility(self, other) -> (bool, str):
        """ Validates compatibility of two models."""
        other_model_info = other.get_model_info()
        other_t = other_model_info[2]
        delta_time = other_t[1] - other_t[0]
        if delta_time != self._dt:
            return False, "Step size of given models is not the same."
        else:
            return True, ""

    def feedback(self, other: 'FsrModel', sign: int = -1) -> 'FsrModel':
        """Returns the feedback connection of this system and other.

        This is a wrapper for FsrModels __truediv__ method. This system is 
        in forwards direction and other in backwards direction. Both systems 
        time bases must match. Returned system will be normalized to a 
        step of amplitude 1.
    
        Examples::
            `
            ys3 = sys1.feedback(sys2)  # same as sys1 / sys2
    
        Args:
            other: Other system.
            sign (Optional): Sign of feedback. -1 indicates a negative feedback,
                1 a positive feedback. Defaults to -1.
    
        Returns:
            The resulting system.
    
        Raises:
            TypeError: if parameter type is not supported
        """
        if sign > 0:
            return self / (-1 * other)
        else:
            return self / other

    def __add__(self, other: Union['FsrModel', int, float]) -> 'FsrModel':

        if isinstance(other, FsrModel):
            ok, msg = self._validate_model_compatibility(other)
            if not ok:
                raise ValueError(msg)
            other_y, other_u, other_t = other.get_model_info()
            y1 = self._y
            y2 = other_y
            if len(self._t) > len(other_t):
                y = np.zeros(len(self._t))
                y2_static = y2[-1]
                for i in range(len(y2)):
                    y[i] = y1[i] + y2[i]
                for i in range(len(y2), len(y1)):
                    y[i] = y1[i] + y2_static
                return FsrModel(y, t=self._t)
            else:
                y = np.zeros(len(other_t))
                y1_static = y1[-1]
                for i in range(len(y1)):
                    y[i] = y1[i] + y2[i]
                for i in range(len(y1), len(y2)):
                    y[i] = y1_static + y2[i]
                return FsrModel(y, t=other_t, optimize=False)
        elif isinstance(other, int) or isinstance(other, float):
            new_y = np.zeros(len(self._y))
            for i in range(len(new_y)):
                new_y[i] = self._y[i] + other
            return FsrModel(new_y, t=self._t, u=self._u,
                            optimize=False)
        else:
            msg = ("Unsupported type %s." % (
                str(type(other))))
            raise TypeError(msg)

    def __radd__(self, other: Union['FsrModel', int, float]):

        return self + other

    def __sub__(self, other: Union['FsrModel', int, float]) -> 'FsrModel':

        if isinstance(other, FsrModel):
            ok, msg = self._validate_model_compatibility(other)
            if not ok:
                raise ValueError(msg)
            other_y, other_u, other_t = other.get_model_info()
            y1 = self._y
            y2 = other_y
            if len(self._t) > len(other_t):
                y = np.zeros(len(self._t))
                y2_static = y2[-1]
                for i in range(len(y2)):
                    y[i] = y1[i] - y2[i]
                for i in range(len(y2), len(y1)):
                    y[i] = y1[i] - y2_static
                return FsrModel(y, t=self._t)
            else:
                y = np.zeros(len(other_t))
                y1_static = y1[-1]
                for i in range(len(y1)):
                    y[i] = y1[i] - y2[i]
                for i in range(len(y1), len(y2)):
                    y[i] = y1_static - y2[i]
                return FsrModel(y, t=other_t, optimize=False)
        elif isinstance(other, int) or isinstance(other, float):
            new_y = np.zeros(len(self._y))
            for i in range(len(new_y)):
                new_y[i] = self._y[i] - other
            return FsrModel(new_y, t=self._t, u=self._u,
                            optimize=False)
        else:
            msg = ("Unsupported type %s." % (
                str(type(other))))
            raise TypeError(msg)

    def __rsub__(self, other: Union['FsrModel', int, float]) -> 'FsrModel':

        return -1 * self + other

    def __mul__(self, other: Union['FsrModel', int, float]) -> 'FsrModel':

        if isinstance(other, FsrModel):
            ok, msg = self._validate_model_compatibility(other)
            if not ok:
                raise ValueError(msg)
            y1 = self._y
            _, _, other_t = other.get_model_info()
            time = self._dt * range(len(self._t) + len(other_t))
            u = np.zeros(len(time))
            for i in range(len(y1)):
                u[i] = y1[i]
            for i in range(len(y1), len(u)):
                u[i] = y1[-1]
            _, y2 = forced_response(other, time, u)
            return FsrModel(y2, t=time)
        elif isinstance(other, int) or isinstance(other, float):
            new_y = np.zeros(len(self._y))
            for i in range(len(new_y)):
                new_y[i] = self._y[i] * other
            return FsrModel(new_y, t=self._t, u=self._u,
                            optimize=False)
        else:
            msg = ("Unsupported type %s." % (
                str(type(other))))
            raise TypeError(msg)

    def __rmul__(self, other: Union['FsrModel', int, float]) -> 'FsrModel':

        return self * other

    def __truediv__(self, other: Union['FsrModel', int, float]) -> 'FsrModel':

        if isinstance(other, FsrModel):
            ok, msg = self._validate_model_compatibility(other)
            if not ok:
                raise ValueError(msg)
            u_tilde = self._y
            p = (1 + (self * other))
            p_out, _, _ = p.get_model_info()
            # This is just a wild guess.
            # TODO: Find a way to calculate exact size or guess it reliably.
            length = len(self._y) + len(p_out)
            y = np.zeros(length)
            # Calculate new model output. Formula is as follows:
            # y[k] = y[k-1] + (u_tilde - (y[k-1]-y[k-2])*p_out[2] -
            # (y[k-2]-y[k-3])*p_out[3] - ...) / p_out[0]
            for i in range(1, length):
                y[i] = y[i - 1]
                if i < len(u_tilde):
                    right_side = u_tilde[i]
                else:
                    right_side = u_tilde[-1]
                for j in range(1, i):
                    if j < len(p_out) - 1:
                        right_side -= p_out[j + 1] / p_out[0] * \
                                      (y[i - j] - y[i - j - 1])
                    else:
                        right_side -= p_out[-1] / p_out[0] * \
                                      (y[i - j] - y[i - j - 1])
                y[i] += right_side
            time = self._dt * np.arange(length)
            return FsrModel(y=y, t=time)
        elif isinstance(other, int) or isinstance(other, float):
            sys2 = FsrModel([other, other], t=[0, self._dt])
            return self / sys2
        else:
            msg = ("Unsupported type %s." % (
                str(type(other))))
            raise TypeError(msg)

    def __rtruediv__(self, other) -> 'FsrModel':

        sys2 = FsrModel(np.array([other, other]), t=np.array([0, self._dt]),
                        optimize=False)
        return sys2 / self

    def simulate_step(self, u: float) -> float:
        """Simulate a single step for input signal value u.
        
        This method is intended to be used for integration in simulation
        algorithms. Make sure simulation time step is equal to model
        time step. Internally a memory of last inputs is created, it can be
        cleared by calling clear_sim_mem. Do that if you want to use a single 
        model instance for multiple simulations.
        
        Args:
            u: Input signal value.
            
        Raises:
            TypeError: if input argument is of wrong type.
        """
        if len(self._sim_u) == 0:
            self._sim_du.append(u)
        else:
            last_u = self._sim_u[-1]
            self._sim_du.append(u - last_u)
        self._sim_u.append(u)
        out = 0
        for i in range(len(self._sim_du)):
            if i < len(self._y):
                out += self._sim_du[-i - 1] * self._y[i]
            else:
                out += self._sim_du[-i - 1] * self._y[-1]
        return out

    def clear_sim_mem(self) -> None:
        """Clears simulation memory.
        
        Use this method if you want use this model instance for multiple 
        simulations with :func:`simulate_step`. This makes sure there is 
        nothing left from the last simulation.
        """
        self._sim_du = []
        self._sim_u = []

    def get_model_info(self) \
            -> Tuple[Iterable[float], Iterable[float], Iterable[float]]:
        """Gets info on the model.

        Returns a tuple with relevant information stored in the model.

        Returns:
            A tuple with following entries

            * y : Step response of system.
            * u : Input to create y.
            * t : Model timebase.
        """
        # array[:] creates a copy of that array. We don't want to return
        # references here.
        return self._y[:], self._u[:], self._t[:]


class Mpc(object):
    """A class representing a model-predictive-controller.
    
    This class allows to simulate a MPC with a SISO or MIMO system.
    The most important parameters of a MPC are the time horizon in which
    the MPC simulates the given system, and the cost functional.
    
    There are only rough guidelines to decide what the time horizon should
    be. Generally speaking, a shorter time horizon leads to a more aggressive
    control strategy and thus higher controller output signals. If the MPC 
    manages to keep the system stable, the time horizon also equates roughly
    to the rise time of the system response.
    
    A cost functional can be specified, with :func:`set_cost_func`.
    If no cost functional is set, a default is used.
    This function is defined as follows::
        
       def _default_cost_func(u: Iterable, y_d: Iterable,
                           y: Iterable, time: Iterable) -> float:
        
            j = 0
            for i in range(len(time)):
                t = time[i]
                y_cost = 0
                for j in range(len(y_d)):
                    y_cost += (y_d[j][i]-y[j][i])**2
                j += y_cost*t**2
            return j
            
    This cost functional should in most cases generate good results.
    
    The minimizing routine used is ``scipy.optimize.minimize``. You can affect
    its behavior by setting additional parameters with 
    :func:`set_minimizer_kwargs`. For further details see its documentation.
    """

    def __init__(self, cost_func: Callable = None):
        """ Creates a model-predictive controller from parameters.
        
        Args:
            cost_func (Optional): A function representing a cost functional. 
                Must be of form ``f(u, y_d, y, t)``, where u is controller 
                output, y_d is desired plant output, y is plant output and t is
                the time vector.
        """

        self._cost_func = cost_func if cost_func else Mpc._default_cost_func
        self._input_func = None
        self._minimizer_kwargs = {}
        self._constraints = []

    def set_minimizer_kwargs(self, **kwargs) -> None:
        """Sets additional parameters of the underlying minimize routine.
        
        The used routine is ``scipy.optimize.minimize``. For further details 
        see its documentation. The ``args``, ``bounds`` and ``constraints`` 
        parameters can not be set. For ``constraints`` use 
        :func:`set_constraints`.
        """
        if "args" in kwargs:
            msg = ("'args' parameter can not be set. This certainly won't do "
                   "what you wanted it to do.")
            raise ValueError(msg)
        if "constraints" in kwargs:
            msg = ("'constraints' parameter can not be set with this function."
                   " Use set_constraints instead.")
            raise ValueError(msg)
        if "bounds" in kwargs:
            msg = ("'bounds' parameter can not be set. You should instead "
                   "use constraints.")
            raise ValueError(msg)
        self._minimizer_kwargs = kwargs

    def set_constraints(self, constraints: Iterable[Dict]) -> None:
        """Sets equality and inequality constraints for the minimize routine.
        
        Each constraint is needed in a dictionary, which needs following
        entries:
            
            type: str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            
            fun: callable
                The function defining the constraint.
        
        The given functions must be of form ``f(u, y, t)``. Equality 
        constraints means the constraint function result is to be zero,
        inequality constraints means constraint function result is to be
        non-negative. This is merely a wrapper for the constraint functionality
        in ``scipy.optimize.minimize``. See its documentation for further 
        details.
        
        Args:
            constraints: Either a dict or iterable of dicts which looks as
                described above.
        
        """
        self._constraints = constraints
        if type(self._constraints) is not list:
            self._constraints = [self._constraints]
        for constraint in self._constraints:
            # remove unnecessary parameters for constraint functions
            f = constraint["fun"]
            constraint["fun"] = lambda m, u, y, t: f(u, y, t)

    def set_cost_func(self, func: Callable) -> None:
        """Sets the cost functional which is used for the MPC.
        
        The given function must be of form ``f(u, y_d, y, t)``. All input 
        parameters are vectors of fitting dimension derived from the system. 
        u is controller output, y_d plant desired output, y plant output and t 
        time.
                
        Args:
            func: The cost functional.
        """
        self._cost_func = func

    def simulate(self, sys, y_d: Iterable, time_horizon: Iterable[float],
                 time: Iterable[float]) -> Tuple[Iterable, Iterable, Iterable]:
        """Simulates a given system connected with the MPC.
        
        This simulation combines a model-predictive-control algorithm with a
        simulation of the system. The MPC tries to have system outputs match 
        ``y_d``. 
        
        Examples:
            Our example system has 2 inputs and 2 outputs, with 4 FsrModels 
            describing each I/O behavior. A time vector t and a time horizon
            vector t_horizon are given::
            
                sys = numpy.ndarray((2, 2), pystrem.FsrModel)
                sys[0][0] = model_11  # output 1 from input 1
                sys[0][1] = model_12  # output 1 from input 2
                sys[1][1] = model_22  # output 2 from input 2
                sys[1][0] = model_21  # output 2 from input 1
                mpc = pystrem.Mpc()
                y_d = numpy.ndarray((2, len(t)))
                y_d[0] = numpy.ones(len(t))   # step for input 1
                y_d[1] = numpy.zeros(len(t))  # and nothing for input 2
                t, y, u = mpc.simulate(sys, y_d, t_horizon, t)
            
        Args:
            sys: System to simulate.
            y_d: Desired outputs for each system output.
                Must fit system dimensions. Each entry has same length as time.
            time_horizon: This is the time horizon the MPC simulates the 
                system with. Generally, if this time is shorter the control 
                strategy is more aggressive.
            time: Time vector.
            
        Returns:
            A tuple (t, y, u) where t is the time vector, y is the vector
            of system outputs and u the vector of system inputs.
        
        Raises:
            ValueError: if input argument are not compatible with each other or
                the system.
            TypeError: if input argument is of wrong type.
        """
        sys = np.array(sys)
        if sys.shape == ():
            # This means a single FsrModel was passed.
            # A wrapper array is needed for it.
            sys_wrapper = np.ndarray((1, 1), FsrModel)
            sys_wrapper[0] = sys
            sys = sys_wrapper
        for i in range(len(sys)):
            for j in range(len(sys[i])):
                if not isinstance(sys[i][j], FsrModel):
                    msg = ("Unsupported type: Got %s at position "
                           "(%d, %d).") % (type(sys[i][j]), i, j)
                    raise TypeError(msg)
        t_dt = time[1] - time[0]
        th_dt = time_horizon[1] - time_horizon[0]
        if t_dt != th_dt:
            msg = ("Timebase of time (%f) and time horizon (%f) do not "
                   "match.") % (t_dt, th_dt)
            raise ValueError(msg)
        y_d = np.array(y_d)
        try:
            y_d.shape[1]
        except IndexError:
            # This means we reveived a single y_d array.
            # Create a wrapper array for it.
            y_d_wrapper = np.ndarray((1, len(y_d)))
            y_d_wrapper[0] = y_d
            y_d = y_d_wrapper
        if len(y_d) != len(sys):
            msg = ("y_d dimensions do not fit system: len(y_d)=%d, "
                   "len(sys)=%d.") % (len(y_d), len(sys))
            raise ValueError(msg)
        for i in range(len(y_d)):
            if len(y_d[i]) != len(time):
                msg = ("Desired output %d has wrong length: %d, "
                       "expected %d") % (i, len(y_d[i]), len(time))
                raise ValueError(msg)
        min_kwargs = self._minimizer_kwargs.copy()
        if "method" not in min_kwargs:
            if len(self._constraints) == 0:
                # No constraints, use Nelder-Mead as default.
                min_kwargs["method"] = "Nelder-Mead"
            else:
                # Constraints, use COBYLA as default.
                min_kwargs["method"] = "COBYLA"

        # Add additional data for simulation after time.
        # This is needed because we have to simulate for time_horizon even
        # after normal simulation time is reached.
        y_d_ext = np.ndarray((len(y_d), len(time) + len(time_horizon)))
        for i in range(len(y_d)):
            y_d_ext[i] = np.r_[y_d[i], np.ones(len(time_horizon)) * y_d[i][-1]]
        ext_time = np.arange(0, time[-1] + time_horizon[-1] + 2 * t_dt, t_dt)
        y = np.zeros((len(sys), len(ext_time)))
        u = np.zeros((len(sys[0]), len(time)))
        u_0 = np.zeros(len(sys[0]))  # Initial guesses for minimized parameters
        # Is 0 because we describe it as the differences from last value.
        for i in range(len(time)):
            # Creating frames of all relevant data for the cost functional
            time_frame = ext_time[i:i + len(time_horizon)]
            y_frame = np.ndarray((len(y), len(time_frame)))
            y_d_frame = np.ndarray((len(y), len(time_frame)))
            u_frame = np.ones((len(u), len(time_frame)))
            for j in range(len(u_frame)):
                u_frame[j] *= u[j][i - 1]
            for j in range(len(y_d_frame)):
                y_frame[j] = y[j][i:len(time_frame) + i]
                y_d_frame[j] = y_d_ext[j][i:len(time_frame) + i]
            # add still missing information for constraints
            for constraint in self._constraints:
                constraint["args"] = (u_frame, y_frame, time_frame)
            res = spop.minimize(self._cost_func_wrapper, u_0,
                                (sys, y_d_frame, y_frame, u_frame, time_frame),
                                constraints=self._constraints,
                                **min_kwargs)
            if not res.success:
                # This sometimes happens when response is static or getting 
                # unstable, continue anyways, but warn user.
                msg = "Minimizing failed with following message:\n%s" % (
                    res.message)
                warnings.warn(RuntimeWarning(msg))
            for j in range(len(res.x)):
                u[j][i] = u[j][i - 1] + res.x[j]
            for j in range(len(sys)):
                row = sys[j]
                for k in range(len(row)):
                    model = row[k]
                    _, out = step_response(model,
                                           ext_time[:len(ext_time) - i],
                                           res.x[k])
                    for n in range(len(out)):
                        y[j][i + n] += out[n]
        y_out = np.ndarray((len(y), len(time)))
        for i in range(len(y)):
            y_out[i] = y[i][:len(time)]  # only show y for requested time
        return time, y_out, u

    def _cost_func_wrapper(self, m, sys, y_d_frame, y_frame, u_frame,
                           time_frame) -> float:
        """A wrapper which takes care of everything around the cost
        functional """

        y_mpc = np.zeros((len(sys), len(time_frame)))
        for i in range(len(sys)):
            row = sys[i]
            for j in range(len(row)):
                u_frame[j] += m[j]
                model = row[j]
                _, out = step_response(model, time_frame, m[j])
                y_mpc[i] += out
        y_mpc += y_frame
        return self._cost_func(u_frame, y_d_frame, y_mpc, time_frame)

    @staticmethod
    def _default_cost_func(u: Iterable, y_d: Iterable,
                           y: Iterable, time: Iterable) -> float:
        """The default cost functional used if none is given"""

        j = 0
        for i in range(len(time)):
            t = time[i]
            y_cost = 0
            for j in range(len(y_d)):
                y_cost += (y_d[j][i] - y[j][i]) ** 2
            j += y_cost * t ** 2
        return j


def step_response(sys: FsrModel, t: Iterable[float] = None,
                  amplitude: float = 1) -> Tuple[
                                                 Iterable[float],
                                                 Iterable[float]]:
    """Simulates the the step response of given system.

    Args:
        sys: System to simulate.
        t: Time vector.
        amplitude (Optional): Step amplitude. Defaults to 1.

    Returns:
        A touple (t, y) of two iterables. t is the time vector used for
        simulation. y is the step response.

    Raises:
        ValueError: if input arguments are not compatible with each other
            or the system.
        TypeError: if argument sys is of wrong type.
    """

    y, _, sys_t = sys.get_model_info()
    if t is None:
        t = sys_t
    if len(sys_t) < len(t):
        # append static values to y, if t is longer than y
        y_extension = np.ones(len(t) - len(sys_t)) * y[-1]
        y = np.r_[y, y_extension]
    elif len(sys_t) > len(t):
        y = y[:len(t)]
    y = y * amplitude
    return t, y


def forced_response(sys: FsrModel, t: Iterable[float], u: Iterable[float]) \
        -> Tuple[Iterable[float], Iterable[float]]:
    """Simulates the response of this system to given input.

    Args:
        sys: The system.
        t: Time vector.
        u: Input vector.

    Returns:
        A touple (t, y) of two iterables. t is the time vector used for
        simulation. y is the step response.

    Raises:
        ValueError: if input arguments are not compatible with each other
            or the system.
        TypeError: if argument sys is of wrong type.
    """
    try:
        sys_y, sys_u, sys_t = sys.get_model_info()
    except AttributeError:
        msg = ("Unsupported type %s for parameter sys." % (
            str(type(sys))))
        raise TypeError(msg)
    ok, msg = sys.validate_input(t, u)
    if not ok:
        raise ValueError(msg)
    c_buf = collections.deque(
        np.zeros(len(sys_t)), len(sys_t))
    t_idx = 0
    y = np.zeros(len(t))
    out_of_buf_value = 0
    # Calculates output according to following formula:
    # y[k] = (u[k]-u[k-1])*_y[1] + (u[k-1]-u[k-2])*_y[2] + ...
    for _ in t:
        if t_idx != 0:
            du_rel = (u[t_idx] - u[t_idx - 1])
        else:
            du_rel = u[t_idx]
        # Whenever we add a value to our buffer we need to remember the
        # ones thrown out of it.
        if c_buf[0] != 0:
            out_of_buf_value += c_buf[0] * sys_y[-1]
        c_buf.append(du_rel)
        step_out = 0
        for buf_pos in range(len(c_buf)):
            # buffer is filled from right side
            step_out += c_buf[-(buf_pos + 1)] * sys_y[buf_pos]
        y[t_idx] = step_out + out_of_buf_value
        t_idx += 1
    return t, y

def forced_response_cython_wrapper(sys: FsrModel, t: Iterable[float],
                                   u: Iterable[float]):
    try:
        sys_y, _, sys_t = sys.get_model_info()
    except AttributeError:
        msg = ("Unsupported type %s for parameter sys." % (
            str(type(sys))))
        raise TypeError(msg)
    ok, msg = sys.validate_input(t, u)
    if not ok:
        raise ValueError(msg)
    return optimisations.forced_response(sys_y, sys_t, t, u)


def parallel(sys1: FsrModel, sys2: FsrModel, sign: int = 1) -> FsrModel:
    """Returns the parallel connection of sys1 and sys2.

    This is a wrapper for FsrModels __add__ and __sub__ methods.
    Both systems time bases must match. If subtraction instead of addition is
    wanted, use ``sign=-1``.

    Examples::
        
        sys3 = parallel(sys1, sys2)  # same as sys3 = sys1 + sys2
        sys3 = parallel(sys1, sys2, -1)  # same as sys3 = sys1 - sys2

    Args:
        sys1: First system.
        sys2: Second system.
        sign (Optional): Sign of parallel connection. Returns sys1 - sys2 if
            negative. Defaults to 1.

    Returns:
        The resulting system.

    Raises:
        TypeError: if parameter type is not supported.
    """

    if sign < 0:
        return sys1 - sys2
    else:
        return sys1 + sys2


def series(sys1: FsrModel, sys2: FsrModel) -> FsrModel:
    """Returns the serial connection of sys1 and sys2.

    This is a wrapper for FsrModels __mul__ method. Both systems time bases 
    must match.

    Examples::
    
        sys3 = series(sys1, sys2)  # same as sys3 = sys1 * sys2

    Args:
        sys1: First system.
        sys2: Second system.

    Returns:
        The resulting system.

    Raises:
        TypeError: if parameter type is not supported.
    """
    return sys1 * sys2


def feedback(sys1: FsrModel, sys2: FsrModel, sign: int = -1) -> FsrModel:
    """Returns the feedback connection of sys1 and sys2.

    This is a wrapper for FsrModels __truediv__ method. ``sys1`` is in forwards
    direction and ``sys2`` in backwards direction. Both systems time bases
    must match.

    Examples::
    
        sys3 = feedback(sys1, sys2)  # same as sys1 / sys2

    Args:
        sys1: First system.
        sys2: Second system.
        sign (Optional): Sign of feedback. -1 indicates a negative feedback,
            1 a positive feedback. Defaults to -1.

    Returns:
        The resulting system.

    Raises:
        TypeError: if parameter type is not supported.
    """
    if sign > 0:
        return sys1 / (-1 * sys2)
    else:
        return sys1 / sys2


def import_csv(filehandle: IO, delimiter: str = ',',
               quotechar: str = '"') -> FsrModel:
    """Imports a system from a CSV file.

    Format is "time, input, output" in each line.
    Expects a file handle in read-mode. CSV dialect can be specified by
    ``delimiter`` and ``quotechar`` parameters. 
    Imported models will not be optimized.

    Args:
        filehandle: Handle of the file.
        delimiter (Optional): Delimiter string for CSV dialect.
            Defaults to ','.
        quotechar (Optional): Quote character string for CSV dialect.
            Defaults to '"'.

    Returns:
        Model found in file.

    Raises:
        IOError: if file can not be properly read.
    """

    reader = csv.reader(filehandle, delimiter=delimiter, quotechar=quotechar,
                        lineterminator='\n')
    t = []
    u = []
    y = []
    try:
        for row in reader:
            t.append(float(row[0]))
            u.append(float(row[1]))
            y.append(float(row[2]))
    except Exception:
        msg = ("Something went wrong while reading the file. The format in the "
               "file might be unsupported.")
        raise IOError(msg) from None
    return FsrModel(y, t=t, optimize=False)


def export_csv(model: FsrModel, filehandle: IO, delimiter: str = ',',
               quotechar: str = '"') -> None:
    """Exports a system to a CSV file.

    Expects a file handle of the file in write-mode. CSV dialect can be
    specified with ``delimiter`` and ``quotechar`` parameters.

    Args:
        model: Model to export.
        filehandle: Handle of the file.
        delimiter (Optional): Delimiter string for CSV dialect.
            Defaults to ','.
        quotechar (Optional): Quote character string for CSV dialect.
            Defaults to '"'.

    Raises:
        TypeError: if model parameter is of wrong type.
    """
    if isinstance(model, FsrModel):
        writer = csv.writer(filehandle, delimiter=delimiter,
                            quotechar=quotechar,
                            lineterminator="\n")
        y, u, t = model.get_model_info()
        for i in range(len(t)):
            writer.writerow([t[i], u[i], y[i]])
    else:
        msg = "Expected type <class 'FsrModel'>, got type %s." % (
            str(type(model)))
        raise TypeError(msg)


class UnstableSystemException(Exception):
    pass


class UnstableSystemWarning(Warning):
    pass
