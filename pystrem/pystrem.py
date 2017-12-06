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
import collections
import math
import warnings
from typing import Iterable, IO, Union, Tuple
import csv


class FsrModel(object):
    """Handles simulation of Finite-Step-Response models.

    FsrModel contains all necessary information to simulate with
    FSR models. It is intended to be used exactly as a transfer function
    would be. TODO: expand this
    """
    _unstable_allowed = False

    def __init__(self, y: Iterable[float], t: Iterable[float],
                 u: Iterable[float]=None, optimize: bool=True) -> None:
        """Creates FsrModel from given parameters.

        u should be a step, otherwise simulation accuracy will be bad. y is
        the resulting step response of the system to input u.
        
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
        self._u_0 = self._u[-1]
        # considered step amplitude for all purposes, important for calculation
        self._dt = self._t[1] - self._t[0]
        # model step size
        for i in range(1, len(self._t)):
            dt = self._t[i] - self._t[i-1]
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
        if not math.isclose((y[-1]-y[-2]) / self._dt, 0., abs_tol=1e-2):
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

    def optimize(self, delta: float=0.01) -> None:
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
                if (current_delta > delta):
                    if ((i <= 1) or (i < (0.01 * len(self._t)))):
                        # only a small amount would be cut, better not cut it
                        return
                    crop_idx = len(self._y) - i + 1
                    self._y = self._y[:crop_idx]
                    self._u = self._u[:crop_idx]
                    self._t = self._t[:crop_idx]
                    return
        else:
            return

    def _validate_input(
            self, t: Iterable[float], u: Iterable[float]) -> (bool, str):
        """ Validates t and u input."""
        if len(t) != len(u):
            msg = "Given 't' and 'u' dimensions do not match."
            return False, msg
        for i in range(1, len(t)):
            if not math.isclose(t[i] - t[i - 1], self._dt, rel_tol=1e-3):
                msg = ("Given 't' step size does not match model step size at "
                       "index %d." % i)
                return False, msg
        return True, ""

    def _validate_model_compatibility(self, other) -> (bool, str):
        """ Validates compatibility of two models."""
        delta_time = other._t[1] - other._t[0]
        if delta_time != self._dt:
            return False, "Step size of given models is not the same."
        else:
            return True, ""

    def feedback(self, other: 'FsrModel', sign: int=-1) -> 'FsrModel':
        """Returns the feedback connection of this system and other.

        This is a wrapper for FsrModels __truediv__ method. This system is 
        in forwards direction and other in backwards direction. Both systems 
        timebases must match. Returned system will be normalized to a 
        step of amplitude 1.
    
        Examples:
            ``sys3 = sys1.feedback(sys2)  # same as sys1 / sys2``
    
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
            other_u_0 = other_u[-1]
            y1 = self._y / self._u_0
            y2 = other_y / other_u_0
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
            other_u_0 = other_u[-1]
            y1 = self._y / self._u_0
            y2 = other_y / other_u_0
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
            y1 = self._y / self._u_0
            _, _, other_t = other.get_model_info()
            time = self._dt * \
                range(len(self._t) + len(other_t))
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
            u_tilde = self._y / self._u_0
            p = (1 + (self * other))
            p_out, _, _ = p.get_model_info()
            # This is just a wild guess.
            # TODO: Find a way to calculate exact size or guess it reliably.
            length = len(self._y) + len(p_out)
            y = np.zeros(length)
            y[0] = 0
            # Calculate new model output. Formula is as follows:
            # y[k] = y[k-1] + (u_tilde - (y[k-1]-y[k-2])*p_out[2] -
            # (y[k-2]-y[k-3])*p_out[3] - ...) / p_out[1]
            for i in range(1, length):
                y[i] = y[i - 1]
                if i < len(u_tilde):
                    right_side = u_tilde[i]
                else:
                    right_side = u_tilde[-1]
                for j in range(1, i):
                    if j < len(p_out) - 1:
                        right_side -= p_out[j + 1] / p_out[1] * \
                            (y[i - j] - y[i - j - 1])
                    else:
                        right_side -= p_out[-1] / p_out[1] * \
                            (y[i - j] - y[i - j - 1])
                y[i] += right_side
            time = self._dt * np.arange(length)
            # if system has differentiating properties, this fixes the value @t=0
            if not math.isclose(0., (self._y[0] / self._u_0) / 
                                (1 + other._y[0] / other._u_0)):
                y = y[1:]
                time = time[:-1]
            # TODO: this is a hack, find mathematical solution or explanation
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
            self._sim_du.append(u-last_u)
        self._sim_u.append(u)
        out = 0
        for i in range(len(self._sim_du)):
            if i < len(self._y):
                out += self._sim_du[-i-1] * self._y[i]
            else:
                out += self._sim_du[-i-1] * self._y[-1]
        return out
    
    def clear_sim_mem(self) -> None:
        """Clears simulation memory.
        
        Use this method if you want use this model instance for multiple 
        simulations with simulate_step method. This makes sure there is nothing
        left from the last simulation.
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
        return (self._y[:], self._u[:], self._t[:])


def step_response(sys: FsrModel, t: Iterable[float]=None,
                  amplitude: float=1) -> Tuple[Iterable[float], Iterable[float]]:
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
    if t is not None:
        u = amplitude * np.ones(len(t))
    else:
        _, _, t = sys.get_model_info()
        u = amplitude * np.ones(len(t))
    t, y = forced_response(sys, t, u)
    return t, y

def forced_response(sys: FsrModel, t: Iterable[float], u: Iterable[float]) \
        -> Tuple[Iterable[float], Iterable[float]]:
    """Simulates the response of this system to given input.

    Args:
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
    except:
        msg = ("Unsupported type %s for parameter sys." % (
            str(type(sys))))
        raise TypeError(msg)
    sys_u_0 = sys_u[-1]
    ok, msg = sys._validate_input(t, u)
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
            du_rel = (u[t_idx] - u[t_idx - 1]) / sys_u_0
        else:
            du_rel = u[t_idx] / sys_u_0
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

def parallel(sys1: FsrModel, sys2: FsrModel, sign: int=1) -> FsrModel:
    """Returns the parallel connection of sys1 and sys2.

    This is a wrapper for FsrModels __add__ and __sub__ methods.
    Both systems timebases must match. Returned system will be
    normalized to a step of amplitude 1. If subtraction instead of addition is
    wanted, use sign=-1.

    Examples: 
        ``sys3 = parallel(sys1, sys2)  # same as sys3 = sys1 + sys2``

        ``sys3 = parallel(sys1, sys2, -1)  # same as sys3 = sys1 - sys2``

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

    This is a wrapper for FsrModels __mul__ method. Both systems timebases must
    match. Returned system will be normalized to a step of amplitude 1.

    Examples:
        ``sys3 = series(sys1, sys2)  # same as sys3 = sys1 * sys2``

    Args:
        sys1: First system.
        sys2: Second system.

    Returns:
        The resulting system.

    Raises:
        TypeError: if parameter type is not supported.
    """
    return sys1 * sys2


def feedback(sys1: FsrModel, sys2: FsrModel, sign: int=-1) -> FsrModel:
    """Returns the feedback connection of sys1 and sys2.

    This is a wrapper for FsrModels __truediv__ method. sys1 is in forwards
    direction and sys2 in backwards direction. Both systems timebases 
    must match. Returned system will be normalized to a step of amplitude 1.

    Examples:
        ``sys3 = feedback(sys1, sys2)  # same as sys1 / sys2``

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


def import_csv(filehandle: IO, delimiter: str=',',
                quotechar: str='"') -> FsrModel:
    """Imports a system from a CSV file.

    Expects a file handle in read-mode. CSV dialect can be specified by
    delimiter and quotechar parameters. Imported models will not be optimized.

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
    except:
        msg = ("Something went wrong while reading the file. The format in the "
               "file might be unsupported.")
        raise IOError(msg) from None
    return FsrModel(y, t=t, optimize=False)


def export_csv(model: FsrModel, filehandle: IO, delimiter: str=',', 
               quotechar: str='"') -> None:
    """Exports a system to a CSV file.

    Expects a file handle of the file in write-mode. CSV dialect can be
    specified with delimiter and quotechar parameters.

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
