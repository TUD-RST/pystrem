#
#    Copyright (C) 2017
#    by Christoph Steiger, TODO: add email here
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
from typing import Iterable, Dict, IO
import csv


class FsrModel(object):
    """Handles simulation of Finite-Step-Response models.

    FsrModel contains all necessary information to simulate with
    FSR models. It is intended to be used exactly as a transfer function
    would be. TODO: maybe explain a bit more?

    Attributes:
        show_warnings: A boolean indicating whether warnings should be used.
    """

    def __init__(self, y: Iterable[float], t: Iterable[float],
                 u: Iterable[float]=None, max_delta: float=0.01,
                 show_warnings: bool=True) -> None:
        """Inits FsrModel from given parameters.

        u must be a step and y the resulting step response of the system,
        otherwise simulation accuracy will be bad. The response will be
        shortened to only show the dynamic range of the system. What still is
        considered dynamic and what not can be influenced by max_delta.

        Args:
            y: Step response of the system.
            t: time vector.
            u (Optional): Input which was given to the system to create y. 
                Defaults to a step of height one starting at index zero.
            max_delta (Optional): Used to determine at what point y is to 
                be considered static. At the point where response last differs 
                more than max_delta in % relative to static value, it is
                considered static. Defaults to 0.01.
            show_warnings (Optional): Indicates whether to show warnings.
                Defaults to True.
        """
        
        self.show_warnings = show_warnings
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
            # default: step with height 1
            self._u = np.ones(len(self._t))
        # thrown Errors: ValueError, TypeError
        idx = self._find_step_in_u(self._u)
        self._y = self._y[idx:]
        self._u = self._u[idx:]
        # considered step height for all purposes, important for calculation
        self._u_0 = self._u[-1]
        # difference in percent to static value to consider response done
        self._max_delta = max_delta
        self._end_of_dynamic, self._static_val = self._find_dynamic_range(
            self._y, t)
        self._y = self._y[:self._end_of_dynamic]
        self._u = self._u[:self._end_of_dynamic]
        self._t = self._t[:self._end_of_dynamic]
        self._dt = self._t[1] - self._t[0]  # model step size

    def _find_step_in_u(self, u: Iterable[float]) -> int:
        """ Looks for the index where the jump in _u occurs."""
        u_max = abs(max(u))
        for i in range(len(u)):
            if abs(u[i]) > (0.1 * u_max):
                return i
        return 0

    def _find_dynamic_range(
            self, y: Iterable[float], t: Iterable[float]) -> (int, float):
        """ Looks for the end index of the dynamic phase of a step response."""
        static_val = y[-1]
        if math.isclose(static_val, 0):
            static_val = max(y)
        for i in range(1, len(t)):
            if i == (len(t) - 1):
                msg = ("Could not detect any dynamic in step response. "
                       "This might mean your system is static or max_delta "
                       "is set too high.")
                warnings.warn(msg, RuntimeWarning)
                return len(y), static_val
            val = self._y[-i]
            current_delta = (abs((val - static_val) / static_val)) * 100.
            if (current_delta > self._max_delta):
                if ((i <= 1) or (i < (0.01 * len(t)))) and self.show_warnings:
                    msg = ("End of dynamic is very close to end of "
                           "step response.\nYour system might be unstable or "
                           "your step response does not show the full "
                           "dynamics of your system.")
                    warnings.warn(msg, RuntimeWarning)
                return len(y) - i + 1, static_val

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

    def step_response(self, t: Iterable[float]=None,
                      height: float=1) -> (Iterable[float], Iterable[float]):
        """Simulates the the step response of this system.

        Args:
            t: Time vector.
            height (Optional): Step height. Defaults to 1.

        Returns:
            A touple (t, y) of two iterables. t is the time vector used for
            simulation. y is the step response.

        Raises:
            ValueError: The input arguments are not compatible with each other
                or the system.
        """
        if t is not None:
            u = height * np.ones(len(t))
        else:
            u = height * np.ones(len(self._t))
            t = self._t
        t, y = self.forced_response(t, u)
        return t, y

    def forced_response(self, t: Iterable[float], u: Iterable[float]) \
            -> (Iterable[float], Iterable[float]):
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
        """
        ok, msg = self._validate_input(t, u)
        if not ok:
            raise ValueError(msg)
        c_buf = collections.deque(
            np.zeros(self._end_of_dynamic), self._end_of_dynamic)
        t_idx = 0
        y = np.zeros(len(t))
        out_of_buf_value = 0
        # Calculates output according to following formula:
        # y[k] = (u[k]-u[k-1])*_y[1] + (u[k-1]-u[k-2])*_y[2] + ...
        for _ in t:
            if t_idx != 0:
                du_rel = (u[t_idx] - u[t_idx - 1]) / self._u_0
            else:
                du_rel = u[t_idx] / self._u_0
            # Whenever we add a value to our buffer we need to remember the
            # ones thrown out of it.
            if c_buf[0] != 0:
                out_of_buf_value += c_buf[0] * self._y[-1]
            c_buf.append(du_rel)
            step_out = 0
            for buf_pos in range(len(c_buf)):
                # buffer is filled from right side
                step_out += c_buf[-(buf_pos + 1)] * self._y[buf_pos]
            y[t_idx] = step_out + out_of_buf_value
            t_idx += 1
        return t, y

    def _validate_model_compatibility(self, other) -> (bool, str):
        """ Validates compatibility of two models."""
        delta_time = other._t[1] - other._t[0]
        if delta_time != self._dt:
            return False, "Step size of given models is not the same."
        else:
            return True, ""

    def __add__(self, other: 'FsrModel') -> 'FsrModel':

        if isinstance(other, FsrModel):
            ok, msg = self._validate_model_compatibility(other)
            if not ok:
                raise ValueError(msg)
            # TODO: accessing privates here is not nice, use get_model_info
            y1 = self._y / self._u_0
            y2 = other._y / other._u_0
            delta = self._max_delta if (
                self._max_delta <= other._max_delta) else other._max_delta
            if len(self._t) > len(other._t):
                y = np.zeros(len(self._t))
                y2_static = y2[-1]
                for i in range(len(y2)):
                    y[i] = y1[i] + y2[i]
                for i in range(len(y2), len(y1)):
                    y[i] = y1[i] + y2_static
                return FsrModel(y, t=self._t, max_delta=delta,
                                show_warnings=False)
            else:
                y = np.zeros(len(other._t))
                y1_static = y1[-1]
                for i in range(len(y1)):
                    y[i] = y1[i] + y2[i]
                for i in range(len(y1), len(y2)):
                    y[i] = y1_static + y2[i]
                return FsrModel(y, t=other._t, max_delta=delta,
                                show_warnings=False)
        else:
            msg = ("Operator + is only supported for objects of type "
                   "<class 'FsrModel'>, got type %s." % (
                       str(type(other))))
            raise TypeError(msg)

    def __sub__(self, other: 'FsrModel') -> 'FsrModel':

        if isinstance(other, FsrModel):
            ok, msg = self._validate_model_compatibility(other)
            if not ok:
                raise ValueError(msg)
            # TODO: accessing privates here is not nice, use get_model_info
            y1 = self._y / self._u_0
            y2 = other._y / other._u_0
            delta = self._max_delta if (
                self._max_delta <= other._max_delta) else other._max_delta
            if len(self._t) > len(other._t):
                y = np.zeros(len(self._t))
                y2_static = y2[-1]
                for i in range(len(y2)):
                    y[i] = y1[i] - y2[i]
                for i in range(len(y2), len(y1)):
                    y[i] = y1[i] - y2_static
                return FsrModel(y, t=self._t, max_delta=delta,
                                show_warnings=False)
            else:
                y = np.zeros(len(other._t))
                y1_static = y1[-1]
                for i in range(len(y1)):
                    y[i] = y1[i] - y2[i]
                for i in range(len(y1), len(y2)):
                    y[i] = y1_static - y2[i]
                return FsrModel(y, t=other._t, max_delta=delta,
                                show_warnings=False)
        else:
            msg = ("Operator - is only supported for objects of type "
                   "<class 'FsrModel'>, got type %s." % (
                       str(type(other))))
            raise TypeError(msg)

    def __mul__(self, other: 'FsrModel') -> 'FsrModel':

        if isinstance(other, FsrModel):
            ok, msg = self._validate_model_compatibility(other)
            if not ok:
                raise ValueError(msg)
            y1 = self._y / self._u_0
            # TODO: accessing privates here is not nice, use get_model_info
            delta = self._max_delta if (
                self._max_delta <= other._max_delta) else other._max_delta
            time = self._dt * \
                range(self._end_of_dynamic + other._end_of_dynamic)
            u = np.zeros(len(time))
            for i in range(len(y1)):
                u[i] = y1[i]
            for i in range(len(y1), len(u)):
                u[i] = y1[-1]
            _, y2 = other.forced_response(time, u)
            return FsrModel(y2, t=time, max_delta=delta, show_warnings=False)
        else:
            msg = ("Operator * is only supported for objects of type "
                   "<class 'FsrModel'>, got type %s." % (
                       str(type(other))))
            raise TypeError(msg)

    def __truediv__(self, other: 'FsrModel') -> 'FsrModel':

        if isinstance(other, FsrModel):
            ok, msg = self._validate_model_compatibility(other)
            if not ok:
                raise ValueError(msg)
            u_tilde = self._y / self._u_0
            # Calculate 1 + G1*G2
            # TODO: accessing privates here is not nice, use get_model_info
            p = (self * other)._y
            for i in range(len(p)):
                p[i] += 1
            # This is just a wild guess.
            # TODO: Find a way to calculate exact size or guess it reliably.
            sim_len = len(self._y) + len(p)
            y = np.zeros(sim_len)
            y[0] = 0
            # Calculate new model output. Formula is as follows:
            # y[k] = y[k-1] + (u_tilde - (y[k-1]-y[k-2])*p[2] - (y[k-2]-y[k-3])*
            #        p[3] - ...) / p[1]
            for i in range(1, sim_len):
                y[i] = y[i - 1]
                if i < len(u_tilde):
                    right_side = u_tilde[i]
                else:
                    right_side = u_tilde[-1]
                for j in range(1, i):
                    if j < len(p) - 1:
                        right_side -= p[j + 1] / p[1] * \
                            (y[i - j] - y[i - j - 1])
                    else:
                        right_side -= p[-1] / p[1] * (y[i - j] - y[i - j - 1])
                y[i] += right_side
            delta = self._max_delta if (
                self._max_delta <= other._max_delta) else other._max_delta
            time = self._dt * np.arange(sim_len)
            return FsrModel(y=y, t=time, max_delta=delta, show_warnings=False)
        else:
            msg = ("Operator / is only supported for objects of type "
                   "<class 'FsrModel'>, got type %s." % (
                       str(type(other))))
            raise TypeError(msg)

    def get_model_info(self) -> Dict:
        """Gets info on the model.
        
        Returns a dictionary with relevant information stored in the model.
        
        Returns:
            A dictionary with following entries
            
            * 'y' (Iterable): Step response of system.
            * 'u' (Iterable): Input to create y.
            * 't' (Iterable): Model timebase.
            * 'delta' (float): Model parameter delta_max.
        """
        # array[:] creates a copy of that array. We don't want to return
        # references here.
        return {"y": self._y[:], "u": self._u[:], "t": self._t[:],
                "delta": self._max_delta}


def parallel(sys1: FsrModel, sys2: FsrModel, neg: bool=False) -> FsrModel:
    """Returns the parallel connection of sys1 and sys2.

    This is a wrapper for FsrModels __add__ and __sub__ methods.
    Both systems timebases must match. Returned systems max_delta is
    the smaller of the two connected systems. Returned system will be
    normalized to a step of height 1. If subtraction instead of addition is
    wanted, use neg=True. Warnings for the returned system are turned off.

    Examples: 
        ``sys3 = parallel(sys1, sys2)  # same as sys3 = sys1 + sys2``

        ``sys3 = parallel(sys1, sys2, True)  # same as sys3 = sys1 - sys2``

    Args:
        sys1: First system.
        sys2: Second system.
        neg (Optional): When True, returns sys1 - sys2 instead of sys1 + sys2.
            Defaults to False.

    Returns:
        The resulting system.

    Raises:
        TypeError: if input type is not supported.
    """
    try:
        if neg:
            return sys1 - sys2
        else:
            return sys1 + sys2
    except NotImplementedError:
        msg = ("parallel is only supported for objects of type "
               "<class 'FsrModel'>, got types %s and %s." % (
                   str(type(sys1)), str(type(sys2))))
        raise TypeError(msg)


def series(sys1: FsrModel, sys2: FsrModel) -> FsrModel:
    """Returns the serial connection of sys1 and sys2.

    This is a wrapper for FsrModels __mul__ method. Both systems timebases must
    match. Returned systems max_delta is the smaller of the two connected ones.
    Returned system will be normalized to a step of height 1. Warnings for the
    returned system are turned off.

    Examples:
        ``sys3 = series(sys1, sys2)  # same as sys3 = sys1 * sys2``

    Args:
        sys1: First system.
        sys2: Second system.

    Returns:
        The resulting system.

    Raises:
        TypeError: if input type is not supported.
    """
    try:
        return sys1 * sys2
    except NotImplementedError:
        msg = ("series is only supported for objects of type "
               "<class 'FsrModel'>, got types %s and %s." % (
                   str(type(sys1)), str(type(sys2))))
        raise TypeError(msg)


def feedback(sys1: FsrModel, sys2: FsrModel) -> FsrModel:
    """Returns the feedback connection of sys1 and sys2.

    This is a wrapper for FsrModels __truediv__ method. sys1 is in forwards
    direction and sys2 in backwards direction. Both systems timebases 
    must match. Returned systems max_delta is the smaller of the two connected
    ones. Returned system will be normalized to a step of height 1. 
    Warnings for the returned system are turned off.

    Examples:
        ``sys3 = feedback(sys1, sys2)  # same as sys1 / sys2``

    Args:
        sys1: First system.
        sys2: Second system.

    Returns:
        The resulting system.

    Raises:
        TypeError: if input type is not supported.
    """
    try:
        return sys1 / sys2
    except NotImplementedError:
        msg = ("feedback is only supported for objects of type "
               "<class 'FsrModel'>, got types %s and %s." % (
                   str(type(sys1)), str(type(sys2))))
        raise TypeError(msg)


def import_csv(filehandle: IO, max_delta: float=0.01, show_warnings: bool=True, 
               delimiter: str=',', quotechar: str='"') -> FsrModel:
    """Imports a system from a CSV file.
    
    Expects a file handle in read-mode. CSV dialect can be specified by
    delimiter and quotechar parameters. Returned model parameters max_delta and
    show_warnings can also be specified.
    
    Args:
        filehandle: Handle of the file.
        max_delta (Optional): Specifies the model parameter max_delta. Defaults
            to 0.01.
        show_warnings (Optional): Specifies the model parameter show_warnings.
            Defaults to True.
        delimiter (Optional): Delimiter string for CSV dialect.
            Defaults to ','.
        quotechar (Optional): Quote character string for CSV dialect.
            Defaults to '"'.
            
    Returns:
        Model found in file.
        
    Raises:
        IOError: if file can not be properly read.
    """

    reader = csv.reader(filehandle, delimiter=delimiter, quotechar=quotechar, lineterminator='\n')
    t = []
    u = []
    y = []
    try:
        for row in reader:
            t.append(float(row[0]))
            u.append(float(row[1]))
            y.append(float(row[2]))
    except:
        msg = "Something went wrong while reading the file."
        raise IOError(msg)
    return FsrModel(np.array(y), t=np.array(t), max_delta=max_delta, 
                              show_warnings=show_warnings)


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
        TypeError: model parameter is of wrong type.
    """
    if isinstance(model, FsrModel):
        writer = csv.writer(filehandle, delimiter=delimiter,
                            quotechar=quotechar, lineterminator='\n')
        info = model.get_model_info()
        t = info['t']
        y = info['y']
        u = info['u']
        for i in range(len(t)):
            writer.writerow([t[i], u[i], y[i]])
    else:
        msg = "Expected type <class 'FsrModel'>, got type %s." % (
            str(type(model)))
        raise TypeError(msg)

# TODO: forced_response and step_response as module methods
