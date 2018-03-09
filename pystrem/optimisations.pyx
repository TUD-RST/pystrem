import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def forced_response(np.ndarray[DTYPE_t] sys_y, np.ndarray[DTYPE_t] sys_t,
                           np.ndarray[DTYPE_t] t, np.ndarray[DTYPE_t] u):

    cdef int i = 0
    cdef DTYPE_t val
    cdef np.ndarray[DTYPE_t] c_buf = np.zeros(len(sys_t), dtype=DTYPE)
    cdef int t_len = t.shape[0]
    cdef DTYPE_t out_of_buf_value = 0
    cdef DTYPE_t step_out = 0
    cdef int buf_pos = 0
    cdef DTYPE_t du_rel
    cdef int idx_ptr = 0
    cdef np.ndarray[DTYPE_t] y = np.zeros(len(t), DTYPE)
    cdef int j = 0
    i = 0
    for i in range(t_len):
        if i != 0:
            du_rel = (u[i] - u[i - 1])
        else:
            du_rel = u[i]
        # Whenever we add a value to our buffer we need to remember the
        # ones thrown out of it.
        if c_buf[idx_ptr] != 0:
            out_of_buf_value += c_buf[idx_ptr] * sys_y[-1]
        c_buf[idx_ptr] = du_rel
        step_out = 0
        buf_pos = 0
        for j in range(len(c_buf)):
            step_out += c_buf[(idx_ptr+j)%len(c_buf)] * sys_y[j]
        y[i] = step_out + out_of_buf_value
        idx_ptr -= 1
        idx_ptr %= len(c_buf)
    return t, y