import numpy as np
cimport numpy as np


cdef extern from "./CUDAOrbits.h":
    int integrate_euler_cuda(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                             float *vy_out, int n, int steps, float dt, int n_intermediate_steps) nogil


class CUDAOrbits:
    def __init__(self, x, y, vx, vy):
        """
        Create an array of Orbit

        :param x: x-locations in AU
        :type x: np.ndarray
        :param y: y-locations in AU
        :type y: np.ndarray
        :param vx: x-velocity in AU/yr
        :type vx: np.ndarray
        :param vy: y-velocity in AU/yr
        :type vy: np.ndarray
        """
        # make sure CUDA received float32 ndarray
        self.x = np.array(x).astype(np.float32)
        self.y = np.array(y).astype(np.float32)
        self.vx = np.array(vx).astype(np.float32)
        self.vy = np.array(vy).astype(np.float32)

        # number of object in this orbits
        self.num_of_obj = self.x.shape[0]

        # assert to make sure, CUDA is lame and crash system otherwise
        assert self.x.shape[0] == self.y.shape[0] == self.vx.shape[0] == self.vy.shape[0]
        assert len(self.x.shape) == len(self.y.shape) == len(self.vx.shape) == len(self.vy.shape)

        self.mode = 'CUDA'

    def integrate(self, steps=1000, dt=0.1,n_intermediate_steps=1):
        """
        Orbit Integration

        :param steps: time steps to integrate
        :type steps: int
        :param dt: delta t between steps
        :type dt: float
        :type n_intermediate_steps: int
        """

        # declare type
        cdef np.ndarray[np.float32_t, ndim=1] x_c = self.x
        cdef np.ndarray[np.float32_t, ndim=1] y_c = self.y
        cdef np.ndarray[np.float32_t, ndim=1] vx_c = self.vx
        cdef np.ndarray[np.float32_t, ndim=1] vy_c = self.vy
        cdef np.ndarray[np.float32_t, ndim=1] x_c_out = np.tile(x_c, steps)
        cdef np.ndarray[np.float32_t, ndim=1] y_c_out = np.tile(y_c, steps)
        cdef np.ndarray[np.float32_t, ndim=1] vx_c_out = np.tile(vx_c, steps)
        cdef np.ndarray[np.float32_t, ndim=1] vy_c_out = np.tile(vy_c, steps)

        # integrate on CUDA GPU
        integrate_euler_cuda(&x_c[0], &y_c[0], &vx_c[0], &vy_c[0], &x_c_out[0], &y_c_out[0], &vx_c_out[0], &vy_c_out[0],
                             self.num_of_obj, steps, dt,
			     n_intermediate_steps)

        # CUDA should send back the whole array back to corresponding CPU memory address, we just need to put it in
        # the right python variable
        self.x = x_c_out.reshape(steps, self.num_of_obj).T
        self.y = y_c_out.reshape(steps, self.num_of_obj).T
        self.vx = vx_c_out.reshape(steps, self.num_of_obj).T
        self.vy = vy_c_out.reshape(steps, self.num_of_obj).T

        return None

    @property
    def R(self):
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def E(self):
        return 0.5*(self.vx**2.+self.vy**2.)+np.log(self.R)
#        return 0.5*(self.vx**2.+self.vy**2.)-39.5/np.sqrt(self.x**2 + self.y**2)