int integrate_euler_cuda(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                         float *vy_out, int n, int steps, float dt,
			 int n_intermediate_steps);
