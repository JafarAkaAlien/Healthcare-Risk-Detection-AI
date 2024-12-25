import cupy as cp

# Custom CUDA kernel for gradient descent step in linear regression
gradient_descent_kernel = cp.RawKernel(r'''
extern "C" __global__ void gradient_descent(const float* X, const float* y, float* weights, float* gradient, int N, int M, float learning_rate) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < M) {
        float grad = 0.0;
        for (int i = 0; i < N; i++) {
            float prediction = 0.0;
            for (int j = 0; j < M; j++) {
                prediction += X[i * M + j] * weights[j];
            }
            grad += (prediction - y[i]) * X[i * M + idx];
        }
        gradient[idx] = grad / N;
        weights[idx] -= learning_rate * gradient[idx];
    }
}
''', 'gradient_descent')

# Example usage
X = cp.random.rand(1000, 2, dtype=cp.float32)
y = cp.random.rand(1000, dtype=cp.float32)
weights = cp.zeros(2, dtype=cp.float32)
gradient = cp.zeros(2, dtype=cp.float32)

# Configure kernel launch parameters
threads_per_block = 256
blocks_per_grid = (X.shape[1] + threads_per_block - 1) // threads_per_block

# Launch kernel (simulate multiple iterations)
learning_rate = 0.01
for _ in range(100):
    gradient_descent_kernel((blocks_per_grid,), (threads_per_block,), (X, y, weights, gradient, X.shape[0], X.shape[1], learning_rate))