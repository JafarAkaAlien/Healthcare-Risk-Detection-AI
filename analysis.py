import time
import cudapart
import cupy as cp
import main
import preprocessing
import plotting

# Timing scikit-learn (CPU)
start = time.time()
main.model.fit(main.X_train, main.y_train)
time_cpu = time.time() - start

# Timing CUDA kernel (GPU)
startgpu = time.time()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for _ in range(100):
    cudapart.gradient_descent_kernel((cudapart.blocks_per_grid,), (cudapart.threads_per_block,), (cudapart.X, cudapart.y, cudapart.weights, cudapart.gradient, cudapart.X.shape[0], cudapart.X.shape[1], cudapart.learning_rate))
end.record()
end.synchronize()
time_gpu = time.time()-startgpu

print(f'CPU Time: {time_cpu} seconds')
print(f'GPU Time: {time_gpu} seconds')




import matplotlib.pyplot as plt

# Visualization of computation times
labels = ['CPU', 'GPU']
times = [time_cpu, time_gpu ]  # Convert GPU time to seconds
plt.bar(labels, times, color=['blue', 'green'])
plt.xlabel('Processing Unit')
plt.ylabel('Time (seconds)')
plt.title('CPU vs GPU Computation Time')
plt.show()

# Visualization of model predictions vs actual values
#plt.scatter
plotting.plot_decision_regions(main.X_train, main.y_train, main.model)
plt.xlabel('Age')
plt.ylabel('Blood Pressure Ratio')
plt.legend(loc='upper left')
plt.show()