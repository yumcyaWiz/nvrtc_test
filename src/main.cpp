#include <cuda.h>
#include <format>
#include <iostream>
#include <memory>
#include <nvrtc.h>
#include <source_location>

const char *kernel_code = R"(
extern "C" __global__ void addKernel(const float *a, const float *b, float *c,
                                    int N)
{
    int i = threadIdx.x;
    if (i >= N) return;

    c[i] = a[i] + b[i];
}
)";

void cudaCheckError(
    const CUresult &result,
    const std::source_location &loc = std::source_location::current()) {
  if (result == CUDA_SUCCESS)
    return;

  const char *errorName = nullptr;
  cuGetErrorName(result, &errorName);
  const char *errorString = nullptr;
  cuGetErrorString(result, &errorString);

  throw std::runtime_error(
      std::format("{}({}:{}) {}: {}\n", loc.file_name(), loc.line(),
                  loc.column(), loc.function_name(), errorName, errorString));
}

void nvrtcCheckError(
    const nvrtcResult &result,
    const std::source_location &loc = std::source_location::current()) {
  if (result == NVRTC_SUCCESS)
    return;

  throw std::runtime_error(std::format("{}({}:{}) {}\n", loc.file_name(),
                                       loc.line(), loc.column(),
                                       nvrtcGetErrorString(result)));
}

template <typename T> class CUDABuffer {
private:
  CUdeviceptr dptr = 0;
  int size = 0;

public:
  CUDABuffer(int size) : size(size) {
    cudaCheckError(cuMemAlloc(&dptr, sizeof(T) * size));
  }

  CUDABuffer(const CUDABuffer &) = delete;

  CUDABuffer(CUDABuffer &&other) : dptr(other.dptr), size(other.size) {
    other.dptr = 0;
  }

  ~CUDABuffer() { cudaCheckError(cuMemFree(dptr)); }

  const CUdeviceptr &getDevicePtr() const { return dptr; }

  void copyHtoD(const T *hptr) const {
    cudaCheckError(cuMemcpyHtoD(dptr, hptr, sizeof(T) * size));
  }

  void copyDtoH(T *hptr) const {
    cudaCheckError(cuMemcpyDtoH(hptr, dptr, sizeof(T) * size));
  }
};

class CUDADevice {
private:
  CUdevice device = 0;
  CUcontext context = nullptr;

public:
  CUDADevice(CUdevice device) : device(device) {
    // check device availability
    int nDevices = 0;
    cudaCheckError(cuDeviceGetCount(&nDevices));
    if (device >= nDevices) {
      throw std::runtime_error(
          std::format("device {} is not available\n", device));
    }

    cudaCheckError(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
    cuCtxPushCurrent(context);
  }

  CUDADevice(const CUDADevice &) = delete;

  CUDADevice(CUDADevice &&other)
      : device(other.device), context(other.context) {
    other.device = 0;
    other.context = nullptr;
  }

  ~CUDADevice() {
    cuCtxPopCurrent(&context);
    cuCtxDestroy(context);
  }

  void synchronize() const { cudaCheckError(cuCtxSynchronize()); }
};

class CUDAKernel {
private:
  CUmodule module = nullptr;
  CUfunction function = nullptr;

public:
  CUDAKernel(const std::string &filename, const std::string &kernelName) {
    cudaCheckError(cuModuleLoad(&module, filename.c_str()));
    cudaCheckError(cuModuleGetFunction(&function, module, kernelName.c_str()));
  }

  CUDAKernel(const char *ptx, const std::string &kernelName) {
    cudaCheckError(cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr));
    cudaCheckError(cuModuleGetFunction(&function, module, kernelName.c_str()));
  }

  CUDAKernel(const CUDAKernel &) = delete;

  CUDAKernel(CUDAKernel &&other)
      : module(other.module), function(other.function) {
    other.module = nullptr;
    other.function = nullptr;
  }

  ~CUDAKernel() { cudaCheckError(cuModuleUnload(module)); }

  void launch(const int gridX, const int gridY, const int gridZ,
              const int blockX, const int blockY, const int blockZ,
              const void *args[]) const {
    cudaCheckError(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY,
                                  blockZ, 0, nullptr, const_cast<void **>(args),
                                  nullptr));
  }
};

class CUDAKernelRTC {
public:
  CUDAKernelRTC(const std::string &kernel_code, const std::string &kernelName) {
    nvrtcCheckError(nvrtcCreateProgram(&program, kernel_code.c_str(), nullptr,
                                       0, nullptr, nullptr));
    nvrtcCheckError(nvrtcCompileProgram(program, 0, nullptr));

    size_t log_size;
    nvrtcCheckError(nvrtcGetProgramLogSize(program, &log_size));
    char log[log_size];
    nvrtcCheckError(nvrtcGetProgramLog(program, log));
    std::cout << std::format("NVRTC: {}\n", std::string(log)) << std::endl;

    size_t ptx_size;
    nvrtcCheckError(nvrtcGetPTXSize(program, &ptx_size));
    ptx = new char[ptx_size];
    nvrtcCheckError(nvrtcGetPTX(program, ptx));
    nvrtcCheckError(nvrtcDestroyProgram(&program));

    kernel = std::make_unique<CUDAKernel>(ptx, kernelName);
  }

  ~CUDAKernelRTC() {
    delete[] ptx;
    if (kernel) {
      kernel.reset();
    }
  }

  void launch(const int gridX, const int gridY, const int gridZ,
              const int blockX, const int blockY, const int blockZ,
              const void *args[]) const {
    kernel->launch(gridX, gridY, gridZ, blockX, blockY, blockZ, args);
  }

private:
  nvrtcProgram program = nullptr;
  char *ptx = nullptr;
  std::unique_ptr<CUDAKernel> kernel = nullptr;
};

int main() {
  // init CUDA
  cudaCheckError(cuInit(0));

  // init CUDA context
  CUDADevice device(0);

  CUDAKernelRTC kernel(std::string(kernel_code), "addKernel");

  constexpr int N = 10;

  float a[N], b[N], c[N];
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i;
  }

  CUDABuffer<float> ad(N);
  ad.copyHtoD(a);
  CUDABuffer<float> bd(N);
  bd.copyHtoD(b);
  CUDABuffer<float> cd(N);

  const void *args[] = {&ad.getDevicePtr(), &bd.getDevicePtr(),
                        &cd.getDevicePtr(), &N};
  kernel.launch(1, 1, 1, N, 1, 1, args);

  cd.copyDtoH(c);
  for (int i = 0; i < N; ++i) {
    std::cout << c[i] << std::endl;
  }

  return 0;
}