#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

namespace {

__device__ inline float atomic_max_float(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i;
  int assumed;

  do {
    assumed = old;
    float old_val = __int_as_float(assumed);
    if (old_val >= val) {
      break;
    }
    old = atomicCAS(address_as_i, assumed, __float_as_int(val));
  } while (assumed != old);

  return __int_as_float(old);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <typename scalar_t>
__global__ void softagg_max_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ groups,
    float* __restrict__ maxes,
    int64_t edges,
    int64_t channels,
    int64_t num_groups) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = edges * channels;
  if (idx >= total) {
    return;
  }

  int64_t e = idx / channels;
  int64_t c = idx - e * channels;
  int64_t g = groups[e];
  if (g >= 0 && g < num_groups) {
    atomic_max_float(&maxes[g * channels + c], scalar_to_float(logits[idx]));
  }
}

template <typename scalar_t>
__global__ void softagg_sum_kernel(
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ groups,
    const float* __restrict__ maxes,
    float* __restrict__ denom,
    float* __restrict__ accum,
    int64_t edges,
    int64_t channels,
    int64_t num_groups) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = edges * channels;
  if (idx >= total) {
    return;
  }

  int64_t e = idx / channels;
  int64_t c = idx - e * channels;
  int64_t g = groups[e];
  if (g < 0 || g >= num_groups) {
    return;
  }

  int64_t out_idx = g * channels + c;
  float m = maxes[out_idx];
  float w = expf(scalar_to_float(logits[idx]) - m);
  atomicAdd(&denom[out_idx], w);
  atomicAdd(&accum[out_idx], scalar_to_float(values[idx]) * w);
}

__global__ void softagg_out_kernel(
    const float* __restrict__ denom,
    const float* __restrict__ accum,
    float* __restrict__ out,
    int64_t total) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  float d = denom[idx];
  float y = d > 0.0f ? accum[idx] / d : 0.0f;
  out[idx] = y;
}

template <typename scalar_t>
__global__ void softagg_sorted_kernel(
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ offsets,
    float* __restrict__ out,
    int64_t groups,
    int64_t channels) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = groups * channels;
  if (idx >= total) {
    return;
  }

  int64_t g = idx / channels;
  int64_t c = idx - g * channels;
  int64_t start = offsets[g];
  int64_t end = offsets[g + 1];

  if (start >= end) {
    out[idx] = 0.0f;
    return;
  }

  float max_logit = -INFINITY;
  for (int64_t e = start; e < end; ++e) {
    float v = scalar_to_float(logits[e * channels + c]);
    max_logit = v > max_logit ? v : max_logit;
  }

  float denom = 0.0f;
  float accum = 0.0f;
  for (int64_t e = start; e < end; ++e) {
    float w = expf(scalar_to_float(logits[e * channels + c]) - max_logit);
    denom += w;
    accum += scalar_to_float(values[e * channels + c]) * w;
  }

  out[idx] = denom > 0.0f ? accum / denom : 0.0f;
}

template <typename scalar_t>
__global__ void softagg_ordered_kernel(
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ offsets,
    float* __restrict__ out,
    int64_t groups,
    int64_t channels) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = groups * channels;
  if (idx >= total) {
    return;
  }

  int64_t g = idx / channels;
  int64_t c = idx - g * channels;
  int64_t start = offsets[g];
  int64_t end = offsets[g + 1];

  if (start >= end) {
    out[idx] = 0.0f;
    return;
  }

  float max_logit = -INFINITY;
  for (int64_t p = start; p < end; ++p) {
    int64_t e = order[p];
    float v = scalar_to_float(logits[e * channels + c]);
    max_logit = v > max_logit ? v : max_logit;
  }

  float denom = 0.0f;
  float accum = 0.0f;
  for (int64_t p = start; p < end; ++p) {
    int64_t e = order[p];
    float w = expf(scalar_to_float(logits[e * channels + c]) - max_logit);
    denom += w;
    accum += scalar_to_float(values[e * channels + c]) * w;
  }

  out[idx] = denom > 0.0f ? accum / denom : 0.0f;
}

}  // namespace

torch::Tensor softagg_forward_cuda(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor groups,
    int64_t num_groups) {
  const auto edges = values.size(0);
  const auto channels = values.size(1);
  auto float_opts = values.options().dtype(torch::kFloat32);
  auto out = torch::empty({num_groups, channels}, float_opts);
  auto maxes = torch::full(
      {num_groups, channels}, -std::numeric_limits<float>::infinity(), float_opts);
  auto denom = torch::zeros({num_groups, channels}, float_opts);
  auto accum = torch::zeros({num_groups, channels}, float_opts);

  const int threads = 256;
  const int64_t edge_channel_total = edges * channels;
  const int blocks_edges = static_cast<int>((edge_channel_total + threads - 1) / threads);
  const int blocks_out = static_cast<int>((num_groups * channels + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "softagg_forward_cuda", [&] {
    softagg_max_kernel<scalar_t><<<blocks_edges, threads, 0, stream>>>(
        logits.data_ptr<scalar_t>(),
        groups.data_ptr<int64_t>(),
        maxes.data_ptr<float>(),
        edges,
        channels,
        num_groups);

    softagg_sum_kernel<scalar_t><<<blocks_edges, threads, 0, stream>>>(
        values.data_ptr<scalar_t>(),
        logits.data_ptr<scalar_t>(),
        groups.data_ptr<int64_t>(),
        maxes.data_ptr<float>(),
        denom.data_ptr<float>(),
        accum.data_ptr<float>(),
        edges,
        channels,
        num_groups);

  });

  softagg_out_kernel<<<blocks_out, threads, 0, stream>>>(
      denom.data_ptr<float>(),
      accum.data_ptr<float>(),
      out.data_ptr<float>(),
      num_groups * channels);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor softagg_forward_sorted_cuda(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor offsets) {
  const auto edges = values.size(0);
  const auto channels = values.size(1);
  const auto groups = offsets.size(0) - 1;
  auto float_opts = values.options().dtype(torch::kFloat32);
  auto out = torch::empty({groups, channels}, float_opts);

  const int threads = 256;
  const int64_t total = groups * channels;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "softagg_forward_sorted_cuda", [&] {
    softagg_sorted_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        values.data_ptr<scalar_t>(),
        logits.data_ptr<scalar_t>(),
        offsets.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        groups,
        channels);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor softagg_forward_ordered_cuda(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor order,
    torch::Tensor offsets) {
  const auto channels = values.size(1);
  const auto groups = offsets.size(0) - 1;
  auto float_opts = values.options().dtype(torch::kFloat32);
  auto out = torch::empty({groups, channels}, float_opts);

  const int threads = 256;
  const int64_t total = groups * channels;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "softagg_forward_ordered_cuda", [&] {
    softagg_ordered_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        values.data_ptr<scalar_t>(),
        logits.data_ptr<scalar_t>(),
        order.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        groups,
        channels);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
