#include <torch/extension.h>

torch::Tensor softagg_forward_cuda(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor groups,
    int64_t num_groups);

torch::Tensor softagg_forward(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor groups,
    int64_t num_groups) {
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  TORCH_CHECK(groups.is_cuda(), "groups must be CUDA");
  TORCH_CHECK(values.dim() == 2, "values must have shape [E, C]");
  TORCH_CHECK(logits.sizes() == values.sizes(), "logits must match values");
  TORCH_CHECK(groups.dim() == 1, "groups must have shape [E]");
  TORCH_CHECK(groups.size(0) == values.size(0), "groups length must match E");
  TORCH_CHECK(groups.scalar_type() == torch::kLong, "groups must be int64");
  TORCH_CHECK(num_groups > 0, "num_groups must be positive");

  return softagg_forward_cuda(
      values.contiguous(), logits.contiguous(), groups.contiguous(), num_groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &softagg_forward, "Fused SoftAgg forward (CUDA)");
}
