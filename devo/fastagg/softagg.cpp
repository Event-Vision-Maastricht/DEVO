#include <torch/extension.h>

torch::Tensor softagg_forward_cuda(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor groups,
    int64_t num_groups);

torch::Tensor softagg_forward_sorted_cuda(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor offsets);

torch::Tensor softagg_forward_ordered_cuda(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor order,
    torch::Tensor offsets);

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

torch::Tensor softagg_forward_sorted(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor offsets) {
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
  TORCH_CHECK(values.dim() == 2, "values must have shape [E, C]");
  TORCH_CHECK(logits.sizes() == values.sizes(), "logits must match values");
  TORCH_CHECK(offsets.dim() == 1, "offsets must have shape [G + 1]");
  TORCH_CHECK(offsets.scalar_type() == torch::kLong, "offsets must be int64");
  TORCH_CHECK(offsets.size(0) > 1, "offsets must contain at least one group");

  return softagg_forward_sorted_cuda(
      values.contiguous(), logits.contiguous(), offsets.contiguous());
}

torch::Tensor softagg_forward_ordered(
    torch::Tensor values,
    torch::Tensor logits,
    torch::Tensor order,
    torch::Tensor offsets) {
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  TORCH_CHECK(order.is_cuda(), "order must be CUDA");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
  TORCH_CHECK(values.dim() == 2, "values must have shape [E, C]");
  TORCH_CHECK(logits.sizes() == values.sizes(), "logits must match values");
  TORCH_CHECK(order.dim() == 1, "order must have shape [E]");
  TORCH_CHECK(order.size(0) == values.size(0), "order length must match E");
  TORCH_CHECK(order.scalar_type() == torch::kLong, "order must be int64");
  TORCH_CHECK(offsets.dim() == 1, "offsets must have shape [G + 1]");
  TORCH_CHECK(offsets.scalar_type() == torch::kLong, "offsets must be int64");
  TORCH_CHECK(offsets.size(0) > 1, "offsets must contain at least one group");

  return softagg_forward_ordered_cuda(
      values.contiguous(), logits.contiguous(), order.contiguous(), offsets.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &softagg_forward, "Fused SoftAgg forward (CUDA)");
  m.def("forward_sorted", &softagg_forward_sorted, "Segmented SoftAgg forward (CUDA)");
  m.def("forward_ordered", &softagg_forward_ordered, "Ordered segmented SoftAgg forward (CUDA)");
}
