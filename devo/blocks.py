import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter

try:
    import cuda_softagg
except ImportError:
    cuda_softagg = None

class LayerNorm1D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)

    def forward(self, x):
        return self.norm(x.transpose(1,2)).transpose(1,2)

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid())

        self.res = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)

class SoftAgg(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAgg, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim, self.dim)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, ix, jx=None, num_groups=None, order=None, offsets=None):
        if jx is None:
            _, jx = torch.unique(ix, return_inverse=True)

        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]
            
        return self.h(y)

class FusedSoftAgg(nn.Module):
    def __init__(self, agg, segmented=True):
        super(FusedSoftAgg, self).__init__()
        self.dim = agg.dim
        self.expand = agg.expand
        self.f = agg.f
        self.g = agg.g
        self.h = agg.h
        self.enabled = cuda_softagg is not None
        self.segmented = segmented and cuda_softagg is not None and hasattr(cuda_softagg, "forward_sorted")
        self.ordered = segmented and cuda_softagg is not None and hasattr(cuda_softagg, "forward_ordered")

    def forward(self, x, ix, jx=None, num_groups=None, order=None, offsets=None):
        if not self.enabled or x.shape[0] != 1 or not x.is_cuda:
            return self.forward_reference(x, ix, jx)

        if jx is None:
            _, jx = torch.unique(ix, return_inverse=True)
            num_groups = None

        if num_groups is None:
            # Fallback for non-topology calls. This path is rare; topology calls
            # pass an exact group count and avoid empty-group work.
            num_groups = x.shape[1]

        values = self.f(x)[0].contiguous()
        logits = self.g(x)[0].contiguous()

        if self.ordered and order is not None and offsets is not None:
            y = cuda_softagg.forward_ordered(
                values,
                logits,
                order.contiguous(),
                offsets.contiguous()).unsqueeze(0)
        elif self.segmented:
            if order is None or offsets is None:
                order = torch.argsort(jx)
                counts = torch.bincount(jx, minlength=num_groups)
                offsets = torch.cat([
                    torch.zeros(1, dtype=torch.long, device=x.device),
                    torch.cumsum(counts, dim=0)
                ])
            y = cuda_softagg.forward_sorted(
                values[order].contiguous(),
                logits[order].contiguous(),
                offsets).unsqueeze(0)
        else:
            y = cuda_softagg.forward(
                values,
                logits,
                jx.contiguous(),
                num_groups).unsqueeze(0)

        if self.expand:
            return self.h(y)[:,jx]

        return self.h(y)

    def forward_reference(self, x, ix, jx=None):
        if jx is None:
            _, jx = torch.unique(ix, return_inverse=True)
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)
        if self.expand:
            return self.h(y)[:,jx]
        return self.h(y)

    def validate_kernel(self, atol=1e-3, rtol=1e-3):
        if cuda_softagg is None:
            self.enabled = False
            return False

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        edges = 257
        num_groups = 73

        x = torch.randn(1, edges, self.dim, device=device, dtype=dtype)
        jx = torch.arange(edges, device=device, dtype=torch.long) % num_groups
        perm = torch.randperm(edges, device=device)
        jx = jx[perm].contiguous()
        ix = jx

        with torch.no_grad():
            ref = self.forward_reference(x, ix, jx).float()
            fused = self.forward(x, ix, jx, num_groups).float()
            ok = torch.allclose(ref, fused, atol=atol, rtol=rtol)

        self.enabled = bool(ok)
        return self.enabled

class SoftAggBasic(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAggBasic, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim,        1)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, ix, jx=None, num_groups=None, order=None, offsets=None):
        if jx is None:
            _, jx = torch.unique(ix, return_inverse=True)

        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]
            
        return self.h(y)


### Gradient Clipping and Zeroing Operations ###

GRAD_CLIP = 0.1

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)

class GradZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        grad_x = torch.where(torch.abs(grad_x) > GRAD_CLIP, torch.zeros_like(grad_x), grad_x)
        return grad_x

class GradientZero(nn.Module):
    def __init__(self):
        super(GradientZero, self).__init__()

    def forward(self, x):
        return GradZero.apply(x)


class GradMag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        print(grad_x.abs().mean())
        return grad_x
