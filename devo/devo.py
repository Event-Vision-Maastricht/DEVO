import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

# from .net import VONet # TODO add net.py
from .enet import eVONet
from .utils import *
from . import projective_ops as pops
from .blocks import SoftAggBasic

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")

from utils.viz_utils import visualize_voxel


class DEVO:
    def __init__(self, cfg, network, evs=False, ht=480, wd=640, viz=False, viz_flow=False, dim_inet=384, dim_fnet=128, dim=32):
        self.cfg = cfg
        self.evs = evs

        self.dim_inet = dim_inet
        self.dim_fnet = dim_fnet
        self.dim = dim
        # TODO add patch_selector
        
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False # TODO timing in param

        self.viz_flow = viz_flow
        
        self.n = 0      # active keyframes/frames (every frames == keyframe)
        self.m = 0      # number active patches
        self.M = self.cfg.PATCHES_PER_FRAME     # (default: 96)
        self.N = self.cfg.BUFFER_SIZE           # max number of keyframes (default: 2048)

        self.ht = ht    # image height
        self.wd = wd    # image width

        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0 # how often this network is called __call__()

        self.flow_data = {}

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda") # 3 channels = (x, y, depth)
        self.patches_gt_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, self.dim_inet, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, self.dim_fnet, self.P, self.P, **kwargs)

        ht = int(ht // RES)
        wd = int(wd // RES)

        self.fmap1_ = torch.zeros(1, self.mem, self.dim_fnet, int(ht // 1), int(wd // 1), **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, self.dim_fnet, int(ht // 4), int(wd // 4), **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, self.dim_inet, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.active_target = torch.zeros(1, 0, 2, dtype=torch.float, device="cuda")
        self.active_weight = torch.zeros(1, 0, 2, dtype=torch.float, device="cuda")
        self.active_delta = torch.zeros(1, 0, 2, dtype=torch.float, device="cuda")
        self.active_confidence = torch.as_tensor([], dtype=torch.float, device="cuda")
        self.active_delta_norm = torch.as_tensor([], dtype=torch.float, device="cuda")
        self.active_keepalive = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.marg_ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.marg_jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.marg_kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.marg_target = torch.zeros(1, 0, 2, dtype=torch.float, device="cuda")
        self.marg_delta = torch.zeros(1, 0, 2, dtype=torch.float, device="cuda")
        self.marg_weight = torch.zeros(1, 0, 2, dtype=torch.float, device="cuda")
        self.marg_weight_scale = torch.zeros(1, 0, 1, dtype=torch.float, device="cuda")
        self.marginalize_update_count = 0
        self.marginalize_freeze_cooldown = 0
        self.active_edge_budget = getattr(self.cfg, "MARGINALIZE_MAX_ACTIVE_EDGES", 0)
        self.neural_edge_budget = getattr(self.cfg, "WARM_BASE_NEURAL_EDGES", 0)
        self.graph_version = 0
        self.update_topology_cache = {}
        self.active_order_version = -1
        self.active_edges_ordered = False
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.viewer = None
        if viz:
            self.start_viewer()

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            print(f"Loading from {network}")
            checkpoint = torch.load(network)
            # TODO infer dim_inet=self.dim_inet, dim_fnet=self.dim_fnet, dim=self.dim
            self.network = VONet(patch_selector=self.cfg.PATCH_SELECTOR) if not self.evs else \
                eVONet(dim_inet=self.dim_inet, dim_fnet=self.dim_fnet, dim=self.dim, patch_selector=self.cfg.PATCH_SELECTOR)
            if 'model_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['model_state_dict'])
            else:
                # legacy
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if "update.lmbda" not in k:
                        new_state_dict[k.replace('module.', '')] = v
                self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.dim_inet = self.network.dim_inet
        self.dim_fnet = self.network.dim_fnet
        self.dim = self.network.dim
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()
        self.configure_update_aggregation()
        self.network.requires_grad_(False)
        self.update_op = self.network.update
        self.update_op_compiled = False

        if getattr(self.cfg, "COMPILE_UPDATE_NET", False) and hasattr(torch, "compile"):
            try:
                if hasattr(torch, "_dynamo"):
                    torch._dynamo.config.suppress_errors = True
                self.update_op = torch.compile(
                    self.network.update,
                    mode=getattr(self.cfg, "COMPILE_UPDATE_MODE", "reduce-overhead"),
                    dynamic=getattr(self.cfg, "COMPILE_UPDATE_DYNAMIC", True))
                self.update_op_compiled = True
            except Exception as e:
                print(f"Warning: update_net compile disabled ({e})")

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()

    def configure_update_aggregation(self):
        if not getattr(self.cfg, "SCALAR_SOFTAGG", False):
            return

        self.network.update.agg_kk = self.scalarize_softagg(self.network.update.agg_kk)
        self.network.update.agg_ij = self.scalarize_softagg(self.network.update.agg_ij)

    def scalarize_softagg(self, agg):
        scalar = SoftAggBasic(agg.dim, expand=agg.expand).to(next(agg.parameters()).device)
        scalar.f.load_state_dict(agg.f.state_dict())
        scalar.h.load_state_dict(agg.h.state_dict())

        with torch.no_grad():
            scalar.g.weight.copy_(agg.g.weight.mean(dim=0, keepdim=True))
            scalar.g.bias.copy_(agg.g.bias.mean().view(1))

        return scalar

    def bump_graph_version(self):
        self.graph_version += 1
        self.update_topology_cache = {}
        self.active_edges_ordered = False

    def update_topology(self, ii, jj, kk, cacheable=False):
        if not getattr(self.cfg, "UPDATE_TOPOLOGY_CACHE", False):
            return None

        key = None
        if cacheable:
            key = (self.graph_version, len(ii), ii.data_ptr(), jj.data_ptr(), kk.data_ptr())
            cached = self.update_topology_cache.get(key)
            if cached is not None:
                return cached

        ordered = cacheable and self.active_edges_ordered and \
            ii.data_ptr() == self.ii.data_ptr() and \
            jj.data_ptr() == self.jj.data_ptr() and \
            kk.data_ptr() == self.kk.data_ptr()

        ix, jx = self.temporal_neighbors(kk, jj, ordered=ordered)
        kk_group = self.ordered_group_inverse(kk) if ordered else \
            torch.unique(kk, return_inverse=True)[1]
        _, ij_group = torch.unique(ii * 12345 + jj, return_inverse=True)
        topology = {
            "ix": ix,
            "jx": jx,
            "kk_group": kk_group,
            "ij_group": ij_group,
        }

        if key is not None:
            self.update_topology_cache[key] = topology

        return topology

    def temporal_neighbors(self, kk, jj, ordered=False):
        if not getattr(self.cfg, "UPDATE_GPU_TOPOLOGY", True) or len(kk) == 0:
            return fastba.neighbors(kk, jj)

        try:
            if ordered:
                ix = torch.full_like(kk, -1)
                jx = torch.full_like(kk, -1)
                if len(kk) > 1:
                    same_prev = kk[1:] == kk[:-1]
                    ids = torch.arange(len(kk), dtype=kk.dtype, device=kk.device)
                    ix[1:] = torch.where(same_prev, ids[:-1], ix[1:])
                    jx[:-1] = torch.where(same_prev, ids[1:], jx[:-1])
                return ix, jx

            key_stride = max(getattr(self.cfg, "BUFFER_SIZE", self.N), self.N) + 1
            edge_stride = len(kk) + 1
            edge_order = torch.arange(len(kk), dtype=kk.dtype, device=kk.device)
            order = torch.argsort((kk * key_stride + jj) * edge_stride + edge_order)
            kk_sorted = kk[order]

            prev_sorted = torch.full_like(order, -1)
            next_sorted = torch.full_like(order, -1)

            if len(order) > 1:
                same_prev = kk_sorted[1:] == kk_sorted[:-1]
                prev_sorted[1:] = torch.where(same_prev, order[:-1], prev_sorted[1:])
                next_sorted[:-1] = torch.where(same_prev, order[1:], next_sorted[:-1])

            ix = torch.empty_like(order)
            jx = torch.empty_like(order)
            ix[order] = prev_sorted
            jx[order] = next_sorted
            return ix, jx
        except TypeError:
            return fastba.neighbors(kk, jj)

    def ordered_group_inverse(self, keys):
        if len(keys) == 0:
            return keys

        group = torch.zeros_like(keys)
        if len(keys) > 1:
            starts = torch.ones_like(keys)
            starts[0] = 0
            starts[1:] = (keys[1:] != keys[:-1]).long()
            group = torch.cumsum(starts, dim=0)
        return group

    def run_update_net(self, net, ctx, corr, flow, ii, jj, kk, topology=None):
        try:
            return self.update_op(net, ctx, corr, flow, ii, jj, kk, topology)
        except Exception as e:
            if self.update_op_compiled and getattr(self.cfg, "COMPILE_UPDATE_FALLBACK", True):
                print(f"Warning: compiled update_net failed; falling back ({e})")
                self.update_op = self.network.update
                self.update_op_compiled = False
                return self.update_op(net, ctx, corr, flow, ii, jj, kk, topology)
            raise


    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)
    
    @property
    def patches_gt(self):
        return self.patches_gt_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.dim_inet)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, self.dim_fnet, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        print("keyframes", self.n)
        self.final_refine()
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        if self.is_initialized:
            poses = [self.get_pose(t) for t in range(self.counter)]
            poses = lietorch.stack(poses, dim=0)
            poses = poses.inv().data.cpu().numpy()
            poses = self.smooth_trajectory(poses)
        else:
            print(f"Warning: Model is not initialized. Using Identity.") # eval still runs bug
            id = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            poses = np.array([id for t in range(self.counter)])
            poses[:, :3] = poses[:, :3] + np.random.randn(self.counter, 3) * 0.01 # small random trans

        tstamps = np.array(self.tlist, dtype=np.float64)

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps

    def smooth_trajectory(self, poses):
        if not getattr(self.cfg, "TRAJECTORY_SMOOTHING", False) or len(poses) < 3:
            return poses

        alpha = getattr(self.cfg, "TRAJECTORY_SMOOTHING_ALPHA", 0.25)
        passes = max(getattr(self.cfg, "TRAJECTORY_SMOOTHING_PASSES", 1), 1)
        alpha = min(max(alpha, 0.0), 0.5)
        smoothed = poses.copy()

        for _ in range(passes):
            prev = smoothed[:-2]
            curr = smoothed[1:-1]
            nxt = smoothed[2:]

            out = smoothed.copy()
            out[1:-1, :3] = alpha * prev[:, :3] + (1.0 - 2.0 * alpha) * curr[:, :3] + alpha * nxt[:, :3]

            qprev = prev[:, 3:7]
            qcurr = curr[:, 3:7]
            qnext = nxt[:, 3:7]
            qprev = np.where((qprev * qcurr).sum(axis=1, keepdims=True) < 0, -qprev, qprev)
            qnext = np.where((qnext * qcurr).sum(axis=1, keepdims=True) < 0, -qnext, qnext)
            q = alpha * qprev + (1.0 - 2.0 * alpha) * qcurr + alpha * qnext
            q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
            out[1:-1, 3:7] = q
            smoothed = out

        return smoothed

    def final_refine(self):
        if not self.is_initialized or not getattr(self.cfg, "FINAL_BA_ENABLED", False):
            return

        ba_ii, ba_jj, ba_kk, ba_target, ba_weight = self.ba_factors()
        if len(ba_ii) == 0:
            return

        lmbda = torch.as_tensor([1e-4], device="cuda")
        window = getattr(self.cfg, "FINAL_BA_WINDOW", 0)
        t0 = 1 if window <= 0 else max(self.n - window, 1)
        rounds = max(getattr(self.cfg, "FINAL_BA_ROUNDS", 1), 1)
        iterations = max(getattr(self.cfg, "FINAL_BA_ITERATIONS", 6), 1)

        for _ in range(rounds):
            try:
                fastba.BA(self.poses, self.patches, self.intrinsics,
                    ba_target, ba_weight, lmbda, ba_ii, ba_jj, ba_kk, t0, self.n,
                    iterations)
            except:
                print("Warning final BA failed...")
                return
    
    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        if getattr(self.cfg, "CORR_SINGLE_LEVEL", False):
            corr2 = torch.zeros_like(corr1)
        else:
            corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]]) 
        self.bump_graph_version()
        # TODO: self.ix.shape = self.M*self.N
        # self.ix is filled dynamically

        net = torch.zeros(1, len(ii), self.dim_inet, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)
        self.active_target = torch.cat([
            self.active_target,
            torch.zeros(1, len(ii), 2, dtype=torch.float, device="cuda")], dim=1)
        self.active_weight = torch.cat([
            self.active_weight,
            torch.zeros(1, len(ii), 2, dtype=torch.float, device="cuda")], dim=1)
        self.active_delta = torch.cat([
            self.active_delta,
            torch.zeros(1, len(ii), 2, dtype=torch.float, device="cuda")], dim=1)
        self.active_confidence = torch.cat([
            self.active_confidence,
            torch.zeros(len(ii), dtype=torch.float, device="cuda")])
        self.active_delta_norm = torch.cat([
            self.active_delta_norm,
            torch.full((len(ii),), torch.inf, dtype=torch.float, device="cuda")])
        self.active_keepalive = torch.cat([
            self.active_keepalive,
            torch.zeros(len(ii), dtype=torch.long, device="cuda")])

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]
        self.active_target = self.active_target[:,~m]
        self.active_weight = self.active_weight[:,~m]
        self.active_delta = self.active_delta[:,~m]
        self.active_confidence = self.active_confidence[~m]
        self.active_delta_norm = self.active_delta_norm[~m]
        self.active_keepalive = self.active_keepalive[~m]
        if m.any():
            self.bump_graph_version()

    def remove_marginalized_factors(self, m):
        self.marg_ii = self.marg_ii[~m]
        self.marg_jj = self.marg_jj[~m]
        self.marg_kk = self.marg_kk[~m]
        self.marg_target = self.marg_target[:,~m]
        self.marg_delta = self.marg_delta[:,~m]
        self.marg_weight = self.marg_weight[:,~m]
        self.marg_weight_scale = self.marg_weight_scale[:,~m]

    def remove_all_factors(self, m_active, m_marg=None):
        self.remove_factors(m_active)
        if m_marg is not None:
            self.remove_marginalized_factors(m_marg)

    def order_active_factors(self):
        if not getattr(self.cfg, "EDGE_ORDERING_ENABLED", False) or len(self.ii) <= 1:
            return
        if self.active_order_version == self.graph_version:
            return

        stride = max(getattr(self.cfg, "BUFFER_SIZE", self.N), self.N) + 1
        edge_stride = len(self.kk) + 1
        original = torch.arange(len(self.kk), dtype=self.kk.dtype, device=self.kk.device)
        order = torch.argsort((self.kk * stride + self.jj) * edge_stride + original)

        if torch.equal(order, original):
            self.active_order_version = self.graph_version
            self.active_edges_ordered = True
            return

        self.ii = self.ii[order]
        self.jj = self.jj[order]
        self.kk = self.kk[order]
        self.net = self.net[:,order]
        self.active_target = self.active_target[:,order]
        self.active_weight = self.active_weight[:,order]
        self.active_delta = self.active_delta[:,order]
        self.active_confidence = self.active_confidence[order]
        self.active_delta_norm = self.active_delta_norm[order]
        self.active_keepalive = self.active_keepalive[order]
        self.bump_graph_version()
        self.active_order_version = self.graph_version
        self.active_edges_ordered = True

    def marginalization_enabled(self):
        return getattr(self.cfg, "ACTIVE_EDGE_MARGINALIZATION", False)

    def marginalize_factors(self, m, target, weight):
        if m.sum().item() == 0:
            return

        if not getattr(self.cfg, "MARGINALIZE_USE_FROZEN_IN_BA", True):
            self.remove_factors(m)
            return

        self.marg_ii = torch.cat([self.marg_ii, self.ii[m]])
        self.marg_jj = torch.cat([self.marg_jj, self.jj[m]])
        self.marg_kk = torch.cat([self.marg_kk, self.kk[m]])
        self.marg_target = torch.cat([self.marg_target, target[:,m].detach().float()], dim=1)
        self.marg_delta = torch.cat([self.marg_delta, self.active_delta[:,m].detach().float()], dim=1)
        self.marg_weight = torch.cat([self.marg_weight, weight[:,m].detach().float()], dim=1)
        scale = torch.ones(1, m.sum().item(), 1, dtype=torch.float, device="cuda")
        self.marg_weight_scale = torch.cat([self.marg_weight_scale, scale], dim=1)
        self.remove_factors(m)
        self.prune_marginalized_factors()

    def reactivate_marginalized_factors(self, m):
        if m.sum().item() == 0:
            return

        num = m.sum().item()
        self.ii = torch.cat([self.ii, self.marg_ii[m]])
        self.jj = torch.cat([self.jj, self.marg_jj[m]])
        self.kk = torch.cat([self.kk, self.marg_kk[m]])
        self.bump_graph_version()

        net = torch.zeros(1, num, self.dim_inet, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

        weight = self.marg_weight[:,m] * self.marg_weight_scale[:,m]
        self.active_target = torch.cat([self.active_target, self.marg_target[:,m]], dim=1)
        self.active_weight = torch.cat([self.active_weight, weight], dim=1)
        self.active_delta = torch.cat([self.active_delta, self.marg_delta[:,m]], dim=1)
        self.active_confidence = torch.cat([
            self.active_confidence,
            weight[0].mean(dim=-1).float()])
        self.active_delta_norm = torch.cat([
            self.active_delta_norm,
            torch.full((num,), torch.inf, dtype=torch.float, device="cuda")])

        keepalive = getattr(self.cfg, "MARGINALIZE_REACTIVATE_KEEPALIVE", 2)
        self.active_keepalive = torch.cat([
            self.active_keepalive,
            torch.full((num,), keepalive, dtype=torch.long, device="cuda")])
        self.remove_marginalized_factors(m)

    def current_active_budget(self):
        if not getattr(self.cfg, "MARGINALIZE_ADAPTIVE_BUDGET", False):
            return getattr(self.cfg, "MARGINALIZE_MAX_ACTIVE_EDGES", 0)
        return self.active_edge_budget

    def update_active_budget(self, confidence, delta_norm):
        if not getattr(self.cfg, "MARGINALIZE_ADAPTIVE_BUDGET", False) or len(confidence) == 0:
            self.active_edge_budget = getattr(self.cfg, "MARGINALIZE_MAX_ACTIVE_EDGES", 0)
            return

        interval = max(getattr(self.cfg, "MARGINALIZE_ADAPT_INTERVAL", 1), 1)
        if self.marginalize_update_count % interval != 0:
            return

        finite_delta = torch.where(torch.isfinite(delta_norm), delta_norm, torch.zeros_like(delta_norm))
        delta_thresh = getattr(self.cfg, "MARGINALIZE_ADAPT_DELTA_THRESH", 0.75)
        conf_thresh = getattr(self.cfg, "MARGINALIZE_ADAPT_CONF_THRESH", 0.45)
        motion_thresh = getattr(self.cfg, "MARGINALIZE_ADAPT_MOTION_THRESH", 0.45)
        hard_motion_thresh = getattr(self.cfg, "MARGINALIZE_ADAPT_HARD_MOTION_THRESH", 0.8)

        difficult = (finite_delta > delta_thresh) | (confidence < conf_thresh)
        hard_ratio = difficult.float().mean().item()
        motion = finite_delta.mean().item()

        hard = hard_ratio >= getattr(self.cfg, "MARGINALIZE_ADAPT_HARD_RATIO", 0.25) or \
            motion >= hard_motion_thresh
        medium = hard_ratio >= getattr(self.cfg, "MARGINALIZE_ADAPT_MEDIUM_RATIO", 0.12) or \
            motion >= motion_thresh

        if hard:
            target_active = getattr(self.cfg, "MARGINALIZE_HARD_ACTIVE_EDGES", 3800)
            target_neural = getattr(self.cfg, "WARM_HARD_NEURAL_EDGES", 2600)
            self.marginalize_freeze_cooldown = max(
                self.marginalize_freeze_cooldown,
                getattr(self.cfg, "MARGINALIZE_HARD_FREEZE_COOLDOWN", 0))
        elif medium:
            target_active = getattr(self.cfg, "MARGINALIZE_BASE_ACTIVE_EDGES", 3200)
            target_neural = getattr(self.cfg, "WARM_BASE_NEURAL_EDGES", 2100)
            self.marginalize_freeze_cooldown = max(
                self.marginalize_freeze_cooldown,
                getattr(self.cfg, "MARGINALIZE_MEDIUM_FREEZE_COOLDOWN", 0))
        else:
            target_active = getattr(self.cfg, "MARGINALIZE_MIN_ACTIVE_EDGES", 3000)
            target_neural = getattr(self.cfg, "WARM_MIN_NEURAL_EDGES", 1800)

        max_step = getattr(self.cfg, "MARGINALIZE_ADAPT_MAX_STEP", 400)
        if self.active_edge_budget <= 0 or max_step <= 0:
            self.active_edge_budget = target_active
        elif target_active > self.active_edge_budget:
            self.active_edge_budget = min(target_active, self.active_edge_budget + max_step)
        else:
            self.active_edge_budget = max(target_active, self.active_edge_budget - max_step)

        self.neural_edge_budget = target_neural

    def select_neural_factors(self):
        if not getattr(self.cfg, "WARM_UPDATE_ENABLED", False) or len(self.ii) == 0:
            return torch.ones(len(self.ii), dtype=torch.bool, device="cuda")

        budget = self.neural_edge_budget
        if budget <= 0 or len(self.ii) <= budget:
            return torch.ones(len(self.ii), dtype=torch.bool, device="cuda")

        core_window = getattr(self.cfg, "MARGINALIZE_CORE_WINDOW", 4)
        newest_core = max(self.n - core_window, 0)
        core = (self.ii >= newest_core) | (self.jj >= newest_core)
        fresh = torch.isinf(self.active_delta_norm) | (self.active_weight[0].mean(dim=-1) <= 0)
        stable_conf = getattr(self.cfg, "WARM_STABLE_CONF_THRESH", 0.6)
        stable_delta = getattr(self.cfg, "WARM_STABLE_DELTA_THRESH", 0.35)
        skip_pool = (~core) & (~fresh) & \
            (self.active_confidence >= stable_conf) & \
            (self.active_delta_norm <= stable_delta)

        num_to_skip = min(len(self.ii) - budget, skip_pool.sum().item())
        neural = torch.ones(len(self.ii), dtype=torch.bool, device="cuda")
        if num_to_skip <= 0:
            return neural

        stability = self.active_confidence - getattr(self.cfg, "WARM_DELTA_WEIGHT", 2.0) * self.active_delta_norm
        stability = stability.masked_fill(~skip_pool, -torch.inf)
        _, skip_idx = torch.topk(stability, k=num_to_skip)
        neural[skip_idx] = False
        return neural

    def select_adaptive_neural_factors(self):
        base = self.select_neural_factors()
        if not getattr(self.cfg, "ADAPTIVE_UPDATE_ENABLED", False) or len(self.ii) == 0:
            return base

        interval = max(getattr(self.cfg, "ADAPTIVE_FULL_UPDATE_INTERVAL", 2), 1)
        if interval <= 1 or self.marginalize_update_count % interval == 0:
            return base

        fresh = torch.isinf(self.active_delta_norm) | (self.active_weight[0].mean(dim=-1) <= 0)

        core_window = getattr(self.cfg, "ADAPTIVE_UPDATE_CORE_WINDOW", 1)
        newest_core = max(self.n - core_window, 0)
        core = (self.ii >= newest_core) | (self.jj >= newest_core)

        hard_delta = getattr(self.cfg, "ADAPTIVE_UPDATE_DELTA_THRESH", 0.75)
        hard_conf = getattr(self.cfg, "ADAPTIVE_UPDATE_CONF_THRESH", 0.45)
        hard = torch.isfinite(self.active_delta_norm) & \
            ((self.active_delta_norm >= hard_delta) | (self.active_confidence <= hard_conf))

        neural = base & (fresh | core | hard)
        min_edges = getattr(self.cfg, "ADAPTIVE_UPDATE_MIN_EDGES", 0)
        if min_edges <= 0 or neural.sum().item() >= min_edges:
            return neural

        num_to_add = min(min_edges - neural.sum().item(), (base & (~neural)).sum().item())
        if num_to_add <= 0:
            return neural

        score = self.active_delta_norm.float() - self.active_confidence.float()
        score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
        score = score.masked_fill(~(base & (~neural)), -torch.inf)
        _, add_idx = torch.topk(score, k=num_to_add)
        neural[add_idx] = True
        return neural

    def select_marginalized_factors(self, confidence, delta_norm, enforce_budget=False):
        if not self.marginalization_enabled() or len(self.ii) == 0:
            return torch.zeros(len(self.ii), dtype=torch.bool, device="cuda")

        if self.marginalize_freeze_cooldown > 0:
            return torch.zeros(len(self.ii), dtype=torch.bool, device="cuda")

        weight_thresh = getattr(self.cfg, "MARGINALIZE_WEIGHT_THRESH", 0.75)
        delta_thresh = getattr(self.cfg, "MARGINALIZE_DELTA_THRESH", 0.25)
        min_age = getattr(self.cfg, "MARGINALIZE_MIN_AGE", 3)
        core_window = getattr(self.cfg, "MARGINALIZE_CORE_WINDOW", 4)
        max_active_edges = self.current_active_budget()
        force_budget = getattr(self.cfg, "MARGINALIZE_FORCE_BUDGET", False)
        force_delta_thresh = getattr(self.cfg, "MARGINALIZE_FORCE_DELTA_THRESH", 10.0)
        freeze_delta_weight = getattr(self.cfg, "MARGINALIZE_FREEZE_DELTA_WEIGHT", 1.0)
        protect_delta_thresh = getattr(self.cfg, "MARGINALIZE_PROTECT_DELTA_THRESH", 0.75)
        protect_conf_thresh = getattr(self.cfg, "MARGINALIZE_PROTECT_CONF_THRESH", 0.45)
        coverage_stride = getattr(self.cfg, "MARGINALIZE_COVERAGE_STRIDE", 0)
        coverage_penalty = getattr(self.cfg, "MARGINALIZE_COVERAGE_PENALTY", 0.75)
        age_weight = getattr(self.cfg, "MARGINALIZE_AGE_WEIGHT", 0.02)
        backbone_window = getattr(self.cfg, "MARGINALIZE_BACKBONE_WINDOW", 0)
        backbone_penalty = getattr(self.cfg, "MARGINALIZE_BACKBONE_PENALTY", 0.0)

        patch_frame = self.ix[self.kk]
        newest_core = max(self.n - core_window, 0)

        old_enough = patch_frame <= self.n - min_age
        outside_core = (self.ii < newest_core) & (self.jj < newest_core)
        fresh = torch.isinf(delta_norm) | (self.active_weight[0].mean(dim=-1) <= 0)
        finite = torch.isfinite(delta_norm) & torch.isfinite(confidence)
        reusable = self.active_keepalive <= 0
        candidates = old_enough & outside_core & reusable & (~fresh) & finite
        protected = torch.zeros(len(self.ii), dtype=torch.bool, device="cuda")

        if protect_delta_thresh > 0:
            protected |= delta_norm >= protect_delta_thresh

        if protect_conf_thresh > 0:
            protected |= confidence <= protect_conf_thresh

        coverage_anchor = torch.zeros(len(self.ii), dtype=torch.bool, device="cuda")
        if coverage_stride > 1:
            coverage_anchor = (self.kk % coverage_stride) == 0

        converged = (confidence >= weight_thresh) & \
            (delta_norm <= delta_thresh) & candidates & (~protected)

        if not enforce_budget or max_active_edges <= 0 or len(self.ii) <= max_active_edges:
            return converged

        num_to_freeze = len(self.ii) - max_active_edges
        if force_budget:
            freeze_pool = candidates & (delta_norm <= force_delta_thresh)
        else:
            freeze_pool = converged & (~protected)

        num_to_freeze = min(num_to_freeze, freeze_pool.sum().item())
        if num_to_freeze <= 0:
            return torch.zeros(len(self.ii), dtype=torch.bool, device="cuda")

        age = (self.n - patch_frame).float()
        score = confidence - freeze_delta_weight * delta_norm + age_weight * age
        score = score - coverage_penalty * coverage_anchor.float()
        if backbone_window > 0 and backbone_penalty > 0:
            temporal_backbone = (self.ii - self.jj).abs() <= backbone_window
            score = score - backbone_penalty * temporal_backbone.float()
        score = score - 10.0 * protected.float()
        score = score.masked_fill(~freeze_pool, -torch.inf)
        _, freeze_idx = torch.topk(score, k=num_to_freeze)

        to_freeze = torch.zeros(len(self.ii), dtype=torch.bool, device="cuda")
        to_freeze[freeze_idx] = True
        return to_freeze

    def marginalize_cached_factors(self):
        to_marginalize = self.select_marginalized_factors(
            self.active_confidence, self.active_delta_norm, enforce_budget=True)
        self.marginalize_factors(to_marginalize, self.active_target, self.active_weight)

    def validate_marginalized_factors(self):
        if not getattr(self.cfg, "MARGINALIZE_USE_FROZEN_IN_BA", True):
            return

        if not getattr(self.cfg, "MARGINALIZE_VALIDATE_FROZEN", True) or len(self.marg_ii) == 0:
            return

        interval = max(getattr(self.cfg, "MARGINALIZE_VALIDATE_INTERVAL", 1), 1)
        if self.marginalize_update_count % interval != 0:
            return

        coords = self.reproject(indicies=(self.marg_ii, self.marg_jj, self.marg_kk))
        current = coords[...,self.P//2,self.P//2]
        target = current + self.marg_delta if getattr(self.cfg, "MARGINALIZE_REFRESH_TARGETS", False) else self.marg_target
        residual = (target - current).norm(dim=-1)[0]
        soft_residual = getattr(self.cfg, "MARGINALIZE_SOFT_FROZEN_RESIDUAL", 2.0)
        max_residual = getattr(self.cfg, "MARGINALIZE_MAX_FROZEN_RESIDUAL", 8.0)
        min_scale = getattr(self.cfg, "MARGINALIZE_MIN_FROZEN_WEIGHT_SCALE", 0.2)

        if max_residual > soft_residual:
            scale = 1.0 - (residual - soft_residual) / (max_residual - soft_residual)
            scale = scale.clamp(min=min_scale, max=1.0)
            scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
            self.marg_weight_scale = scale.view(1, -1, 1)

        stale = (~torch.isfinite(residual)) | (residual > max_residual)

        if getattr(self.cfg, "MARGINALIZE_REACTIVATE_FROZEN", False):
            reactivate_residual = getattr(self.cfg, "MARGINALIZE_REACTIVATE_RESIDUAL", 4.0)
            max_reactivate = getattr(self.cfg, "MARGINALIZE_MAX_REACTIVATE", 128)
            reactivate_pool = torch.isfinite(residual) & \
                (residual > reactivate_residual) & \
                (residual <= max_residual)

            num_reactivate = min(max_reactivate, reactivate_pool.sum().item())
            if num_reactivate > 0:
                score = residual.masked_fill(~reactivate_pool, -torch.inf)
                _, reactivate_idx = torch.topk(score, k=num_reactivate)
                reactivate = torch.zeros(len(self.marg_ii), dtype=torch.bool, device="cuda")
                reactivate[reactivate_idx] = True
                self.reactivate_marginalized_factors(reactivate)
                stale = stale[~reactivate]

        if stale.any():
            self.remove_marginalized_factors(stale)

        self.prune_marginalized_factors()

    def prune_marginalized_factors(self):
        max_frozen_edges = getattr(self.cfg, "MARGINALIZE_MAX_FROZEN_EDGES", 0)
        if max_frozen_edges <= 0 or len(self.marg_ii) <= max_frozen_edges:
            return

        interval = max(getattr(self.cfg, "MARGINALIZE_PRUNE_INTERVAL", 1), 1)
        if self.marginalize_update_count % interval != 0:
            return

        score = self.marg_weight[0].mean(dim=-1)
        _, keep_idx = torch.topk(score, k=max_frozen_edges)
        keep = torch.zeros(len(self.marg_ii), dtype=torch.bool, device="cuda")
        keep[keep_idx] = True
        self.remove_marginalized_factors(~keep)

    def print_marginalization_stats(self):
        if getattr(self.cfg, "MARGINALIZE_PRINT_STATS", False):
            print(f"edges active={len(self.ii)} frozen={len(self.marg_ii)} active_budget={self.current_active_budget()} neural_budget={self.neural_edge_budget} freeze_cooldown={self.marginalize_freeze_cooldown}")

    def ba_factors(self, target=None, weight=None):
        use_frozen = getattr(self.cfg, "MARGINALIZE_USE_FROZEN_IN_BA", True)
        frozen_weight_scale = getattr(self.cfg, "MARGINALIZE_FROZEN_BA_WEIGHT", 1.0)
        refresh_frozen = getattr(self.cfg, "MARGINALIZE_REFRESH_TARGETS", False)
        if target is None:
            if len(self.ii) == 0:
                if not use_frozen:
                    return self.ii, self.jj, self.kk, self.active_target, self.active_weight
                marg_target = self.refreshed_marginalized_targets() if refresh_frozen else self.marg_target
                return self.marg_ii, self.marg_jj, self.marg_kk, marg_target, \
                    self.marg_weight * self.marg_weight_scale * frozen_weight_scale
            target = self.active_target
            weight = self.active_weight

        if len(self.marg_ii) == 0 or not use_frozen:
            return self.ii, self.jj, self.kk, target.float(), weight.float()

        marg_target = self.refreshed_marginalized_targets() if refresh_frozen else self.marg_target
        ii = torch.cat([self.ii, self.marg_ii])
        jj = torch.cat([self.jj, self.marg_jj])
        kk = torch.cat([self.kk, self.marg_kk])
        target = torch.cat([target.float(), marg_target], dim=1)
        marg_weight = self.marg_weight * self.marg_weight_scale * frozen_weight_scale
        weight = torch.cat([weight.float(), marg_weight], dim=1)
        return ii, jj, kk, target, weight

    def refreshed_marginalized_targets(self):
        if len(self.marg_ii) == 0:
            return self.marg_target

        coords = self.reproject(indicies=(self.marg_ii, self.marg_jj, self.marg_kk))
        current = coords[...,self.P//2,self.P//2]
        return current + self.marg_delta

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.dim_inet, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with torch.inference_mode():
            with autocast(enabled=self.cfg.MIXED_PRECISION):
                corr = self.corr(coords, indicies=(kk, jj))
                ctx = self.imap[:,kk % (self.M * self.mem)]
                net, (delta, weight, _) = \
                    self.run_update_net(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        if getattr(self.cfg, "MARGINALIZE_USE_FROZEN_IN_BA", True):
            mk = (self.marg_ii == i) & (self.marg_jj == j)
            ii = torch.cat([self.ii[k], self.marg_ii[mk]])
            jj = torch.cat([self.jj[k], self.marg_jj[mk]])
            kk = torch.cat([self.kk[k], self.marg_kk[mk]])
        else:
            ii = self.ii[k]
            jj = self.jj[k]
            kk = self.kk[k]

        if len(ii) == 0:
            return float("inf")

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        # described in 3.3. Keyframing DPVO paper
        # "after each update, compute flow_mag <t-5, t-3> and remove <t-4> if less than 64px"
        i = self.n - self.cfg.KEYFRAME_INDEX - 1 # t-5, KF_INDEX = 4 per default
        j = self.n - self.cfg.KEYFRAME_INDEX + 1 # t-3
        m = self.motionmag(i, j) + self.motionmag(j, i) 
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX # scalar
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP) # store relative pose between <t-5, t-4>

            to_remove = (self.ii == k) | (self.jj == k)
            marg_to_remove = (self.marg_ii == k) | (self.marg_jj == k)
            self.remove_all_factors(to_remove, marg_to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1
            self.marg_kk[self.marg_ii > k] -= self.M
            self.marg_ii[self.marg_ii > k] -= 1
            self.marg_jj[self.marg_jj > k] -= 1
            self.bump_graph_version()

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.patches_gt_[i] = self.patches_gt_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1 # remove frame
            self.m -= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        marg_to_remove = self.ix[self.marg_kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_all_factors(to_remove, marg_to_remove)

    def update(self):
        self.marginalize_update_count += 1
        self.marginalize_cached_factors()
        self.order_active_factors()
        self.print_marginalization_stats()

        if len(self.ii) > 0:
            all_neural = not getattr(self.cfg, "WARM_UPDATE_ENABLED", False) and \
                not getattr(self.cfg, "ADAPTIVE_UPDATE_ENABLED", False)
            neural = None if all_neural else self.select_adaptive_neural_factors()

            if all_neural or neural.any():
                ii = self.ii if all_neural else self.ii[neural]
                jj = self.jj if all_neural else self.jj[neural]
                kk = self.kk if all_neural else self.kk[neural]
                net_in = self.net if all_neural else self.net[:,neural]
                with torch.inference_mode():
                    coords = self.reproject(indicies=(ii, jj, kk))

                    with autocast(enabled=True):

                        corr = self.corr(coords, indicies=(kk, jj))
                        ctx = self.imap[:,kk % (self.M * self.mem)]
                        topology = self.update_topology(ii, jj, kk, cacheable=all_neural)
                        with Timer("other", enabled=self.enable_timing):
                            net, (delta, weight, _) = \
                                self.run_update_net(net_in, ctx, corr, None, ii, jj, kk, topology)

                if all_neural:
                    self.net = net
                else:
                    self.net[:,neural] = net
                weight = weight.float()
                target = coords[...,self.P//2,self.P//2] + delta.float()
                confidence = weight[0].mean(dim=-1)
                delta_norm = delta[0].float().norm(dim=-1)

                if all_neural:
                    self.active_target = target.detach().float()
                    self.active_weight = weight.detach().float()
                    self.active_delta = delta.detach().float()
                    self.active_confidence = confidence.detach().float()
                    self.active_delta_norm = delta_norm.detach().float()
                else:
                    self.active_target[:,neural] = target.detach().float()
                    self.active_weight[:,neural] = weight.detach().float()
                    self.active_delta[:,neural] = delta.detach().float()
                    self.active_confidence[neural] = confidence.detach().float()
                    self.active_delta_norm[neural] = delta_norm.detach().float()
                self.update_active_budget(confidence, delta_norm)

            to_marginalize = self.select_marginalized_factors(
                self.active_confidence, self.active_delta_norm)
            self.marginalize_factors(to_marginalize, self.active_target, self.active_weight)

        if self.active_keepalive.numel() > 0:
            self.active_keepalive = torch.clamp(self.active_keepalive - 1, min=0)

        if self.marginalize_freeze_cooldown > 0:
            self.marginalize_freeze_cooldown -= 1

        # Decay frozen edge weights so stale targets fade out gracefully.
        if self.marg_weight.numel() > 0:
            decay = getattr(self.cfg, "MARGINALIZE_WEIGHT_DECAY", 0.99)
            self.marg_weight *= decay

        lmbda = torch.as_tensor([1e-4], device="cuda")
        self.validate_marginalized_factors()
        ba_ii, ba_jj, ba_kk, ba_target, ba_weight = self.ba_factors()

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            if len(ba_ii) > 0:
                try:
                    fastba.BA(self.poses, self.patches, self.intrinsics,
                        ba_target, ba_weight, lmbda, ba_ii, ba_jj, ba_kk, t0, self.n,
                        getattr(self.cfg, "BA_ITERATIONS", 2))
                except:
                    print("Warning BA failed...")
            
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]

    def flow_viz_step(self):
        # [DEBUG]
        # dij = (self.ii - self.jj).abs()
        # assert (dij==0).sum().item() == len(torch.unique(self.kk)) 
        # [DEBUG]

        coords_est = pops.transform(SE3(self.poses), self.patches, self.intrinsics, self.ii, self.jj, self.kk) # p_ij (B,close_edges,P,P,2)
        self.flow_data[self.counter-1] = {"ii": self.ii, "jj": self.jj, "kk": self.kk,\
                                          "coords_est": coords_est, "img": self.image_, "n": self.n}

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(self.image_)
        # plt.show()
                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    def __thin_edges(self, ii, jj):
        if not getattr(self.cfg, "EDGE_THINNING", False) or len(ii) == 0:
            return ii, jj

        stride = max(getattr(self.cfg, "EDGE_THIN_STRIDE", 1), 1)
        if stride <= 1:
            return ii, jj

        dense_window = getattr(self.cfg, "EDGE_THIN_DENSE_WINDOW", 3)
        patch_frame = self.ix[ii]
        recent = (patch_frame - jj).abs() <= dense_window
        keep = recent | ((ii % stride) == 0)
        return ii[keep], jj[keep]

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        ii, jj = flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')
        return self.__thin_edges(ii, jj)

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        ii, jj = flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')
        return self.__thin_edges(ii, jj)

    def __call__(self, tstamp, image, intrinsics, scale=1.0):
        """ track new frame """

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image)

        if self.viz_flow:
            self.image_ = image.detach().cpu().permute((1, 2, 0)).numpy()

        if not self.evs:
            image = 2 * (image[None,None] / 255.0) - 0.5 
        else:
            image = image[None,None]

            # [DEBUG]
            # import matplotlib
            # matplotlib.use('Qt5Agg')
            # visualize_voxel(image[0][0].detach().cpu(), EPS=1e-3)
            # i2 = image[image!=0]
            # print("stats before norm", i2.min().item(), i2.max().item(), i2.mean().item(), i2.std().item(), i2.median().item())
            
            if self.n == 0:
                nonzero_ev = (image != 0.0)
                zero_ev = ~nonzero_ev
                num_nonzeros = nonzero_ev.sum().item()
                num_zeros = zero_ev.sum().item()
                # [DEBUG]
                # print("nonzero-zero-ratio", num_nonzeros, num_zeros, num_nonzeros / (num_zeros + num_nonzeros))
                if num_nonzeros / (num_zeros + num_nonzeros) < 2e-2: # TODO eval hyperparam (add to config.py)
                    print(f"skip voxel at {tstamp} due to lack of events!")
                    return

            b, n, v, h, w = image.shape
            flatten_image = image.view(b,n,-1)
            
            if self.cfg.NORM.lower() == 'none':
                pass
            elif self.cfg.NORM.lower() == 'rescale' or self.cfg.NORM.lower() == 'norm':
                # Normalize (rescaling) neg events into [-1,0) and pos events into (0,1] sequence-wise
                # Preserve pos-neg inequality (quantity only)
                pos = flatten_image > 0.0
                neg = flatten_image < 0.0
                vx_max = torch.Tensor([1]).to("cuda") if pos.sum().item() == 0 else flatten_image[pos].max(dim=-1, keepdim=True)[0]
                vx_min = torch.Tensor([1]).to("cuda") if neg.sum().item() == 0 else flatten_image[neg].min(dim=-1, keepdim=True)[0]
                # [DEBUG]
                # print("vx_max", vx_max.item())
                # print("vx_min", vx_min.item())
                if vx_min.item() == 0.0 or vx_max.item() == 0.0:
                    # no information for at least one polarity
                    print(f"empty voxel at {tstamp}!")
                    return
                flatten_image[pos] = flatten_image[pos] / vx_max
                flatten_image[neg] = flatten_image[neg] / -vx_min
            elif self.cfg.NORM.lower() == 'standard' or self.cfg.NORM.lower() == 'std':
                # Data standardization of events only
                # Does not preserve pos-neg inequality
                # see https://github.com/uzh-rpg/rpg_e2depth/blob/master/utils/event_tensor_utils.py#L52
                nonzero_ev = (flatten_image != 0.0)
                num_nonzeros = nonzero_ev.sum(dim=-1)
                if torch.all(num_nonzeros > 0):
                    # compute mean and stddev of the **nonzero** elements of the event tensor
                    # we do not use PyTorch's default mean() and std() functions since it's faster
                    # to compute it by hand than applying those funcs to a masked array

                    mean = torch.sum(flatten_image, dim=-1, dtype=torch.float32) / num_nonzeros  # force torch.float32 to prevent overflows when using 16-bit precision
                    stddev = torch.sqrt(torch.sum(flatten_image ** 2, dim=-1, dtype=torch.float32) / num_nonzeros - mean ** 2)
                    mask = nonzero_ev.type_as(flatten_image)
                    flatten_image = mask * (flatten_image - mean[...,None]) / stddev[...,None]
            else:
                print(f"{self.cfg.NORM} not implemented")
                raise NotImplementedError

            image = flatten_image.view(b,n,v,h,w)

            # [DEBUG]
            # import matplotlib
            # matplotlib.use('Qt5Agg')
            # visualize_voxel(image[0][0].detach().cpu(), EPS=1e-3)
            # i2 = image[image!=0]
            # print(f"stats after norm={self.cfg.NORM}", i2.min().item(), i2.max().item(), i2.mean().item(), i2.std().item(), i2.median().item())

        if image.shape[-1] == 346:
            image = image[..., 1:-1] # hack for MVSEC, FPV,...
    
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(image.detach().cpu().numpy()[0, 0, :1, ...].transpose(1, 2, 0))
        # plt.show()

        # TODO patches with depth is available (val)
        with torch.inference_mode():
            with autocast(enabled=self.cfg.MIXED_PRECISION):
                fmap, gmap, imap, patches, _, clr = \
                    self.network.patchify(image,
                        patches_per_image=self.cfg.PATCHES_PER_FRAME,
                        return_color=True,
                        scorer_eval_mode=self.cfg.SCORER_EVAL_MODE,
                        scorer_eval_use_grid=self.cfg.SCORER_EVAL_USE_GRID)

        self.patches_gt_[self.n] = patches.clone()

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES
        
        # color info for visualization
        if not self.evs:
            clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
            self.colors_[self.n] = clr.to(torch.uint8)
        else:
            clr = (clr[0,:,[0,0,0]] + 0.5) * (255.0 / 2)
            self.colors_[self.n] = clr.to(torch.uint8)
            

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1

        if self.n > 0 and not self.is_initialized:
            thres = 2.0 if scale == 1.0 else scale ** 2 # TODO adapt thres for lite version
            if self.motion_probe() < thres: # TODO: replace by 8 pixels flow criterion (as described in 3.3 Initialization)
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1 # add one (key)frame
        self.m += self.M # add patches per (key)frames to patch number

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            

            for itr in range(12):
                self.update()
        
        elif self.is_initialized:
            self.update()
            self.keyframe()

        if self.viz_flow:
            self.flow_viz_step()
