"""Policies for Stable-Baselines3 PPO.

ConnectomePolicy
    The actor is a recurrent network whose wiring is copied from the fly
    connectome subset. Which neuron talks to which, and whether that
    connection is excitatory or inhibitory, is FIXED. Only the strength of
    each existing connection (plus input weights and neuron biases) is trained.

    Each control step:
        u        = W_in @ obs                  (written into the input neurons)
        repeat K times:
            h   = (1 - a) * h + a * tanh( A @ h + u + b )
        latent   = LayerNorm(h[output neurons]) -> SB3 adds a linear action head

    The hidden state starts from zero at every control step (no memory across
    time). That keeps PPO simple; adding memory is a later experiment.

    wiring="shuffled" keeps the same number of edges, signs and weights but
    connects them to random neurons. Useful control for "is it the wiring?".

The critic (value function) is a small standard network: it only estimates how
good a situation is during training and is not part of the controller.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy


warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")
warnings.filterwarnings("ignore", message=".*Sparse invariant checks are implicitly disabled.*")


class _SparseRecurrent(torch.autograd.Function):
    """rec = h @ A.T for a sparse A given by edge weights w (A[post, pre] = w).

    Uses CSR matrix products without PyTorch's sparse autograd (which leaks memory
    in some versions: ~4 MB per update, enough to crash an 8 GB machine). Gradients are
    written out by hand: d(h) = grad @ A, d(w_e) = sum_b grad[b, post_e] * h[b, pre_e]
    (the latter with a sampled matrix product, so only existing edges are computed)."""

    @staticmethod
    def forward(ctx, h, w, core):
        with torch.no_grad():
            A = torch.sparse_csr_tensor(core.csr_crow, core.csr_col, w[core.csr_order], (core.n, core.n))
            out = torch.sparse.mm(A, h.T).T
        ctx.save_for_backward(h, w)
        ctx.core = core
        return out

    @staticmethod
    def backward(ctx, grad):
        h, w = ctx.saved_tensors
        core = ctx.core
        with torch.no_grad():
            At = torch.sparse_csr_tensor(core.csrT_crow, core.csrT_col, w[core.csrT_order], (core.n, core.n))
            grad_h = torch.sparse.mm(At, grad.T).T
            # d(w_e) = (grad.T @ h)[post_e, pre_e], computed only at existing edges
            patt = torch.sparse_csr_tensor(core.csr_crow, core.csr_col, w.new_zeros(len(w)), (core.n, core.n))
            vals = torch.sparse.sampled_addmm(patt, grad.T.contiguous(), h, beta=0.0).values()
            grad_w = torch.empty_like(w)
            grad_w[core.csr_order] = vals
        return grad_h, grad_w, None


def _inv_softplus(y):
    return y + torch.log(-torch.expm1(-y))


class ConnectomeCore(nn.Module):
    def __init__(self, obs_dim: int, subset_path: str, k_steps: int = 4, leak: float = 0.5,
                 gain: float = 1.5, wiring: str = "real", seed: int = 0):
        super().__init__()
        d = np.load(subset_path)
        n = int(d["n"])
        pre = torch.as_tensor(d["pre"], dtype=torch.long)
        post = torch.as_tensor(d["post"], dtype=torch.long)
        count = torch.as_tensor(d["count"], dtype=torch.float32)
        sign = torch.as_tensor(d["sign"], dtype=torch.float32)

        if wiring == "shuffled":
            g = torch.Generator().manual_seed(seed)
            pre = torch.randint(0, n, pre.shape, generator=g)
            post = torch.randint(0, n, post.shape, generator=g)
        elif wiring != "real":
            raise ValueError(wiring)

        # initial strength: synapse count normalised by each neuron's total input
        in_strength = torch.zeros(n).index_add_(0, post, count).clamp_min(1.0)
        w0 = gain * count / in_strength[post]

        self.n = n
        self.k_steps = k_steps
        self.leak = leak
        self.register_buffer("pre", pre)
        self.register_buffer("post", post)
        self.register_buffer("sign", sign)
        self.register_buffer("input_idx", torch.as_tensor(d["input_idx"], dtype=torch.long))
        self.register_buffer("output_idx", torch.as_tensor(d["output_idx"], dtype=torch.long))
        self.theta = nn.Parameter(_inv_softplus(w0.clamp_min(1e-4)))   # strength > 0 always
        self.bias = nn.Parameter(torch.zeros(n))
        self.out_dim = len(self.output_idx)
        self.density = len(pre) / (n * n)
        # very sparse brains (the fly-robot one is 0.5 % connected) run faster as a CSR
        # sparse matrix; precompute the row order once
        self.use_csr = self.density < 0.02
        if self.use_csr:
            order = torch.argsort(post * n + pre)
            crow = torch.zeros(n + 1, dtype=torch.long)
            crow[1:] = torch.cumsum(torch.bincount(post[order], minlength=n), 0)
            self.register_buffer("csr_order", order)
            self.register_buffer("csr_crow", crow)
            self.register_buffer("csr_col", pre[order])
            order_t = torch.argsort(pre * n + post)          # transpose, for the backward pass
            crow_t = torch.zeros(n + 1, dtype=torch.long)
            crow_t[1:] = torch.cumsum(torch.bincount(pre[order_t], minlength=n), 0)
            self.register_buffer("csrT_order", order_t)
            self.register_buffer("csrT_crow", crow_t)
            self.register_buffer("csrT_col", post[order_t])

        # Fly-robot brain (build_flybrain.py): each camera pixel drives ONE photoreceptor
        # (trainable gain per pixel, no mixing, like a real eye), and the body-sense
        # numbers go into the ascending neurons. Otherwise: one learned layer into all inputs.
        self.flyrobot = "vis_idx" in d.files
        if self.flyrobot:
            self.register_buffer("vis_idx", torch.as_tensor(d["vis_idx"], dtype=torch.long))
            self.register_buffer("asc_idx", torch.as_tensor(d["asc_idx"], dtype=torch.long))
            self.n_vis = len(self.vis_idx)
            # compound eye (drone): each photoreceptor gets the light it sees now AND how
            # much that just changed (real photoreceptors respond strongly to change)
            self.vis_dyn = "vis_dyn" in d.files and bool(d["vis_dyn"])
            n_eye = self.n_vis * (2 if self.vis_dyn else 1)
            if obs_dim < n_eye:
                raise ValueError(f"fly-robot brain needs a camera env: obs has {obs_dim} values, "
                                 f"the eye has {self.n_vis} photoreceptors (use a *_vision env)")
            self.pix_gain = nn.Parameter(torch.ones(self.n_vis))
            self.pix_bias = nn.Parameter(torch.zeros(self.n_vis))
            if self.vis_dyn:
                self.pix_gain_change = nn.Parameter(torch.ones(self.n_vis))
            self.n_eye = n_eye
            self.w_body = nn.Linear(obs_dim - n_eye, len(self.asc_idx))
        else:
            self.w_in = nn.Linear(obs_dim, len(self.input_idx))

    def edge_weights(self):
        return self.sign * F.softplus(self.theta)   # sign fixed (Dale's law)

    def weight_matrix(self):
        """Connectivity matrix A[post, pre]. Only existing edges are non-zero, and
        only those entries get gradients, so the wiring never changes.
        Dense for small, fairly dense networks (fastest on CPU there); CSR sparse for
        very sparse ones; COO sparse for large dense-ish ones."""
        w = self.edge_weights()
        if self.use_csr:
            return torch.sparse_csr_tensor(self.csr_crow, self.csr_col, w[self.csr_order], (self.n, self.n))
        if self.n <= 3000:
            A = w.new_zeros(self.n, self.n)
            return A.index_put((self.post, self.pre), w, accumulate=True)
        idx = torch.stack([self.post, self.pre])
        return torch.sparse_coo_tensor(idx, w, (self.n, self.n)).coalesce()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        A = None if self.use_csr else self.weight_matrix()
        w = self.edge_weights() if self.use_csr else None
        u = obs.new_zeros(b, self.n)
        if self.flyrobot:
            x = obs[:, :self.n_vis] * self.pix_gain + self.pix_bias
            if self.vis_dyn:
                x = x + obs[:, self.n_vis:self.n_eye] * self.pix_gain_change
            u[:, self.vis_idx] = x
            u[:, self.asc_idx] = self.w_body(obs[:, self.n_eye:])
        else:
            u[:, self.input_idx] = self.w_in(obs)
        h = obs.new_zeros(b, self.n)
        for _ in range(self.k_steps):
            if self.use_csr:
                rec = _SparseRecurrent.apply(h, w, self)
            elif A.is_sparse:
                rec = torch.sparse.mm(A, h.T).T
            else:
                rec = h @ A.T
            h = (1 - self.leak) * h + self.leak * torch.tanh(rec + u + self.bias)
        return h[:, self.output_idx]


class ConnectomeExtractor(nn.Module):
    """SB3 'mlp_extractor': returns (actor latent, critic latent)."""

    def __init__(self, obs_dim, subset_path, k_steps, leak, gain, wiring, critic_hidden=64):
        super().__init__()
        self.core = ConnectomeCore(obs_dim, subset_path, k_steps, leak, gain, wiring)
        # The output neurons' activity is small and similar across inputs at init.
        # A LayerNorm readout rescales it so the action head gets a usable signal.
        self.readout_norm = nn.LayerNorm(self.core.out_dim)
        self.critic = nn.Sequential(nn.Linear(obs_dim, critic_hidden), nn.Tanh(),
                                    nn.Linear(critic_hidden, critic_hidden), nn.Tanh())
        self.latent_dim_pi = self.core.out_dim
        self.latent_dim_vf = critic_hidden

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x):
        return self.readout_norm(self.core(x))

    def forward_critic(self, x):
        return self.critic(x)


class ConnectomePolicy(ActorCriticPolicy):
    def __init__(self, *args, subset_path: str = "", k_steps: int = 4, leak: float = 0.5,
                 gain: float = 1.5, wiring: str = "real", **kwargs):
        self._cfg = dict(subset_path=subset_path, k_steps=k_steps, leak=leak, gain=gain, wiring=wiring)
        kwargs["ortho_init"] = False   # keep the connectome-based initial weights
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ConnectomeExtractor(self.features_dim, **self._cfg)


def load_ppo(run_dir: str, model_path: str, env=None, device: str = "cpu", custom=None):
    """Load a saved PPO run on any machine.

    A connectome model remembers the absolute path of its wiring file from the
    machine it was trained on (e.g. /kaggle/working/...). Here that path is
    swapped for <run>/subset.npz, or data/<same file name> in this repo."""
    import json
    import os

    from stable_baselines3 import PPO

    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    if cfg["policy"] != "connectome":
        return PPO.load(model_path, env=env, device=device)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [os.path.join(run_dir, "subset.npz"),
                  os.path.join(root, "data", os.path.basename(cfg["subset"])),
                  cfg["subset"]]
    subset = next((c for c in candidates if os.path.exists(c)), None)
    if subset is None:
        raise FileNotFoundError(f"Wiring file not found. Tried: {candidates}. "
                                "Run build_subset.py with the same --n as the run.")
    kwargs = dict(subset_path=os.path.abspath(subset), k_steps=cfg.get("k_steps") or 4,
                  leak=cfg.get("leak") if cfg.get("leak") is not None else 0.5,
                  gain=cfg.get("gain") if cfg.get("gain") is not None else 1.5,
                  wiring=cfg.get("wiring", "real"), log_std_init=-0.5)
    return PPO.load(model_path, env=env, device=device,
                    custom_objects={"policy_class": ConnectomePolicy, "policy_kwargs": kwargs, **(custom or {})})
