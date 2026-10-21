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
    time). ConnectomeRecurrentPolicy (--memory) instead carries the neurons'
    activity on from one control step to the next (reset only when an episode
    starts), so the wiring's own loops can hold information over time; it is
    trained with recurrent PPO (sb3-contrib) and sequence imitation.

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
            # smell (grown brain, grow_smell_brain.py): the LAST n_smell brain inputs are the two
            # antennae's receptor responses, one per olfactory receptor neuron (gain + bias each, like
            # the photoreceptors); the body sense sits between the eyes and the smell inputs
            self.n_smell = 0
            if "smell_idx" in d.files:
                self.register_buffer("smell_idx", torch.as_tensor(d["smell_idx"], dtype=torch.long))
                self.n_smell = len(self.smell_idx)
                self.smell_gain = nn.Parameter(torch.ones(self.n_smell))
                self.smell_bias = nn.Parameter(torch.zeros(self.n_smell))
            self.w_body = nn.Linear(obs_dim - n_eye - self.n_smell, len(self.asc_idx))
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
        return self.step(obs, None)[0]

    def step(self, obs: torch.Tensor, h0):
        """One control step from neuron state h0 (None = all zero): (output-neuron activity, final state)."""
        b = obs.shape[0]
        A = None if self.use_csr else self.weight_matrix()
        w = self.edge_weights() if self.use_csr else None
        u = obs.new_zeros(b, self.n)
        if self.flyrobot:
            x = obs[:, :self.n_vis] * self.pix_gain + self.pix_bias
            if self.vis_dyn:
                x = x + obs[:, self.n_vis:self.n_eye] * self.pix_gain_change
            u[:, self.vis_idx] = x
            end = obs.shape[1] - self.n_smell
            u[:, self.asc_idx] = self.w_body(obs[:, self.n_eye:end])
            if self.n_smell:
                u[:, self.smell_idx] = obs[:, end:] * self.smell_gain + self.smell_bias
        else:
            u[:, self.input_idx] = self.w_in(obs)
        h = obs.new_zeros(b, self.n) if h0 is None else h0
        for _ in range(self.k_steps):
            if self.use_csr:
                rec = _SparseRecurrent.apply(h, w, self)
            elif A.is_sparse:
                rec = torch.sparse.mm(A, h.T).T
            else:
                rec = h @ A.T
            h = (1 - self.leak) * h + self.leak * torch.tanh(rec + u + self.bias)
        return h[:, self.output_idx], h


class ConnectomeExtractor(nn.Module):
    """SB3 'mlp_extractor': returns (actor latent, critic latent)."""

    def __init__(self, obs_dim, subset_path, k_steps, leak, gain, wiring, critic_hidden=64, n_priv=0, mask_inputs=()):
        super().__init__()
        # n_priv > 0: the LAST n_priv observation values are the simulator's true state (block,
        # goal, grasp ...). Only the critic gets them (asymmetric actor-critic, Pinto et al. 2017);
        # the fly brain is built for, and only ever receives, the first obs_dim - n_priv values.
        self.n_priv = n_priv
        self.n_brain = obs_dim - n_priv
        self.core = ConnectomeCore(self.n_brain, subset_path, k_steps, leak, gain, wiring)
        # brain inputs held at zero (e.g. the previous command: copying it is the classic
        # "copycat" shortcut in imitation learning, so the student must read its eyes instead)
        self.register_buffer("mask_idx", torch.as_tensor(list(mask_inputs), dtype=torch.long), persistent=False)
        # The output neurons' activity is small and similar across inputs at init.
        # A LayerNorm readout rescales it so the action head gets a usable signal.
        self.readout_norm = nn.LayerNorm(self.core.out_dim)
        crit_in = n_priv if n_priv else obs_dim
        if n_priv:
            critic_hidden = max(critic_hidden, 128)
        self.critic = nn.Sequential(nn.Linear(crit_in, critic_hidden), nn.Tanh(),
                                    nn.Linear(critic_hidden, critic_hidden), nn.Tanh())
        self.latent_dim_pi = self.core.out_dim
        self.latent_dim_vf = critic_hidden

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x):
        x = x[:, :self.n_brain]                                       # the fly brain: eyes + body only
        if len(self.mask_idx):
            x = x.clone()
            x[:, self.mask_idx] = 0.0
        return self.readout_norm(self.core(x))

    def forward_critic(self, x):
        return self.critic(x[:, self.n_brain:] if self.n_priv else x)

    def step_actor(self, x, h):
        """With memory: one control step from neuron state h -> (actor latent, new state)."""
        x = x[:, :self.n_brain]
        if len(self.mask_idx):
            x = x.clone()
            x[:, self.mask_idx] = 0.0
        out, h = self.core.step(x, h)
        return self.readout_norm(out), h


class ConnectomePolicy(ActorCriticPolicy):
    def __init__(self, *args, subset_path: str = "", k_steps: int = 4, leak: float = 0.5,
                 gain: float = 1.5, wiring: str = "real", n_priv: int = 0, mask_inputs=(), **kwargs):
        self._cfg = dict(subset_path=subset_path, k_steps=k_steps, leak=leak, gain=gain, wiring=wiring, n_priv=n_priv,
                         mask_inputs=tuple(mask_inputs))
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
                  wiring=cfg.get("wiring", "real"), log_std_init=-0.5, n_priv=int(cfg.get("n_priv", 0) or 0),
                  mask_inputs=tuple(cfg.get("mask_inputs", ()) or ()))
    if cfg.get("memory"):                       # fly brain with memory (recurrent PPO, sb3-contrib)
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(model_path, env=env, device=device,
                                 custom_objects={"policy_class": recurrent_policy_class(), "policy_kwargs": kwargs,
                                                 **(custom or {})})
    return PPO.load(model_path, env=env, device=device,
                    custom_objects={"policy_class": ConnectomePolicy, "policy_kwargs": kwargs, **(custom or {})})


# --------------------------------------------------------------------------------------------- memory
class _CoreAsRNN(nn.Module):
    """Stands in for the LSTM of sb3-contrib's recurrent policy: same call signature
    (sequence, (h, c)) -> (outputs, (h, c)), where h is the fly brain's neuron activity
    (c is unused, kept as zeros). Holds no parameters of its own: they live in the policy's
    mlp_extractor (so names and init_from copies are the same as without memory)."""

    def __init__(self, extractor, input_size):
        super().__init__()
        self.__dict__["_ext"] = extractor           # a reference, not a sub-module (no duplicate parameters)
        self.input_size, self.hidden_size, self.num_layers = input_size, extractor.core.n, 1

    def forward(self, x_seq, state):
        h, c = state
        h = h[0]
        outs = []
        for x in x_seq:                               # (sequence length, batch, features)
            out, h = self._ext.step_actor(x, h)
            outs.append(out)
        return torch.stack(outs), (h.unsqueeze(0), c)


class _CriticFromObs(nn.Module):
    """The critic of the recurrent policy: the same small network on the privileged state."""

    def __init__(self, extractor):
        super().__init__()
        self.__dict__["_ext"] = extractor

    def forward(self, features):
        return self._ext.forward_critic(features)


MEM_INIT = -3.0       # carry-over logit at the start: sigmoid(-3) = 5 % of each neuron's activity is kept


class RecurrentConnectomeExtractor(ConnectomeExtractor):
    """Latents arrive already computed (actor: by _CoreAsRNN; critic: by _CriticFromObs).

    Memory = a learned per-neuron carry-over: a control step starts from r * (last step's activity),
    r = sigmoid(mem_logit) in (0, 1). It starts small (5 %), so the brain copied from a memoryless
    parent first behaves almost as before, and learning then raises r wherever remembering helps
    (warm start + gated recurrence, as in GRU/LSTM forget gates; a full carry-over from the start
    turned the parent's dynamics chaotic and imitation diverged)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem_logit = nn.Parameter(torch.full((self.core.n,), MEM_INIT))

    def step_actor(self, x, h):
        if h is not None:
            h = torch.sigmoid(self.mem_logit) * h
        return ConnectomeExtractor.step_actor(self, x, h)

    def forward_actor(self, latent):
        return latent

    def forward_critic(self, latent):
        return latent

    def critic_latent(self, x):
        return ConnectomeExtractor.forward_critic(self, x)


def _make_recurrent_policy_class():
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

    class ConnectomeRecurrentPolicy(RecurrentActorCriticPolicy):
        """The fly brain WITH memory: neuron activity carries over between control steps."""

        def __init__(self, observation_space, action_space, lr_schedule, *args, subset_path: str = "",
                     k_steps: int = 4, leak: float = 0.5, gain: float = 1.5, wiring: str = "real", n_priv: int = 0,
                     mask_inputs=(), **kwargs):
            self._cfg = dict(subset_path=subset_path, k_steps=k_steps, leak=leak, gain=gain, wiring=wiring,
                             n_priv=n_priv, mask_inputs=tuple(mask_inputs))
            kwargs.update(ortho_init=False, lstm_hidden_size=1, n_lstm_layers=1, shared_lstm=False,
                          enable_critic_lstm=False)
            super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
            ext = self.mlp_extractor
            self.lstm_actor = _CoreAsRNN(ext, self.features_dim)
            self.critic = _CriticFromObs(_CriticAdapter(ext))
            self.lstm_hidden_state_shape = (1, 1, ext.core.n)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        def _build_mlp_extractor(self):
            self.mlp_extractor = RecurrentConnectomeExtractor(self.features_dim, **self._cfg)

    return ConnectomeRecurrentPolicy


class _CriticAdapter:
    def __init__(self, ext):
        self.ext = ext

    def forward_critic(self, x):
        return self.ext.critic_latent(x)


_RECURRENT = None


def recurrent_policy_class():
    global _RECURRENT
    if _RECURRENT is None:
        _RECURRENT = _make_recurrent_policy_class()
    return _RECURRENT


class StatefulModel:
    """A loaded model for evaluation / videos: predict() keeps the fly brain's neuron state from
    one call to the next when the run has memory. Call reset_memory() after every env.reset()
    (a no-op for runs without memory). Everything else is passed through to the model."""

    def __init__(self, model):
        self.__dict__["model"] = model
        self.__dict__["memory"] = hasattr(model.policy, "lstm_actor")
        self.__dict__["_state"] = None
        self.__dict__["_start"] = True

    def reset_memory(self):
        self.__dict__["_state"], self.__dict__["_start"] = None, True

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        if not self.memory:
            return self.model.predict(obs, deterministic=deterministic)
        n = len(obs)
        a, st = self.model.predict(obs, state=self._state, deterministic=deterministic,
                                   episode_start=np.full(n, self._start, bool))
        self.__dict__["_state"], self.__dict__["_start"] = st, bool(self.__dict__.get("forget"))   # forget: test only
        return a, st

    def __getattr__(self, k):
        return getattr(self.model, k)
