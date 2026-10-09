from __future__ import annotations

import torch
from torch import nn


class PartEmbeddingModel(nn.Module):
    """K+1 Gaussian memberships from learned embeddings and prototypes.

    FIRM v3 §22: static anchors are *initialised* strongly static (p_i0 ~= 0.999)
    rather than hard-pinned on every probability call.  Permanent pinning made
    ``L_anchor = -log(1) = 0`` with an identically zero gradient, so a
    spec-required term was decorative.  Hard pinning is still available, but only
    for a finite warmup (``apply_anchor=True`` during the first
    ``anchor_hard_freeze_outer_iters`` outer iterations).
    """

    def __init__(
        self,
        num_gaussians: int,
        num_parts: int,
        embedding_dim: int,
        *,
        temperature: float = 1.0,
        static_anchor_mask: torch.Tensor | None = None,
        init_std: float = 0.05,
        seed: int = 20260918,
        anchor_target_probability: float = 0.999,
        init_logits: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        n, k, d = int(num_gaussians), int(num_parts), int(embedding_dim)
        if n <= 0 or k < 2 or d <= 0:
            raise ValueError("PartEmbeddingModel requires N>0, K+1>=2, D>0")

        # FINAL spec §2/§8: p_ik = softmax(l_ik) with the static logit biased by
        # the diffused anchor affinity and every other logit random. When
        # ``init_logits`` is supplied the model parameterises those logits
        # directly instead of going through embedding/prototype distances.
        self.direct_logits = init_logits is not None
        if self.direct_logits:
            logits = torch.as_tensor(init_logits, dtype=torch.float32).clone()
            if tuple(logits.shape) != (n, k):
                raise ValueError(f"init_logits must be [{n}, {k}], got {tuple(logits.shape)}")
            self.logits = nn.Parameter(logits)
            # Prototypes are kept only so the K>1 embedding/motion term has a
            # handle; they carry no membership meaning in this mode.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            self.prototype = nn.Parameter(torch.randn((k, d), generator=generator) * float(init_std))
            self.embedding = nn.Parameter(torch.zeros((n, d)))
            self.temperature = float(temperature)
            self.anchor_target_probability = float(anchor_target_probability)
            mask = (
                torch.zeros(n, dtype=torch.bool)
                if static_anchor_mask is None
                else torch.as_tensor(static_anchor_mask).detach().cpu().bool().reshape(-1)
            )
            self.register_buffer("static_anchor_mask", mask)
            return

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        # §21: every non-anchor embedding is random. No moving initialisation,
        # no prior, no legacy K-part.
        emb = torch.randn((n, d), generator=generator) * float(init_std)
        proto = torch.randn((k, d), generator=generator) * float(init_std)
        proto[0].zero_()

        mask = None
        if static_anchor_mask is not None:
            mask = torch.as_tensor(static_anchor_mask).detach().cpu().bool().reshape(-1)
            if mask.numel() != n:
                raise ValueError("static_anchor_mask does not match Gaussian count")
            anchor_embedding = _solve_anchor_embedding(
                proto,
                temperature=float(temperature),
                target_probability=float(anchor_target_probability),
            )
            emb[mask] = anchor_embedding

        self.embedding = nn.Parameter(emb)
        self.prototype = nn.Parameter(proto)
        self.temperature = float(temperature)
        self.anchor_target_probability = float(anchor_target_probability)
        self.register_buffer(
            "static_anchor_mask",
            torch.zeros(n, dtype=torch.bool) if mask is None else mask,
        )

    def probabilities(self, temperature: float | None = None, *, apply_anchor: bool = False) -> torch.Tensor:
        """Memberships p_ik = softmax(-||e_i - c_k||^2 / tau).

        ``apply_anchor=True`` hard-pins anchors to the static class. FIRM v3 §22
        allows that only for a finite warmup, so it defaults to False.
        """
        temp = max(float(self.temperature if temperature is None else temperature), 1.0e-6)
        if self.direct_logits:
            prob = torch.softmax(self.logits / temp, dim=-1)
        else:
            dist2 = (self.embedding[:, None, :] - self.prototype[None, :, :]).square().sum(dim=-1)
            prob = torch.softmax(-dist2 / temp, dim=-1)
        if bool(apply_anchor) and bool(self.static_anchor_mask.any()):
            prob = prob.clone()
            prob[self.static_anchor_mask] = 0.0
            prob[self.static_anchor_mask, 0] = 1.0
        return prob

    def hard_labels(self, temperature: float | None = None, *, apply_anchor: bool = False) -> torch.Tensor:
        return self.probabilities(temperature, apply_anchor=apply_anchor).argmax(dim=-1)


def _solve_anchor_embedding(
    prototypes: torch.Tensor,
    *,
    temperature: float,
    target_probability: float,
    max_iterations: int = 200,
) -> torch.Tensor:
    """Place the anchor embedding so that p_i0 reaches the requested probability.

    Moves along the direction from the other prototypes' centroid towards c_0 and
    bisects the displacement, so the result is exact for the actual softmax
    rather than an analytic approximation that ignores the other K-1 classes.
    """
    proto = prototypes.detach()
    target = min(max(float(target_probability), 1.0e-6), 1.0 - 1.0e-9)
    c0 = proto[0]
    if proto.shape[0] < 2:
        return c0.clone()

    direction = c0 - proto[1:].mean(dim=0)
    if float(direction.norm()) < 1.0e-12:
        direction = torch.zeros_like(c0)
        direction[0] = 1.0
    direction = direction / direction.norm().clamp_min(1.0e-12)

    def static_probability(step: float) -> float:
        e = c0 + float(step) * direction
        dist2 = (e[None, :] - proto).square().sum(dim=-1)
        return float(torch.softmax(-dist2 / max(float(temperature), 1.0e-6), dim=-1)[0])

    lo, hi = 0.0, 1.0
    for _ in range(int(max_iterations)):
        if static_probability(hi) >= target:
            break
        hi *= 2.0
        if hi > 1.0e6:
            break
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        if static_probability(mid) < target:
            lo = mid
        else:
            hi = mid
    return c0 + hi * direction
