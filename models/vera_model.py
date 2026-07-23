"""
VERA: Vision-Experience-Reasoning-Action Model
===============================================
Five input streams — three are closed-loop feedback channels:

  Stream 1 — Vision              : T=3 RGB frames → frozen CLIP ViT-B/32 image encoder
  Stream 2 — Instruction         : fixed language goal → frozen CLIP text encoder
  Stream 3a — Action Narration   : prev action → vocabulary → reward gate σ(MLP(r))
                                   → frozen CLIP text encoder  →  E_act token  [NOVEL]
  Stream 3b — Experience         : verbalize_consequence(r, Δd) → frozen CLIP text encoder
                                   → E_exp token  [NEW]  (the "E" in VERA)
  Stream 4  — Action History     : {(action_idx, action_vec, reward)} × H
                                   → 3-signal History Encoder → H tokens

VERA Acronym:
  V = Vision (Stream 1)
  E = Experience / outcome token E_exp (Stream 3b)
  R = Reasoning — the LLaMA fusion backbone that reasons across all streams
  A = Action — dual output head (discrete argmax + continuous regression)

Fusion backbone — LLaMA + ViLT hybrid:
  ┌─ ViLT modality-type embeddings (5 types: instr/CLS=0, action=1, vision=2, history=3, exp=4)
  │   Added to each token BEFORE the main fusion stack.
  └─ LLaMA-style decoder blocks (6 layers, 8 heads, d=256):
      • RMSNorm    instead of LayerNorm   (more stable, fewer params)
      • RoPE       instead of learned PE  (generalises to variable seq lengths)
      • SwiGLU FFN instead of GELU MLP   (better empirical performance)
      • Causal mask — only in Stream-4 TemporalHistoryTransformer; main fusion is bidirectional

Token sequence:
  [ L_instr | E_act | E_exp | V_1 ... V_T | H_1 ... H_H | CLS ]

Dual action head (from CLS):
  Branch 1 — Discrete   : 3-layer MLP → argmax → aₜ
  Branch 2 — Regression : 2-layer MLP + Tanh → v̂ₜ ∈ [-1,1]^d  [NEW]

Ablation flags (configs/config.yaml):
  vera.use_lang_feedback      → False : Ablation A — no language feedback (base VLA)
  vera.use_temporal_history   → False : Ablation B — flat positional history
  vera.use_reward_gate        → False : Ablation C — no reward gate on E_act
  vera.alignment_loss_coef    → 0     : Ablation D — no contrastive alignment loss
  vera.use_consequence_token  → False : Ablation F — action narration only (no E_exp)
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


# ═════════════════════════════════════════════════════════════════════════════
# Part 1: LLaMA-Style Building Blocks
# ═════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).
    Used in LLaMA in place of LayerNorm — more stable, fewer parameters
    (no bias term, no mean-centring step).

    RMSNorm(x) = x / RMS(x) * γ    where RMS(x) = sqrt(mean(x²) + ε)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(d_model))   # learnable γ, no bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) — Su et al., 2021; used in LLaMA.

    RoPE encodes position by rotating Q and K vectors rather than adding
    absolute positional embeddings. Properties:
      • Relative position information is preserved in the attention dot-product.
      • Generalises to sequences longer than seen during training.
      • No learnable parameters — purely functional.

    Applied to queries and keys in every attention head, NOT to values.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        # Precompute inverse frequencies: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache sin/cos tables for efficiency
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t       = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs   = torch.outer(t, self.inv_freq)               # (seq_len, head_dim/2)
        emb     = torch.cat([freqs, freqs], dim=-1)           # (seq_len, head_dim)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])  # (1,1,S,D)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension by splitting in half and negating."""
        d = x.shape[-1] // 2
        return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

    def forward(
        self,
        q: torch.Tensor,   # (B, num_heads, S, head_dim)
        k: torch.Tensor,   # (B, num_heads, S, head_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S = q.size(2)
        if S > self.cos_cache.size(2):
            self._build_cache(S * 2)        # extend cache if needed
        cos = self.cos_cache[:, :, :S, :]
        sin = self.sin_cache[:, :, :S, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network — Shazeer, 2020; used in LLaMA.

    SwiGLU(x) = Swish(W₁x) ⊗ W₂x    (element-wise product of gated branches)
    Swish(x)  = x · σ(x)

    Using two independent linear projections (gate + value) and a hidden
    dimension of (2/3 × 4d) as in LLaMA to keep parameter count comparable
    to a standard 4× FFN.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        # LLaMA convention: hidden_dim = 2/3 of 4×d, rounded to multiple of 64
        if d_ff is None:
            d_ff = int(2 / 3 * 4 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64   # round up to multiple of 64

        self.gate_proj  = nn.Linear(d_model, d_ff, bias=False)   # W₁  (gate)
        self.value_proj = nn.Linear(d_model, d_ff, bias=False)   # W₂  (value)
        self.out_proj   = nn.Linear(d_ff,   d_model, bias=False) # W₃  (output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class LLaMAAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE applied to Q and K.
    Follows LLaMA architecture: no bias in projections, RMSNorm pre-applied.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 max_seq_len: int = 512):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.d_model   = d_model
        self.dropout   = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x:           torch.Tensor,            # (B, S, D)
        attn_mask:   Optional[torch.Tensor],  # (S, S) bool — True = mask out
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        # Project
        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Scaled dot-product attention
        scale  = Dh ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S)

        if attn_mask is not None:
            # attn_mask: True → position is masked (−inf)
            scores = scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))

        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout)

        out = torch.matmul(weights, v)                          # (B, H, S, Dh)
        out = out.transpose(1, 2).contiguous().view(B, S, D)   # (B, S, D)
        return self.o_proj(out)


class LLaMADecoderBlock(nn.Module):
    """
    Single LLaMA-style decoder block:

      x = x + Attention(RMSNorm(x))      ← pre-norm, residual
      x = x + SwiGLU(RMSNorm(x))         ← pre-norm, residual

    Key differences from standard nn.TransformerEncoderLayer:
      • RMSNorm instead of LayerNorm
      • RoPE applied inside attention to Q and K
      • SwiGLU FFN instead of Linear → GELU → Linear
      • No bias terms in projections (LLaMA convention)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: Optional[int] = None,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.norm1    = RMSNorm(d_model)
        self.attn     = LLaMAAttention(d_model, num_heads, dropout=dropout,
                                        max_seq_len=max_seq_len)
        self.norm2    = RMSNorm(d_model)
        self.ffn      = SwiGLU(d_model, d_ff)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class LLaMAFusionTransformer(nn.Module):
    """
    Stack of LLaMA decoder blocks used as the fusion backbone.
    Main VERA fusion uses bidirectional attention (attn_mask=None).
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 d_ff: Optional[int] = None, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.layers   = nn.ModuleList([
            LLaMADecoderBlock(d_model, num_heads, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.final_norm(x)


# ═════════════════════════════════════════════════════════════════════════════
# Part 2: ViLT-Style Modality-Type Embeddings
# ═════════════════════════════════════════════════════════════════════════════

class ViLTModalityEmbedding(nn.Module):
    """
    Learned modality-type embeddings inspired by ViLT (Kim et al., 2021).

    ViLT adds a learnable type embedding to each token based on its modality
    (text vs. image), allowing the Transformer to explicitly distinguish streams.
    We extend this to five modality types for VERA:

      Type 0 — INSTRUCTION     (language goal)
      Type 1 — ACTION_LANG     (language feedback — what action was taken)
      Type 2 — VISION          (video frame patches)
      Type 3 — HISTORY         (numerical action-reward history)
      Type 4 — CONSEQUENCE     (language outcome — what happened as a result) [NEW]

    These are added on top of the projected token features BEFORE the main
    fusion transformer, analogously to BERT's segment embeddings.
    """

    INSTRUCTION = 0
    ACTION_LANG = 1
    VISION      = 2
    HISTORY     = 3
    CONSEQUENCE = 4   # outcome of the previous action — what happened [NEW]

    def __init__(self, d_model: int, num_modalities: int = 5):
        super().__init__()
        self.embed = nn.Embedding(num_modalities, d_model)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        x            : (B, S, d_model)  — projected token sequence
        modality_ids : (S,) int64       — one modality id per position
        returns      : (B, S, d_model)  — tokens with modality type added
        """
        type_emb = self.embed(modality_ids.to(x.device))   # (S, d_model)
        return x + type_emb.unsqueeze(0)                    # broadcast over batch


# ═════════════════════════════════════════════════════════════════════════════
# Part 3: Existing VERA Components
# ═════════════════════════════════════════════════════════════════════════════

# ── Action Vocabulary ─────────────────────────────────────────────────────────

_DEFAULT_VOCAB = {
    0:  "I turned left",
    1:  "I turned right",
    2:  "I moved forward",
    3:  "I moved backward",
    4:  "I attempted to pick up the object",
    5:  "I placed the object down",
    6:  "I opened the door",
    7:  "I stayed in place",
    8:  "I pushed the object forward",
    9:  "I pulled the object toward me",
    10: "I rotated the object clockwise",
    11: "I dropped what I was holding",
    12: "I reached toward the target",
    13: "I grasped the handle",
}


def build_action_vocabulary(num_actions: int, custom_vocab: Optional[Dict] = None) -> Dict:
    """Returns {action_idx: description} for 0 … num_actions (null at num_actions)."""
    if custom_vocab is not None:
        vocab = dict(custom_vocab)
    else:
        vocab = {i: _DEFAULT_VOCAB.get(i, f"I performed action {i}") for i in range(num_actions)}
    vocab[num_actions] = "I have not taken any action yet"
    return vocab


def verbalize_consequence(reward: float, delta_dist: Optional[float] = None) -> str:
    """
    Generate a rich natural-language description of the CONSEQUENCE of an action.

    Encodes what happened as a result of the action — not what the action was.
    The action token says "I moved forward"; this token says "I moved significantly
    closer to the goal and received a high reward."

    Design principle: maximise semantic diversity across the 16 possible strings so
    that CLIP's shared embedding space can clearly separate good from bad outcomes.
    Each string jointly encodes:
      - progress direction  (closer / farther / stationary)
      - progress magnitude  (significantly / slightly / barely)
      - reward quality      (high / moderate / small / none / penalty)

    Args:
        reward    : scalar reward signal received after the action.
        delta_dist: signed change in distance to goal (negative = got closer).
                    None → falls back to reward-only description (SFT phase).

    Returns a short past-tense sentence suitable for CLIP encoding.
    """
    # ── Reward magnitude bucket ────────────────────────────────────────────────
    if reward >= 1.0:
        rew_tag = "high"
    elif reward >= 0.5:
        rew_tag = "moderate"
    elif reward > 0.05:
        rew_tag = "small"
    elif reward >= -0.05:
        rew_tag = "no"
    else:
        rew_tag = "negative"

    # ── Full description: direction × magnitude × reward ──────────────────────
    if delta_dist is not None:
        abs_d = abs(delta_dist)
        mag   = "significantly" if abs_d > 0.15 else ("slightly" if abs_d > 0.05 else "barely")

        if delta_dist < -0.05:                      # moved CLOSER
            if rew_tag in ("high", "moderate"):
                return f"I moved {mag} closer to the goal and received a {rew_tag} reward"
            elif rew_tag == "small":
                return f"I moved {mag} closer to the goal with only a small reward"
            elif rew_tag == "no":
                return f"I moved {mag} closer to the goal but received no reward"
            else:
                return f"I moved {mag} closer to the goal yet received a penalty"

        elif delta_dist > 0.05:                     # moved FARTHER
            if rew_tag == "negative":
                return f"I moved {mag} farther from the goal and received a penalty"
            elif rew_tag == "no":
                return f"I moved {mag} farther from the goal with no reward"
            elif rew_tag in ("small", "moderate"):
                return f"I moved {mag} farther from the goal despite receiving a reward"
            else:
                return f"I moved farther from the goal unexpectedly"

        else:                                        # STATIONARY
            if rew_tag in ("high", "moderate"):
                return "My position barely changed but I received a high reward"
            elif rew_tag == "small":
                return "My position barely changed with only a small positive reward"
            elif rew_tag == "negative":
                return "I remained near the same position and received a penalty"
            else:
                return "I made no progress and received no reward"

    # ── Fallback: reward-only (used during SFT when delta_dist unavailable) ───
    if rew_tag == "high":
        return "I completed the task successfully and received a high reward"
    elif rew_tag == "moderate":
        return "I made good progress toward the goal and received a reward"
    elif rew_tag == "small":
        return "I made partial progress and received a small positive reward"
    elif rew_tag == "no":
        return "I made no clear progress and received no reward"
    else:
        return "The action moved me away from the goal and I received a penalty"


# ── Reward-Gated Action Language Feedback Encoder ────────────────────────────

class ActionLanguageFeedbackEncoder(nn.Module):
    """
    Feedback Channel A (Semantic): encodes the previous action as natural language
    using the frozen CLIP text encoder, then gates the signal by the reward received.

      prev_action_idx ─► vocabulary lookup ─► CLIP text (frozen)
                      ─► separate Linear(512 → d_model) + RMSNorm
                      ─► reward gate σ(MLP(r)) ∈ (0,1)
                      ─► (B, 1, d_model) token

    A separate projection from the instruction projection allows the two language
    streams to specialise independently inside the d_model space.
    """

    def __init__(
        self,
        num_actions:     int,
        clip_model:      nn.Module,
        d_model:         int   = 256,
        clip_dim:        int   = 512,
        dropout:         float = 0.1,
        use_reward_gate: bool  = True,
        action_vocab:    Optional[Dict] = None,
    ):
        super().__init__()
        self.num_actions    = num_actions
        self.use_reward_gate = use_reward_gate

        vocab = action_vocab or build_action_vocabulary(num_actions)
        # Index num_actions is the padding/"no previous action" token used when
        # history is empty at the start of an episode. Auto-add if missing.
        if num_actions not in vocab:
            vocab = {**vocab, num_actions: 'no previous action taken'}
        all_texts = [vocab[i] for i in range(num_actions + 1)]
        self.register_buffer("action_tokens", clip.tokenize(all_texts))

        self.clip_model = clip_model

        self.proj = nn.Sequential(
            nn.Linear(clip_dim, d_model, bias=False),
            RMSNorm(d_model),
        )

        if use_reward_gate:
            self.reward_gate = nn.Sequential(
                nn.Linear(1, 64), nn.SiLU(),
                nn.Linear(64, 32), nn.SiLU(),
                nn.Linear(32, 1), nn.Sigmoid(),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, prev_action_idx: torch.Tensor, prev_reward: torch.Tensor):
        tokens = self.action_tokens[prev_action_idx.clamp(0, self.num_actions)]
        with torch.no_grad():
            raw_emb = self.clip_model.encode_text(tokens).float()   # (B, 512)
        proj = self.proj(raw_emb)                                    # (B, d_model)
        if self.use_reward_gate:
            gate = self.reward_gate(prev_reward.unsqueeze(-1).float())
            proj = gate * proj
        proj = self.dropout(proj)
        return proj.unsqueeze(1), raw_emb                            # (B,1,D), (B,512)


# ── Consequence Language Encoder ──────────────────────────────────────────────

class ConsequenceLanguageEncoder(nn.Module):
    """
    Encodes the CONSEQUENCE of the previous action as dynamic natural language.

    Unlike ActionLanguageFeedbackEncoder (which pre-tokenizes a fixed vocab),
    consequence strings are generated at runtime from reward + optional
    state_delta, so tokenization happens per forward pass. This is necessary
    because the consequence depends on the observed outcome, not just the action.

    The encoder uses the SAME frozen CLIP text encoder as the action and
    instruction streams, placing all three in a shared 512-dim semantic space.
    The fusion transformer can then directly compare:
      • instruction embedding  ("pick up the red cube")
      • action embedding       ("I attempted to pick up the object")
      • consequence embedding  ("The action was successful and I received a positive reward")

    A separate projection head (independent of the action projection) allows
    the consequence stream to specialise independently in d_model space.

    Forward:
      prev_reward  (B,) + state_delta (B,) opt → verbalize_consequence()
      → clip.tokenize (runtime) → frozen CLIP encode_text
      → Linear(512 → d_model) + RMSNorm → (B, 1, d_model)
    """

    def __init__(
        self,
        clip_model: nn.Module,
        d_model:    int   = 256,
        clip_dim:   int   = 512,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.clip_model = clip_model

        # Independent projection — does NOT share weights with action/instruction proj
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, d_model, bias=False),
            RMSNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        prev_reward:  torch.Tensor,                   # (B,) float
        state_delta:  Optional[torch.Tensor] = None,  # (B,) float or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            token   : (B, 1, d_model) — projected consequence token
            raw_emb : (B, 512)        — raw CLIP embedding (for alignment loss)
        """
        B = prev_reward.size(0)

        # Build one consequence string per sample (Python loop — B is small)
        descriptions = []
        for i in range(B):
            r = prev_reward[i].item()
            d = state_delta[i].item() if state_delta is not None else None
            descriptions.append(verbalize_consequence(r, d))

        # Tokenize at runtime (strings are dynamic, so cannot be pre-cached)
        tokens = clip.tokenize(descriptions).to(prev_reward.device)  # (B, 77)

        with torch.no_grad():
            raw_emb = self.clip_model.encode_text(tokens).float()    # (B, 512)

        proj = self.proj(raw_emb)      # (B, d_model)
        proj = self.dropout(proj)
        return proj.unsqueeze(1), raw_emb                            # (B,1,D), (B,512)


# ── Cross-Alignment Module ────────────────────────────────────────────────────

class CrossAlignmentModule(nn.Module):
    """
    Reward-weighted InfoNCE contrastive alignment loss.

    Teaches the model that successful actions (high reward) should have their
    language description semantically close to the task instruction in CLIP space.

    score(instr, action_lang) = cosine_similarity   ∈ [-1, 1]
    loss = reward-weighted symmetric InfoNCE across the batch
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def score(self, instr_emb: torch.Tensor, action_lang_emb: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(instr_emb, action_lang_emb, dim=-1)

    def contrastive_loss(self, instr_emb, action_lang_emb, rewards) -> torch.Tensor:
        B = instr_emb.size(0)
        if B < 2:
            return torch.tensor(0.0, device=instr_emb.device)
        instr_n  = F.normalize(instr_emb,       dim=-1)
        action_n = F.normalize(action_lang_emb, dim=-1)
        sim      = torch.mm(instr_n, action_n.T) / self.temperature    # (B, B)
        labels   = torch.arange(B, device=instr_emb.device)
        loss_sym = (F.cross_entropy(sim,   labels, reduction="none")
                  + F.cross_entropy(sim.T, labels, reduction="none")) * 0.5
        weights  = F.softplus(rewards.float())
        weights  = weights / (weights.sum() + 1e-8)
        return (weights * loss_sym).sum()


# ── Step-conditioned lang-token modulation (inspired by VLAConf §IV-C) ───────

class StepProgressEncoder(nn.Module):
    """Rollout-phase descriptor ψ_k = [embed(k̂); k̄] with fixed horizon K."""

    def __init__(self, step_horizon: int, d_cond: int = 64):
        super().__init__()
        self.step_horizon = max(2, int(step_horizon))
        self.embed = nn.Embedding(self.step_horizon, d_cond)
        self.proj = nn.Sequential(
            nn.Linear(d_cond + 1, d_cond, bias=False),
            nn.SiLU(),
        )

    def forward(self, step_idx: torch.Tensor) -> torch.Tensor:
        k_hat = step_idx.long().clamp(0, self.step_horizon - 1)
        k_bar = k_hat.float() / (self.step_horizon - 1)
        e = self.embed(k_hat)
        return self.proj(torch.cat([e, k_bar.unsqueeze(-1)], dim=-1))


class LangTokenFiLM(nn.Module):
    """FiLM + sigmoid residual gate on a (B,1,D) language-feedback token."""

    def __init__(self, d_cond: int, d_model: int):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(d_cond, d_cond), nn.SiLU())
        self.film = nn.Linear(d_cond, d_model * 2)
        self.gate = nn.Linear(d_cond, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, token: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        c = self.cond(psi)
        gamma, beta = self.film(c).chunk(2, dim=-1)
        mod = self.norm((1 + gamma).unsqueeze(1) * token + beta.unsqueeze(1))
        eta = torch.sigmoid(self.gate(c)).unsqueeze(1)
        return eta * mod + (1 - eta) * token


# ── Temporal History Transformer ──────────────────────────────────────────────

class TemporalHistoryTransformer(nn.Module):
    """
    Small 2-layer LLaMA decoder applied only to the numerical history tokens.
    Detects temporal patterns across (action, reward) pairs before they enter
    the main fusion stack.
    """

    def __init__(self, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tf = LLaMAFusionTransformer(
            d_model=d_model, num_heads=nhead, num_layers=num_layers,
            dropout=dropout, max_seq_len=64,
        )

    def forward(self, history_tokens: torch.Tensor) -> torch.Tensor:
        H = history_tokens.size(1)
        causal = torch.triu(
            torch.ones(H, H, device=history_tokens.device), diagonal=1
        ).bool()
        return self.tf(history_tokens, attn_mask=causal)


# ── Enhanced Numerical History Encoder ────────────────────────────────────────

class ActionRewardHistoryEncoder(nn.Module):
    """
    Stream 4 — Low-Level Action + Reward History Encoder.

    Encodes the last H timesteps, each represented by THREE signals:

      1. Discrete action index  → Embedding lookup               (B, H) int
         Captures which action category was selected.

      2. Low-level action vector → Linear projection             (B, H, action_dim) float
         The actual continuous robot command executed at that step:
           MetaWorld : [Δx, Δy, Δz, gripper]  (4-DoF, from codebook)
           BabyAI    : one-hot of 7 primitives projected to d_model
           Language-Table / CALVIN : full 6-7 DoF joint-space delta
         This is the key upgrade over a pure discrete-index approach:
         two different indices might produce similar movements; the vector
         carries magnitude and direction the index alone cannot.

      3. Scalar reward → Linear projection                       (B, H) float
         Indicates outcome quality of that timestep.

    Fusion:
      discrete_emb ⊕ (gate · lowlevel_emb) ⊕ reward_emb
                         ↑
      gate = σ(MLP(reward)) ∈ (0,1) — reward-magnitude gating.
      Scales the low-level vector contribution by how well that step went,
      consistent with the reward gate used in Stream 3a.

    Then: sinusoidal positional encoding (oldest→newest, left→right)
          → optional 2-layer TemporalHistoryTransformer
          → (B, H, d_model) ready for the main LLaMA fusion stack.

    Inspired by action-history encoding in:
      π₀.5 (continuous action chunk tokens, Black et al. 2024)
      Language-Table (Lynch et al. 2023) — 6-DoF end-effector deltas
      CALVIN (Mees et al. 2022)          — 7-DoF joint-space actions

    Args:
        num_actions  : size of discrete action vocabulary
        action_dim   : dimensionality of the low-level continuous action vector
                       (4 for MetaWorld xyz+gripper; 7 for CALVIN; 6 for Language-Table)
        history_len  : number of past timesteps H to keep
        d_model      : fusion embedding dimension (must match rest of VERA)
    """

    def __init__(self, num_actions: int, history_len: int, d_model: int = 256,
                 action_dim: int = 4, dropout: float = 0.1,
                 use_temporal_transformer: bool = True):
        super().__init__()
        self.action_dim = action_dim

        # ── Signal 1: discrete action index ──────────────────────────────────
        # padding_idx = num_actions → used as the "null action" at t=0
        self.action_embed = nn.Embedding(num_actions + 1, d_model, padding_idx=num_actions)

        # ── Signal 2: low-level continuous action vector ──────────────────────
        # Projects the raw robot command (e.g. [Δx, Δy, Δz, gripper]) to d_model.
        # RMSNorm stabilises the projection — continuous vectors can have
        # large magnitude variance across datasets (Language-Table vs CALVIN).
        self.lowlevel_proj = nn.Sequential(
            nn.Linear(action_dim, d_model, bias=False),
            RMSNorm(d_model),
        )

        # Reward-magnitude gate: scales low-level contribution by step quality.
        # σ(MLP(r)) ∈ (0,1) — mirrors the reward gate in Stream 3a.
        self.lowlevel_gate = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

        # ── Signal 3: scalar reward ───────────────────────────────────────────
        self.reward_proj  = nn.Sequential(nn.Linear(1, d_model, bias=False), RMSNorm(d_model))

        # ── Fusion: concatenate all three, project back to d_model ────────────
        # Input dim = d_model (discrete) + d_model (lowlevel) + d_model (reward)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model, bias=False),
            RMSNorm(d_model),
            nn.SiLU(),
        )

        # ── Sinusoidal positional encoding (oldest→newest, left→right) ────────
        self.register_buffer("pos_enc", self._sinusoidal_pe(history_len, d_model))

        # ── Optional temporal transformer ─────────────────────────────────────
        self.use_temporal = use_temporal_transformer
        if use_temporal_transformer:
            self.temporal_tf = TemporalHistoryTransformer(d_model=d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(length, d_model)
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, H, d_model)

    def forward(
        self,
        action_hist:     torch.Tensor,            # (B, H) int   — discrete indices
        reward_hist:     torch.Tensor,            # (B, H) float — scalar rewards
        action_vec_hist: Optional[torch.Tensor],  # (B, H, action_dim) float | None
    ) -> torch.Tensor:
        """
        Args:
            action_hist     : (B, H) discrete action indices from t-H to t-1.
            reward_hist     : (B, H) scalar rewards from t-H to t-1.
            action_vec_hist : (B, H, action_dim) continuous action vectors.
                              If None (e.g. dummy env or first H steps), falls
                              back to zeros — encoder degrades gracefully to the
                              discrete-only baseline.
        Returns:
            (B, H, d_model) history token sequence, ready for the fusion stack.
        """
        B, H = action_hist.shape

        # Signal 1: discrete embedding
        discrete_emb = self.action_embed(action_hist)                     # (B, H, D)

        # Signal 2: low-level continuous action vector
        if action_vec_hist is not None:
            lowlevel_emb = self.lowlevel_proj(action_vec_hist.float())    # (B, H, D)
        else:
            # Graceful fallback: no continuous data available (dummy env / t<H)
            lowlevel_emb = torch.zeros_like(discrete_emb)

        # Gate: scale low-level by reward magnitude — good steps contribute more
        gate         = self.lowlevel_gate(reward_hist.unsqueeze(-1).float())  # (B, H, 1)
        lowlevel_emb = gate * lowlevel_emb                                     # (B, H, D)

        # Signal 3: reward embedding
        reward_emb   = self.reward_proj(reward_hist.unsqueeze(-1).float())    # (B, H, D)

        # Fuse all three signals
        tokens = self.fusion(
            torch.cat([discrete_emb, lowlevel_emb, reward_emb], dim=-1)       # (B, H, 3D)
        )                                                                       # (B, H, D)

        # Add sinusoidal positional encoding
        tokens = tokens + self.pos_enc[:, :H]
        tokens = self.dropout(tokens)

        # Optional temporal self-attention over the H history steps
        if self.use_temporal:
            tokens = self.temporal_tf(tokens)

        return tokens


# ═════════════════════════════════════════════════════════════════════════════
# Part 4: Main VERA Model
# ═════════════════════════════════════════════════════════════════════════════

class VERAModel(nn.Module):
    """
    Vision-Experience-Reasoning-Action (VERA) Model.

    Fusion backbone: ViLT modality embeddings + LLaMA decoder stack.

    ViLT contribution: explicit modality-type tokens added before the fusion
    transformer, giving each LLaMA block full awareness of which stream each
    token came from — instruction, action language, vision, or history.

    LLaMA contribution: RoPE encodes relative position in the token sequence,
    RMSNorm stabilises deep pre-norm blocks, and SwiGLU outperforms GELU FFNs
    empirically across language and multimodal settings.

    Token sequence:
      [ L_instr(type=0) | L_action(type=1) | V_1..V_T(type=2) | H_1..H_H(type=3) | CLS ]

    forward() returns a dict (not a bare tensor):
      logits            : (B, num_actions)         — discrete action logits
      action_vec        : (B, action_dim)          — continuous action prediction ∈ (-1,1)
      cls_features      : (B, d_model)
      alignment_score   : (B,) cosine(instr, experience)
      consequence_score : (B,) cosine(instr, reasoning)
      instr_emb         : (B, 512) raw CLIP
      action_lang_emb   : (B, 512) raw CLIP
      consequence_emb   : (B, 512) raw CLIP
    """

    def __init__(
        self,
        num_actions:          int,
        history_len:          int   = 4,
        num_vis_frames:       int   = 3,
        fusion_layers:        int   = 6,
        fusion_heads:         int   = 8,
        d_model:              int   = 256,
        d_ff_scale:           int   = 4,
        dropout:              float = 0.1,
        freeze_clip:          bool  = True,
        unfreeze_clip_vision: bool  = False,   # fine-tune ViT vision encoder (keep text frozen)
        use_lang_feedback:    bool  = True,   # legacy master switch (→ use_action_lang_feedback)
        use_action_lang_feedback: Optional[bool] = None,  # Stream 3a; None = use_lang_feedback
        use_temporal_history: bool  = True,
        use_reward_gate:      bool  = True,
        use_consequence_token: bool = True,   # Stream 3b (independent of 3a)
        action_vocab:         Optional[Dict] = None,
        action_dim:           int   = 4,      # Stream 4: low-level action vector dim
                                              # MetaWorld=4 (Δx,Δy,Δz,gripper)
                                              # CALVIN=7   (joint-space deltas)
                                              # Language-Table=6 (end-effector deltas)
        proprio_dim:          int   = 0,      # 0=disabled; CALVIN robot_obs=15
        chunk_size:           int   = 1,      # K-step action chunking (1 = disabled)
        use_step_conditioning: bool = False, # VLAConf-style phase FiLM on lang tokens
        step_horizon:         int   = 96,     # fixed K for step embed + progress scalar
    ):
        super().__init__()
        self.num_actions       = num_actions
        self.history_len       = history_len
        self.num_vis_frames    = num_vis_frames
        self.d_model           = d_model
        self.chunk_size        = max(1, int(chunk_size))
        self.use_step_conditioning = bool(use_step_conditioning)
        self.step_horizon      = max(2, int(step_horizon))
        # Stream 3a / 3b are independently ablatable (no_act ≠ no_lang ≠ bc).
        self.use_action_lang_feedback = (
            use_lang_feedback if use_action_lang_feedback is None
            else bool(use_action_lang_feedback)
        )
        self.use_consequence_token = bool(use_consequence_token)
        self.use_lang_feedback = self.use_action_lang_feedback  # backward compat
        clip_dim                  = 512

        # ── CLIP backbone (shared for all three language/vision streams) ──────
        self.clip_model, _ = clip.load("ViT-B/32")
        self.clip_model    = self.clip_model.float()
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Optionally re-enable CLIP vision encoder for task-specific fine-tuning.
        # Keeps text encoder frozen (language priors are preserved), only allows
        # the ViT image encoder to adapt to the target visual domain (e.g. the
        # overhead synthetic images of Language-Table / CALVIN).
        # Use a small LR (e.g. 1e-5) for these params via the trainer param groups.
        if unfreeze_clip_vision:
            for p in self.clip_model.visual.parameters():
                p.requires_grad = True

        # ── Projections: CLIP → d_model ───────────────────────────────────────
        # Using RMSNorm in place of LayerNorm for consistency with LLaMA backbone
        self.vis_proj  = nn.Sequential(nn.Linear(clip_dim, d_model, bias=False), RMSNorm(d_model))
        self.lang_proj = nn.Sequential(nn.Linear(clip_dim, d_model, bias=False), RMSNorm(d_model))

        # ── Stream 3a: Action Language Feedback Encoder ───────────────────────
        self.action_lang_encoder = None
        self.alignment_module = None
        if self.use_action_lang_feedback:
            self.action_lang_encoder = ActionLanguageFeedbackEncoder(
                num_actions=num_actions, clip_model=self.clip_model,
                d_model=d_model, clip_dim=clip_dim, dropout=dropout,
                use_reward_gate=use_reward_gate, action_vocab=action_vocab,
            )
        if self.use_action_lang_feedback or self.use_consequence_token:
            self.alignment_module = CrossAlignmentModule(temperature=0.07)

        # ── Stream 3b: Consequence Language Encoder ──────────────────────────
        self.consequence_encoder = None
        if self.use_consequence_token:
            self.consequence_encoder = ConsequenceLanguageEncoder(
                clip_model=self.clip_model,
                d_model=d_model, clip_dim=clip_dim, dropout=dropout,
            )

        # ── Step-conditioned FiLM on feedback tokens (VLAConf §IV-C) ─────────
        _d_cond = 64
        self.step_encoder = None
        self.act_film = None
        self.conseq_film = None
        if self.use_step_conditioning:
            self.step_encoder = StepProgressEncoder(self.step_horizon, d_cond=_d_cond)
            if self.use_action_lang_feedback:
                self.act_film = LangTokenFiLM(_d_cond, d_model)
            if self.use_consequence_token:
                self.conseq_film = LangTokenFiLM(_d_cond, d_model)

        # ── Stream 4: Low-Level Action + Reward History Encoder ───────────────
        # Now encodes three signals per history step:
        #   (i)  discrete action index  (ii) continuous action vector  (iii) reward
        # action_dim matches the robot's DoF: 4=MetaWorld, 7=CALVIN, 6=Language-Table
        self.action_dim = action_dim
        self.proprio_dim = int(proprio_dim)
        self.use_proprio = self.proprio_dim > 0
        if self.use_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(self.proprio_dim, d_model, bias=False),
                RMSNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
            )

        self.history_encoder = ActionRewardHistoryEncoder(
            num_actions=num_actions, history_len=history_len, d_model=d_model,
            action_dim=action_dim, dropout=dropout,
            use_temporal_transformer=use_temporal_history,
        )

        # ── ViLT Modality-Type Embeddings [Kim et al., 2021] ─────────────────
        # modality types: INSTRUCTION, ACTION_LANG, VISION, HISTORY, CONSEQUENCE
        _num_mod = 5 if self.use_consequence_token else 4
        self.modality_embed = ViLTModalityEmbedding(d_model, num_modalities=_num_mod)

        # ── LLaMA Fusion Transformer [Touvron et al., 2023] ──────────────────
        d_ff = d_model * d_ff_scale
        self.fusion_transformer = LLaMAFusionTransformer(
            d_model=d_model, num_heads=fusion_heads, num_layers=fusion_layers,
            d_ff=d_ff, dropout=dropout, max_seq_len=512,
        )

        # ── Learnable [CLS] aggregation token (placed LAST) ──────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Discrete action classification head ────────────────────────────────
        # Legacy layout (matches CALVIN core6 checkpoints): 256 → 512 → 256 → A
        self.action_bin_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            RMSNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_actions, bias=False),
        )

        # Optional K-step chunk head (π0-style); old checkpoints omit this safely.
        if self.chunk_size > 1:
            self.action_chunk_head = nn.Linear(
                d_model, self.chunk_size * num_actions, bias=False
            )
        else:
            self.action_chunk_head = None

        # ── Continuous action regression head ──────────────────────────────────
        # Runs in parallel with the classifier on the same CLS token.
        # Predicts the raw continuous robot command; trained with MSE against the
        # expert's executed action vector wherever datasets supply one:
        #   CALVIN         → 7-DoF [x,y,z,roll,pitch,yaw,gripper]
        #   Language-Table → 2-DoF [Δx, Δy]
        #   MetaWorld      → 4-DoF [Δx, Δy, Δz, gripper]
        # Tanh clips output to (-1, 1) matching standard robot action normalisation.
        # Falls back to an auxiliary signal only (no gradient) when target_vec is
        # unavailable (pure discrete datasets like BabyAI).
        self.action_vec_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, action_dim, bias=False),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def _clip_grad(self):
        return any(p.requires_grad for p in self.clip_model.parameters())

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        with torch.set_grad_enabled(self._clip_grad()):
            feats = self.clip_model.encode_image(flat).float()
        return self.vis_proj(feats.view(B, T, -1))   # (B, T, d_model)

    def encode_instruction(self, lang_tokens: torch.Tensor):
        with torch.set_grad_enabled(self._clip_grad()):
            raw_emb = self.clip_model.encode_text(lang_tokens).float()
        token = self.lang_proj(raw_emb).unsqueeze(1)              # (B, 1, d_model)
        return token, raw_emb

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        frames:          torch.Tensor,
        lang_tokens:     torch.Tensor,
        action_hist:     torch.Tensor,                    # (B, H) int   — discrete indices
        reward_hist:     torch.Tensor,                    # (B, H) float — scalar rewards
        prev_action_idx: Optional[torch.Tensor] = None,
        prev_reward:     Optional[torch.Tensor] = None,
        state_delta:     Optional[torch.Tensor] = None,  # (B,) signed dist-to-goal change
        action_vec_hist: Optional[torch.Tensor] = None,  # (B, H, action_dim) low-level vectors
                                                          # MetaWorld: [Δx,Δy,Δz,gripper]
                                                          # CALVIN:    7-DoF joint deltas
                                                          # Language-Table: 6-DoF EE deltas
                                                          # None → graceful fallback to zeros
        robot_obs:       Optional[torch.Tensor] = None,  # (B, proprio_dim) CALVIN state
        step_idx:        Optional[torch.Tensor] = None,  # (B,) rollout step t (0-based)
    ) -> Dict[str, Optional[torch.Tensor]]:

        B = frames.size(0)

        psi = None
        if self.use_step_conditioning and self.step_encoder is not None:
            if step_idx is None:
                step_idx = torch.zeros(B, dtype=torch.long, device=frames.device)
            psi = self.step_encoder(step_idx)

        # 1. Encode all streams
        vis_tokens              = self.encode_frames(frames)          # (B, T, D)
        lang_token,  instr_emb = self.encode_instruction(lang_tokens) # (B,1,D), (B,512)
        hist_tokens             = self.history_encoder(                # (B, H, D)
            action_hist, reward_hist, action_vec_hist
        )

        # 2. Action language feedback — Stream 3a (what was done)
        action_lang_token = action_lang_emb = alignment_score = None
        if self.use_action_lang_feedback and self.action_lang_encoder is not None:
            _pa = prev_action_idx if prev_action_idx is not None else action_hist[:, -1]
            _pr = prev_reward     if prev_reward     is not None else reward_hist[:, -1]
            action_lang_token, action_lang_emb = self.action_lang_encoder(_pa, _pr)
            if psi is not None and self.act_film is not None:
                action_lang_token = self.act_film(action_lang_token, psi)
            if self.alignment_module is not None:
                alignment_score = self.alignment_module.score(
                    lang_token.squeeze(1), action_lang_token.squeeze(1)
                )

        # 2b. Consequence language feedback — Stream 3b (what happened)
        consequence_token = consequence_emb = consequence_score = None
        if self.use_consequence_token and self.consequence_encoder is not None:
            _pr = prev_reward if prev_reward is not None else reward_hist[:, -1]
            consequence_token, consequence_emb = self.consequence_encoder(_pr, state_delta)
            if psi is not None and self.conseq_film is not None:
                consequence_token = self.conseq_film(consequence_token, psi)
            if self.alignment_module is not None:
                consequence_score = self.alignment_module.score(
                    lang_token.squeeze(1), consequence_token.squeeze(1)
                )

        # 3. CLS token at the end
        cls = self.cls_token.expand(B, -1, -1)                       # (B, 1, D)

        # 4. Assemble token sequence and modality ID vector
        #    [ L_instr | L_action | L_consequence | V_1...V_T | H_1...H_H | CLS ]
        T = vis_tokens.size(1)
        H = hist_tokens.size(1)

        parts      = [lang_token]
        mod_ids    = [torch.full((1,),  ViLTModalityEmbedding.INSTRUCTION, dtype=torch.long)]

        if action_lang_token is not None:
            parts.append(action_lang_token)
            mod_ids.append(torch.full((1,), ViLTModalityEmbedding.ACTION_LANG, dtype=torch.long))

        # Consequence token follows immediately after action token [NEW]
        # The CLS can now attend to: what goal? → what I did? → what happened? → what I saw?
        if consequence_token is not None:
            parts.append(consequence_token)
            mod_ids.append(torch.full((1,), ViLTModalityEmbedding.CONSEQUENCE, dtype=torch.long))

        parts.append(vis_tokens)
        mod_ids.append(torch.full((T,), ViLTModalityEmbedding.VISION, dtype=torch.long))

        parts.append(hist_tokens)
        mod_ids.append(torch.full((H,), ViLTModalityEmbedding.HISTORY, dtype=torch.long))

        if self.use_proprio:
            if robot_obs is None:
                robot_obs = torch.zeros(B, self.proprio_dim, device=frames.device)
            proprio_tok = self.proprio_encoder(robot_obs.float()).unsqueeze(1)
            parts.append(proprio_tok)
            mod_ids.append(
                torch.full((1,), ViLTModalityEmbedding.INSTRUCTION, dtype=torch.long)
            )

        # CLS has no dedicated modality type — reuse INSTRUCTION (attends to all)
        parts.append(cls)
        mod_ids.append(torch.full((1,), ViLTModalityEmbedding.INSTRUCTION, dtype=torch.long))

        sequence  = torch.cat(parts, dim=1)                          # (B, S, D)
        mod_ids_t = torch.cat(mod_ids, dim=0).to(sequence.device)   # (S,)

        # 5. Add ViLT modality-type embeddings
        sequence = self.modality_embed(sequence, mod_ids_t)

        # 6. Bidirectional (full) attention — NO causal mask in the main fusion stack.
        # BC policies are not autoregressive token generators; E_act/E_exp must attend
        # to vision + history (see docs/vera_corl.tex §LLaMA Fusion Transformer).
        # Stream-4 history keeps its own causal mask inside TemporalHistoryTransformer.

        # 7. LLaMA fusion transformer
        out          = self.fusion_transformer(sequence, attn_mask=None)
        cls_features = out[:, -1, :]                                 # (B, D)

        # 8. Discrete action logits
        logits = self.action_bin_head(cls_features)                   # (B, A)
        logits_chunk = None
        if self.action_chunk_head is not None:
            logits_chunk = self.action_chunk_head(cls_features).view(
                B, self.chunk_size, self.num_actions
            )

        # 9. Continuous action regression (parallel head, same CLS features)
        action_vec = self.action_vec_head(cls_features)               # (B, action_dim)

        out = {
            "logits":             logits,
            "action_vec":         action_vec,
            "cls_features":       cls_features,
            "alignment_score":    alignment_score,
            "consequence_score":  consequence_score,
            "instr_emb":          instr_emb,
            "action_lang_emb":    action_lang_emb,
            "consequence_emb":    consequence_emb,
            # Projected fusion tokens — used for trainable alignment loss
            "instr_token":        lang_token,
            "action_lang_token":  action_lang_token,
            "consequence_token":  consequence_token,
        }
        if logits_chunk is not None:
            out["logits_chunk"] = logits_chunk
        return out

    def compute_alignment_loss(
        self,
        instr_token:       torch.Tensor,
        action_lang_token: Optional[torch.Tensor],
        rewards:           torch.Tensor,
        consequence_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dual contrastive alignment on **projected d_model fusion tokens** (trainable).
        Using raw frozen CLIP embeddings here is a no-op — only log_temp gets gradients.
        """
        if self.alignment_module is None:
            return torch.tensor(0.0, device=instr_token.device)

        def _sqz(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.squeeze(1) if t.ndim == 3 else t

        instr_t = _sqz(instr_token)
        loss = torch.tensor(0.0, device=instr_t.device)
        n    = 0

        act_t = _sqz(action_lang_token)
        if act_t is not None and self.use_action_lang_feedback:
            loss = loss + self.alignment_module.contrastive_loss(instr_t, act_t, rewards)
            n += 1

        cons_t = _sqz(consequence_token)
        if cons_t is not None and self.use_consequence_token:
            loss = loss + self.alignment_module.contrastive_loss(instr_t, cons_t, rewards)
            n += 1

        return loss / max(n, 1)

    def predict(self, frames, lang_tokens, action_hist, reward_hist,
                prev_action_idx=None, prev_reward=None,
                state_delta=None, action_vec_hist=None, robot_obs=None,
                step_idx=None) -> int:
        """
        Greedy action selection.  All optional arguments mirror forward().
        Pass state_delta during RL / eval so Stream 3b (consequence token)
        uses the full spatial+reward verbalization rather than reward-only.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(frames, lang_tokens, action_hist, reward_hist,
                               prev_action_idx, prev_reward,
                               state_delta=state_delta,
                               action_vec_hist=action_vec_hist,
                               robot_obs=robot_obs,
                               step_idx=step_idx)
        return int(out["logits"].argmax(-1).item())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = self.num_trainable_params()
        return (f"Total params: {total:,}  |  "
                f"Trainable: {trainable:,}  |  "
                f"Frozen (CLIP): {total - trainable:,}")
