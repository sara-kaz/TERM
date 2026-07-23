"""
RLConditionedVLA
================
Architecture:
  - Vision:   CLIP ViT-B/32 image encoder  (frozen by default, optional unfreeze)
  - Language: CLIP text encoder             (frozen by default)
  - History:  Learned embeddings for (action_idx, reward) pairs per timestep
  - Fusion:   Causal Transformer (6 layers) over [lang | vision | history...] tokens
  - Head:     Linear classifier → N discrete actions

Inputs per forward pass
-----------------------
  frames      : (B, T, 3, H, W)   — T video frames (current + recent context)
  lang_tokens : (B, 77)            — tokenized language instruction (CLIP tokenizer)
  action_hist : (B, H_len)         — previous action indices   (int64)
  reward_hist : (B, H_len)         — previous scalar rewards   (float32)

Output
------
  logits      : (B, num_actions)   — unnormalized action scores
"""

import torch
import torch.nn as nn
import clip                         # pip install git+https://github.com/openai/CLIP.git


class ActionRewardHistoryEncoder(nn.Module):
    """Encodes a sequence of (action_idx, reward) pairs into token embeddings."""

    def __init__(self, num_actions: int, history_len: int, embed_dim: int):
        super().__init__()
        self.history_len = history_len
        # +1 for a learnable "no previous action" padding token
        self.action_embed = nn.Embedding(num_actions + 1, embed_dim, padding_idx=num_actions)
        self.reward_proj   = nn.Linear(1, embed_dim)
        self.fusion        = nn.Linear(embed_dim * 2, embed_dim)
        self.pos_embed     = nn.Embedding(history_len, embed_dim)
        self.norm          = nn.LayerNorm(embed_dim)

    def forward(self, action_hist: torch.Tensor, reward_hist: torch.Tensor) -> torch.Tensor:
        """
        action_hist : (B, H)  int64 — action indices, use num_actions for padding
        reward_hist : (B, H)  float32
        returns     : (B, H, embed_dim)
        """
        B, H = action_hist.shape
        a_emb = self.action_embed(action_hist)                          # (B, H, D)
        r_emb = self.reward_proj(reward_hist.unsqueeze(-1).float())     # (B, H, D)
        combined = self.fusion(torch.cat([a_emb, r_emb], dim=-1))       # (B, H, D)
        pos = self.pos_embed(torch.arange(H, device=action_hist.device))
        return self.norm(combined + pos)                                 # (B, H, D)


class RLConditionedVLA(nn.Module):
    """
    Vision-Language-Action model conditioned on (action, reward) history.
    Predicts the next discrete action.
    """

    def __init__(
        self,
        num_actions: int,
        history_len: int = 4,
        fusion_layers: int = 6,
        fusion_heads: int = 8,
        dropout: float = 0.1,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.history_len = history_len

        # ── CLIP backbone ────────────────────────────────────────────────
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.clip_model = self.clip_model.float()          # fp32 for training stability
        clip_dim = 512                                     # ViT-B/32 feature dim

        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # ── Projection to a common fusion dimension ──────────────────────
        fusion_dim = 256
        self.vis_proj  = nn.Linear(clip_dim, fusion_dim)
        self.lang_proj = nn.Linear(clip_dim, fusion_dim)

        # ── Action-reward history encoder ────────────────────────────────
        self.history_encoder = ActionRewardHistoryEncoder(num_actions, history_len, fusion_dim)

        # ── Causal fusion transformer ─────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=fusion_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,           # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)

        # ── Learnable [CLS] token for aggregation ────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Action classification head ────────────────────────────────────
        self.action_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_actions),
        )

    # ── Encoding helpers ─────────────────────────────────────────────────

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, 3, H, W)
        returns: (B, T, fusion_dim)
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.clip_model.visual.parameters())):
            feats = self.clip_model.encode_image(flat).float()   # (B*T, 512)
        feats = feats.view(B, T, -1)
        return self.vis_proj(feats)                               # (B, T, fusion_dim)

    def encode_language(self, lang_tokens: torch.Tensor) -> torch.Tensor:
        """
        lang_tokens : (B, 77)
        returns     : (B, 1, fusion_dim)
        """
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.clip_model.parameters())):
            feats = self.clip_model.encode_text(lang_tokens).float()  # (B, 512)
        return self.lang_proj(feats).unsqueeze(1)                     # (B, 1, fusion_dim)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        frames: torch.Tensor,       # (B, T, 3, H, W)
        lang_tokens: torch.Tensor,  # (B, 77)
        action_hist: torch.Tensor,  # (B, H)  int64
        reward_hist: torch.Tensor,  # (B, H)  float32
    ) -> torch.Tensor:              # (B, num_actions)

        B = frames.size(0)

        vis_tokens  = self.encode_frames(frames)                # (B, T, D)
        lang_tokens_ = self.encode_language(lang_tokens)        # (B, 1, D)
        hist_tokens = self.history_encoder(action_hist, reward_hist)  # (B, H, D)

        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, D)

        # Token sequence: [lang | vision_frames | history | CLS]
        # CLS goes LAST so the causal mask lets it attend to all previous tokens.
        # At position 0, CLS can only attend to itself (causal mask blocks everything
        # to its right), so it would never see vision/language/history — that was a bug.
        # At the last position, it can attend to every token before it across all layers.
        sequence = torch.cat([lang_tokens_, vis_tokens, hist_tokens, cls], dim=1)

        # Causal mask: position i may only attend to positions j <= i
        seq_len = sequence.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=sequence.device), diagonal=1
        ).bool()

        out = self.transformer(sequence, mask=causal_mask)  # (B, seq_len, D)
        cls_out = out[:, -1, :]                             # (B, D) — last token = CLS

        return self.action_head(cls_out)                    # (B, num_actions)

    # ── Utility ──────────────────────────────────────────────────────────

    def predict(
        self,
        frames: torch.Tensor,
        lang_tokens: torch.Tensor,
        action_hist: torch.Tensor,
        reward_hist: torch.Tensor,
    ) -> int:
        """Greedy action selection (single sample, no grad)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(frames, lang_tokens, action_hist, reward_hist)
        return int(logits.argmax(dim=-1).item())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
