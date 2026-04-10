"""
PSTT (Power-Set Tokenizer Transformer) Vision Bridge Module

SigLIP と Gemma デコーダの間に挿入するアダプターモジュール。
「パッチ単位の処理」を「オブジェクト（集合）単位の論理処理」へ昇格させる。

Tensor Shape Transitions (1792×896 入力 → 512 視覚トークン時):
  X           : [B, 512, D=1152]   # SigLIP 出力
  ↓ Step 1 : Pair-wise Scanning & STE Top-K
  X_top10     : [B,  10, D]        # 上位10シードトークン
  ↓ Step 2 : Power-Set Generation & Attention Pooling
  G           : [B, 1023, D]       # 2^10-1 グループトークン
  ↓ Step 3 : Logical Scoring & Gated Filtering
  G_refined   : [B,  20, D]        # 論理完結度上位20グループ
  ↓ Step 4 : Gated Cross-Refinement
  X_final     : [B, 512, D]        # 文脈強化済み視覚トークン
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# Core PSTT Module
# ─────────────────────────────────────────────────────────────────────────────

class PSTTModule(nn.Module):
    """
    PSTT Vision Bridge: SigLIP出力をオブジェクト論理表現で強化する。

    Args:
        d_model   : SigLIP 隠れ次元 (gemma-3-4b-it の SigLIP-400M では 1152)
        n_heads   : Multi-head Attention のヘッド数
        top_k     : Step1で選ぶシードトークン数 (べき集合サイズを決める)
        top_m     : Step3で選ぶ上位グループ数
        use_pos   : パッチ間相対位置情報を連結するか (実験的)
    """

    def __init__(
        self,
        d_model: int = 1152,
        n_heads: int = 8,
        top_k: int = 10,
        top_m: int = 20,
        use_pos: bool = False,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.top_k    = top_k
        self.top_m    = top_m
        self.use_pos  = use_pos
        self.n_subsets = 2 ** top_k - 1  # 1023

        # ── Step 1: Pair-wise Scanning ────────────────────────────────
        # 全トークン間のアテンションで文脈を統合してからスコアリング
        self.pair_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.pair_ln   = nn.LayerNorm(d_model)
        # Objectness スコア: [B, N, D] → [B, N, 1]
        self.objectness_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # ── Step 2: Attention Pooling (Power-Set) ─────────────────────
        # グループ内の各トークンの重みを MLP で算出
        self.pool_score_mlp = nn.Linear(d_model, 1)
        self.pool_ln        = nn.LayerNorm(d_model)

        # ── Step 3: Group-level Self-Attention ────────────────────────
        self.group_attn     = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.group_ln       = nn.LayerNorm(d_model)
        self.group_score    = nn.Linear(d_model, 1)

        # ── Step 4: Gated Cross-Refinement ────────────────────────────
        self.cross_attn     = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_ln       = nn.LayerNorm(d_model)
        self.output_mlp     = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Gate は sigmoid(-10) ≈ 0 から出発 → 恒等写像として初期化
        self.gate_logit = nn.Parameter(torch.full((1,), -10.0))

        # ── Precompute subset mask [n_subsets, top_k] ─────────────────
        # 非空べき集合 (1023個) のバイナリマスクを定数バッファとして保持
        mask = torch.zeros(self.n_subsets, top_k)
        all_subsets = [
            c
            for size in range(1, top_k + 1)
            for c in combinations(range(top_k), size)
        ]
        for idx, subset in enumerate(all_subsets):
            mask[idx, list(subset)] = 1.0
        self.register_buffer("_subset_mask", mask)  # [1023, 10]

    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]  (SigLIP 視覚トークン)
        returns: [B, N, D]  (論理文脈で強化されたトークン)
        """
        B, N, D = x.shape

        # ══ Step 1: Pair-wise Scanning & STE Top-K ═══════════════════
        #
        # 全N個のトークン間でアテンションを計算し、
        # 各トークンの「物体らしさ (objectness)」スコアを算出する。
        #
        # Shapes:
        #   x          : [B, N, D]
        #   ctx        : [B, N, D]  ← pair_attn の出力
        #   scores     : [B, N]
        #   topk_idx   : [B, top_k]
        #   x_topk     : [B, top_k, D]

        ctx, _ = self.pair_attn(x, x, x)          # [B, N, D]
        ctx    = self.pair_ln(x + ctx)             # 残差 + LayerNorm

        scores = self.objectness_mlp(ctx).squeeze(-1)   # [B, N]

        # Hard Top-K (前向き選択)
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        # topk_idx: [B, top_k]

        # index_select でトークンを抽出
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # [B, top_k, D]
        x_topk = torch.gather(x, 1, topk_idx_exp)                 # [B, top_k, D]

        # ── STE Trick ─────────────────────────────────────────────────
        # 選択されたトークンのスコアを付加するが、逆伝播は straight-through。
        # 論文: Bengio et al. 2013 "Estimating or Propagating Gradients
        #        Through Stochastic Neurons"
        #
        #   X_top10 = X_selected + (S_selected - S_selected.detach())
        #
        # forward では S_selected - S_selected.detach() = 0 なので値は変わらない。
        # backward では S_selected.detach() の勾配はゼロなので、
        # ∂L/∂S_selected がそのまま通過する（straight-through）。
        topk_scores_norm = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # [B, top_k, 1]
        x_topk = x_topk + (topk_scores_norm - topk_scores_norm.detach())
        # x_topk: [B, top_k, D]

        # ══ Step 2: Power-Set Generation & Attention Pooling ══════════
        #
        # top_k=10 個のシードから非空部分集合 2^10-1=1023 通りを生成し、
        # 各グループを Attention Pooling で固定次元 D のベクトルに集約する。
        #
        # 【ベクトル化による並列処理】
        # ① pool_score_mlp で全10トークンの重みスコアを一括計算
        # ② subset_mask でマスク付きSoftmax → 重み [B, 1023, 10]
        # ③ einsum で加重和 → [B, 1023, D]
        #
        # Shapes:
        #   raw_w      : [B, top_k]
        #   mask       : [1023, top_k]  (バッファ)
        #   weights    : [B, 1023, top_k]
        #   G          : [B, 1023, D]

        raw_w = self.pool_score_mlp(x_topk).squeeze(-1)         # [B, top_k]

        # マスク付きスコア: サブセット外トークンを -1e9 に
        mask   = self._subset_mask.to(x.device)                  # [1023, 10]
        raw_w_exp  = raw_w.unsqueeze(1).expand(B, self.n_subsets, self.top_k)
        #                                                          [B, 1023, 10]
        masked_raw = raw_w_exp + (mask - 1.0) * 1e9              # [B, 1023, 10]
        weights    = F.softmax(masked_raw, dim=-1)                # [B, 1023, 10]

        # 加重和: [B, 1023, 10] × [B, 10, D] → [B, 1023, D]
        G = torch.einsum("bsn,bnd->bsd", weights, x_topk)        # [B, 1023, D]
        G = self.pool_ln(G)

        # ══ Step 3: Logical Scoring & Gated Filtering ════════════════
        #
        # 1023 個のグループトークン間で Self-Attention を行い、
        # 「グループとしての論理的完結度」を計算。上位 top_m 個を選ぶ。
        #
        # Shapes:
        #   G_attn     : [B, 1023, D]
        #   group_s    : [B, 1023]
        #   topm_idx   : [B, top_m]
        #   G_refined  : [B, top_m, D]

        G_attn, _ = self.group_attn(G, G, G)                     # [B, 1023, D]
        G         = self.group_ln(G + G_attn)

        group_s   = self.group_score(G).squeeze(-1)               # [B, 1023]

        # Top-m selection with STE
        topm_scores, topm_idx = torch.topk(group_s, self.top_m, dim=-1)
        topm_idx_exp = topm_idx.unsqueeze(-1).expand(-1, -1, D)
        G_refined    = torch.gather(G, 1, topm_idx_exp)           # [B, top_m, D]

        topm_scores_norm = F.softmax(topm_scores, dim=-1).unsqueeze(-1)
        G_refined = G_refined + (topm_scores_norm - topm_scores_norm.detach())
        # G_refined: [B, top_m, D]

        # ══ Step 4: Gated Cross-Refinement ═══════════════════════════
        #
        # 元の 512 パッチ (Query) が、20 個の論理グループ (Key/Value) を
        # 参照することで「自分がどの論理オブジェクトに属するか」を学ぶ。
        #
        # X_final = X + MLP(X_refined) * gate
        # gate = sigmoid(gate_logit)  ← 初期値 ≈ 0 (恒等写像から出発)
        #
        # Shapes:
        #   x_refined  : [B, N, D]   (Q=x, KV=G_refined)
        #   gate       : scalar ∈ (0,1)
        #   x_final    : [B, N, D]

        x_ref, _  = self.cross_attn(x, G_refined, G_refined)     # [B, N, D]
        x_ref     = self.cross_ln(x + x_ref)

        gate      = torch.sigmoid(self.gate_logit)                # scalar
        x_final   = x + self.output_mlp(x_ref) * gate            # [B, N, D]

        return x_final

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"top_k={self.top_k} → {self.n_subsets} subsets, top_m={self.top_m}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Vision Tower Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PSTTVisionWrapper(nn.Module):
    """
    ビジョンタワーを包み込み、出力の last_hidden_state に PSTT を適用するラッパー。

    Gemma-3 の forward pass:
        pixel_values → [vision_tower] → last_hidden_state: [B, N, D_vis]
                     → [multi_modal_projector] → [B, N, D_text]
                     → Gemma decoder

    PSTT の挿入点:
        pixel_values → [vision_tower] → [PSTT] → last_hidden_state (refined)
                     → [multi_modal_projector] → Gemma decoder
    """

    def __init__(self, vision_tower: nn.Module, pstt: PSTTModule):
        super().__init__()
        self.vision_tower = vision_tower
        self.pstt = pstt

    def forward(self, *args, **kwargs):
        output = self.vision_tower(*args, **kwargs)

        if hasattr(output, "last_hidden_state"):
            h = output.last_hidden_state              # [B, N, D]
            # PSTT は float32 で計算し、元の dtype に戻す
            refined = self.pstt(h.float()).to(h.dtype)
            output.last_hidden_state = refined
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Installation Helper
# ─────────────────────────────────────────────────────────────────────────────

def install_pstt(
    model,
    d_model:  int  = None,   # None → model config から自動取得
    n_heads:  int  = 8,
    top_k:    int  = 10,
    top_m:    int  = 20,
    freeze_base: bool = True,
) -> PSTTModule:
    """
    Gemma-3 モデルに PSTT を組み込む。

    1. ベースモデル（SigLIP + Gemma decoder）をフリーズ
    2. vision_tower を PSTTVisionWrapper で置き換え
    3. PSTT の重みのみを学習対象とする「アダプター・チューニング」

    Returns:
        pstt_module: 学習・推論に使う PSTTModule インスタンス
    """

    # ── d_model を自動取得 ─────────────────────────────────────────
    if d_model is None:
        vcfg = getattr(model.config, "vision_config", None)
        if vcfg is not None:
            d_model = getattr(vcfg, "hidden_size", 1152)
        else:
            d_model = 1152  # SigLIP-400M デフォルト
        print(f"[PSTT] d_model auto-detected: {d_model}")

    # ── vision_tower を探す ────────────────────────────────────────
    tower_attr = None
    for attr in ["vision_tower", "visual_model", "vision_model"]:
        if hasattr(model, attr):
            tower_attr = attr
            break
        # model.model.* も確認
        if hasattr(getattr(model, "model", None), attr):
            tower_attr = f"model.{attr}"
            break

    if tower_attr is None:
        attrs = [a for a in dir(model) if not a.startswith("_")]
        raise RuntimeError(
            f"vision_tower が見つかりません。\n"
            f"利用可能な属性: {attrs}"
        )

    # ── フリーズ ──────────────────────────────────────────────────
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False
        print("[PSTT] ベースモデルをフリーズしました (requires_grad=False)")

    # ── PSTT モジュールを生成 ──────────────────────────────────────
    pstt = PSTTModule(d_model=d_model, n_heads=n_heads, top_k=top_k, top_m=top_m)

    # ── vision_tower を置き換え ────────────────────────────────────
    if "." in tower_attr:
        # "model.vision_tower" のようなネストしたケース
        parts = tower_attr.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        original_tower = getattr(parent, parts[-1])
        setattr(parent, parts[-1], PSTTVisionWrapper(original_tower, pstt))
    else:
        original_tower = getattr(model, tower_attr)
        setattr(model, tower_attr, PSTTVisionWrapper(original_tower, pstt))

    pstt_params   = sum(p.numel() for p in pstt.parameters())
    base_frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[PSTT] インストール完了: model.{tower_attr} → PSTTVisionWrapper")
    print(f"  d_model={d_model}, top_k={top_k} → {2**top_k-1} subsets, top_m={top_m}")
    print(f"  PSTT パラメータ (学習対象): {pstt_params:,}")
    print(f"  ベースモデル (フリーズ済):  {base_frozen:,}")

    return pstt


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PSTT Smoke Test ===\n")

    B, N, D = 1, 512, 1152   # 1792×896 入力、SigLIP-400M 想定
    top_k, top_m = 10, 20

    pstt = PSTTModule(d_model=D, n_heads=8, top_k=top_k, top_m=top_m)
    pstt.eval()

    x = torch.randn(B, N, D)
    print(f"入力  X         : {list(x.shape)}")

    with torch.no_grad():
        # Step1 中間テンソルを手動で追跡
        ctx, _ = pstt.pair_attn(x, x, x)
        ctx    = pstt.pair_ln(x + ctx)
        scores = pstt.objectness_mlp(ctx).squeeze(-1)
        _, topk_idx = torch.topk(scores, top_k, dim=-1)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        x_topk = torch.gather(x, 1, topk_idx_exp)
        print(f"Step1 X_top{top_k}   : {list(x_topk.shape)}")

        raw_w  = pstt.pool_score_mlp(x_topk).squeeze(-1)
        mask   = pstt._subset_mask
        n_sub  = pstt.n_subsets
        raw_w_exp  = raw_w.unsqueeze(1).expand(B, n_sub, top_k)
        masked_raw = raw_w_exp + (mask - 1.0) * 1e9
        weights    = torch.softmax(masked_raw, dim=-1)
        G = torch.einsum("bsn,bnd->bsd", weights, x_topk)
        G = pstt.pool_ln(G)
        print(f"Step2 G         : {list(G.shape)}  ({n_sub} subsets)")

        G_attn, _ = pstt.group_attn(G, G, G)
        G         = pstt.group_ln(G + G_attn)
        group_s   = pstt.group_score(G).squeeze(-1)
        _, topm_idx = torch.topk(group_s, top_m, dim=-1)
        topm_idx_exp = topm_idx.unsqueeze(-1).expand(-1, -1, D)
        G_refined = torch.gather(G, 1, topm_idx_exp)
        print(f"Step3 G_refined : {list(G_refined.shape)}")

        x_final = pstt(x)
        print(f"Step4 X_final   : {list(x_final.shape)}")

    params = sum(p.numel() for p in pstt.parameters())
    print(f"\nPSTT パラメータ数: {params:,}")
    print("Smoke Test PASSED")
