# ARC-AGI-3 LLM Agent

ARC-AGI-3 のインタラクティブゲームを自律的にクリアする LLM エージェント。
Gemma-3-4b-it（マルチモーダル）をベースに、独自の視覚強化アーキテクチャ **PSTT** を実装。

---

## エージェント一覧

| ファイル | 概要 |
|---|---|
| `agent_basic.py` | シンプルな推論エージェント（ベースライン） |
| `agent_vision.py` | アテンションヒートマップ可視化つき |
| `agent_pstt.py` | **PSTT Vision Bridge** 統合エージェント |

---

## パイプライン

```
ARC-AGI-3 API
  frame: [[0,1,2,...], ...]  (最大64×64, 色インデックス 0〜15)

        ↓ frame_to_image()  cell=14 (64×14=896)

PIL Image  896×896px  (補間なし・くっきりグリッド)

        ↓ processor.apply_chat_template()

pixel_values: [1, 3, 896, 896]

        ↓ SigLIP  (patch_size=14 → 64×64=4096パッチ)

視覚トークン  [1, 4096, 1152]

        ↓ [PSTT Vision Bridge]  ← agent_pstt のみ

視覚トークン  [1, 4096, 1152]  (論理オブジェクト情報で強化)

        ↓ multi_modal_projector  (1152 → 2048)

[1, 4096, 2048]

        ↓ Gemma-3-4b-it decoder

"ACTION: 3"

        ↓ parse_action()

API へアクション送信
```

---

## PSTT (Power-Set Tokenizer Transformer)

SigLIP と Gemma デコーダの間に挿入する Vision Bridge アダプター。
「パッチ単位の処理」を「オブジェクト（集合）単位の論理処理」へ昇格させる。

```
視覚トークン X  [B, 4096, 1152]
  ↓ Step 1: Pair-wise Scanning & STE Top-K
    全トークン間でアテンション → Objectness スコア → 上位10個を選択
    X_top10  [B, 10, 1152]

  ↓ Step 2: Power-Set Generation & Attention Pooling
    2^10 - 1 = 1023通りの部分集合を生成
    マスク付きSoftmax + einsum で並列計算
    G  [B, 1023, 1152]

  ↓ Step 3: Logical Scoring & Gated Filtering
    グループ間 Self-Attention → 論理完結度スコア → 上位20グループ選択
    G_refined  [B, 20, 1152]

  ↓ Step 4: Gated Cross-Refinement
    Cross-Attention (Q=全4096パッチ, KV=選ばれた20グループ)
    X_final = X + MLP(X_refined) × gate
    X_final  [B, 4096, 1152]
```

**パラメータ数:** ~19M（Gemma-3-4b の 0.5% 未満、アダプター・チューニング対応）

---

## セットアップ

### 動作確認済み環境

- Python 3.11
- torch 2.0.1+cpu
- numpy <2 (1.x系)
- Windows 11 (15GB RAM)

```bash
# Python 3.11 で仮想環境を作成
python -m venv .venv
.venv/Scripts/activate

# PyTorch (CPU版, Windows互換)
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2"

# その他依存
pip install transformers huggingface_hub pillow requests python-dotenv
```

### 環境変数 (.env)

```
ARC_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token_here  # Gemma アクセスに必要
```

---

## 実行

```bash
# ベースラインエージェント
python agent_basic.py

# PSTT エージェント
python agent_pstt.py

# PSTT 無効（ベースライン比較）
PSTT_ENABLED=false python agent_pstt.py

# 学習済み PSTT 重みを使う
PSTT_CKPT=path/to/pstt.pt python agent_pstt.py
```

### PSTT モジュール単体テスト

```bash
python pstt_module.py
# Expected output:
# Step1 X_top10   : [1, 10, 1152]
# Step2 G         : [1, 1023, 1152]  (1023 subsets)
# Step3 G_refined : [1, 20, 1152]
# Step4 X_final   : [1, 512, 1152]
# Smoke Test PASSED
```

---

## モデル選定

| モデル | Vision | float16 RAM | 備考 |
|---|---|---|---|
| gemma-3-1b-it | ✗ | ~2GB | テキストのみ |
| **gemma-3-4b-it** | **✓** | **~8GB** | **採用（最小マルチモーダル）** |
| gemma-3-12b-it | ✓ | ~24GB | 15GB RAM では厳しい |

---

## ログ出力

```
logs_basic/YYYYMMDD_HHMMSS/   agent_basic
logs_vision/YYYYMMDD_HHMMSS/  agent_vision  (ヒートマップ含む)
logs_pstt/YYYYMMDD_HHMMSS/    agent_pstt
logs_base/YYYYMMDD_HHMMSS/    agent_pstt (PSTT無効時)
```
