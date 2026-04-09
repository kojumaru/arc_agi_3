#!/usr/bin/env python3
"""
純画像エージェント（vLLM 高速推論版 + アテンションヒートマップ）
- vLLM で高速推論（10-50倍高速化）
- google/gemma-3-4b-it のアテンション重みを取得してヒートマップ生成
"""

import os
import re
import random
import math
from datetime import datetime
from pathlib import Path
import base64
import io

import torch
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# HuggingFace ログイン（Gemma3 アクセスに必要）
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✓ HuggingFace にログインしました")

MODEL_ID  = "google/gemma-3-4b-it"
NUM_LAYERS = 34
MAX_STEPS = 20
API_KEY   = os.environ["ARC_API_KEY"]
BASE_URL  = "https://three.arcprize.org"
HEADERS   = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# グリッド比較画像の設定
GRID_COLS = 5
GRID_THUMB_SCALE = 2  # 1/2 サイズ
GRID_LABEL_H = 18
GRID_BG_COLOR = (20, 20, 20)
GRID_TEXT_COLOR = (220, 220, 220)

ARC_RGB = [
    (  0,   0,   0),  # 0: black
    (  0, 116, 217),  # 1: blue
    (255,  65,  54),  # 2: red
    ( 46, 204,  64),  # 3: green
    (255, 220,   0),  # 4: yellow
    (170, 170, 170),  # 5: gray
    (241,  18, 190),  # 6: magenta
    (255, 133,  27),  # 7: orange
    (127, 219, 255),  # 8: cyan
    (255, 255, 255),  # 9: white
    (135,  12,  37),  # 10: crimson
    (  1, 255, 112),  # 11: lime
    (  0,  31,  63),  # 12: navy
    ( 61, 153, 112),  # 13: olive
    ( 57, 204, 204),  # 14: teal
    (221, 221, 221),  # 15: silver
]

# ─── transformers モデル初期化（起動時に1回だけ） ──────────────────────
print(f"Loading {MODEL_ID} with transformers...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print(f"✓ Model loaded  (device={next(model.parameters()).device})")

# フォント初期化（グリッド生成用）
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
except Exception:
    FONT = ImageFont.load_default()
# ──────────────────────────────────────────────────────────────────────


def frame_to_image(frame, cell=10):
    h, w = len(frame), len(frame[0])
    img  = Image.new("RGB", (w * cell, h * cell))
    pixels = img.load()
    for r in range(h):
        for c in range(w):
            color_idx = frame[r][c]
            rgb = ARC_RGB[color_idx] if color_idx < len(ARC_RGB) else (255, 0, 255)
            for dr in range(cell):
                for dc in range(cell):
                    pixels[c * cell + dc, r * cell + dr] = rgb
    return img


def make_comparison_image(prev_img, curr_img):
    """BEFORE / AFTER を横並びにしてラベルを焼き込んだ1枚のPNGを返す"""
    label_h = 28
    gap     = 8
    w, h    = curr_img.width, curr_img.height
    canvas  = Image.new("RGB", (w * 2 + gap, h + label_h), (30, 30, 30))
    canvas.paste(prev_img, (0, label_h))
    canvas.paste(curr_img, (w + gap, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, w - 1, label_h - 1], fill=(60, 60, 60))
    draw.text((6, 4), "BEFORE (previous state)", fill=(200, 200, 200), font=font)
    draw.rectangle([w + gap, 0, w * 2 + gap - 1, label_h - 1], fill=(40, 80, 40))
    draw.text((w + gap + 6, 4), "AFTER (current state)", fill=(180, 255, 180), font=font)
    return canvas


def hot_colormap(value):
    """value in [0,1] を hot カラーマップ (R,G,B) に変換"""
    # 単純な hot colormap: black → red → yellow → white
    if value < 0.25:
        r, g, b = int(value * 4 * 255), 0, 0
    elif value < 0.5:
        r, g, b = 255, int((value - 0.25) * 4 * 255), 0
    elif value < 0.75:
        r, g, b = 255, 255, int((value - 0.5) * 4 * 255)
    else:
        r, g, b = 255, 255, int(255 * (1 - (value - 0.75) * 2))
    return (r, g, b)


def overlay_heatmap(base_img, heatmap_2d, alpha=0.55):
    """2D アテンションヒートマップを hot カラーマップで画像にオーバーレイ"""
    w, h = base_img.width, base_img.height
    hm = np.array(heatmap_2d, dtype=np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    # バイリニア補完でリサイズ
    hm_pil = Image.fromarray((hm * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    hm_arr = np.array(hm_pil) / 255.0

    # hot colormap を適用
    hm_colored_arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            hm_colored_arr[y, x] = hot_colormap(hm_arr[y, x])

    hm_colored = Image.fromarray(hm_colored_arr)
    return Image.blend(base_img.convert("RGB"), hm_colored, alpha)


def make_heatmap_comparison(prev_img, curr_img, heatmap_2d):
    """3パネル: BEFORE | AFTER | ATTENTION HEATMAP"""
    label_h = 28
    gap     = 8
    w, h    = curr_img.width, curr_img.height
    hm_img  = overlay_heatmap(curr_img, heatmap_2d)

    canvas = Image.new("RGB", (w * 3 + gap * 2, h + label_h), (30, 30, 30))
    canvas.paste(prev_img, (0, label_h))
    canvas.paste(curr_img, (w + gap, label_h))
    canvas.paste(hm_img,   (w * 2 + gap * 2, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, w - 1, label_h - 1], fill=(60, 60, 60))
    draw.text((6, 4), "BEFORE", fill=(200, 200, 200), font=font)

    draw.rectangle([w + gap, 0, w * 2 + gap - 1, label_h - 1], fill=(40, 80, 40))
    draw.text((w + gap + 6, 4), "AFTER", fill=(180, 255, 180), font=font)

    draw.rectangle([w * 2 + gap * 2, 0, w * 3 + gap * 2 - 1, label_h - 1], fill=(80, 40, 40))
    draw.text((w * 2 + gap * 2 + 6, 4), "ATTENTION HEATMAP", fill=(255, 180, 180), font=font)

    return canvas


def _save_heatmap_image(hm, prev_img, curr_img):
    """
    ヒートマップを画像にオーバーレイして返す
    prev_img がある場合は3パネル比較、ない場合は単体オーバーレイ
    """
    if prev_img:
        return make_heatmap_comparison(prev_img, curr_img, hm)
    else:
        return overlay_heatmap(curr_img, hm)


def _clear_cache():
    """GPU/MPS メモリをクリア"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def attention_for_layer(layer_attn, image_positions):
    """
    1層分の attention tensor から画像トークンへの attention を 2D グリッドに変換

    Args:
        layer_attn:      shape (1, num_heads, seq_len, seq_len) - 1層分のattention
        image_positions: shape (n_image_tokens,) - 画像トークンのインデックス

    Returns:
        np.ndarray shape (grid_size, grid_size) の float32
    """
    avg_heads = layer_attn[0].float().mean(dim=0)
    attn_to_img = avg_heads[-1, image_positions].cpu().numpy()
    n = len(attn_to_img)
    grid_size = max(1, int(np.sqrt(n)))
    sq = grid_size * grid_size
    pad = np.zeros(sq, dtype=np.float32)
    pad[:min(n, sq)] = attn_to_img[:min(n, sq)]
    return pad.reshape(grid_size, grid_size)


def infer(system_text, user_text, img):
    """
    transformers による推論 + 全35層 attention ヒートマップ取得
    Returns: (generated_text: str, layer_heatmaps: list[np.ndarray] 長さ35)
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # ① 入力準備
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    # ② テキスト生成
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    generated_text = processor.decode(gen_ids[0, input_len:], skip_special_tokens=True)
    del gen_ids
    _clear_cache()

    # ③ 画像トークン位置特定
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]

    # ④ 全34層 attention 取得
    if len(image_positions) == 0:
        print("[WARN] 画像トークンが見つかりません")
        layer_heatmaps = [np.zeros((1, 1)) for _ in range(NUM_LAYERS)]
        return generated_text, layer_heatmaps

    with torch.no_grad():
        fwd = model(**inputs, output_attentions=True)

    layer_heatmaps = []
    for i, layer_attn in enumerate(fwd.attentions):
        hm = attention_for_layer(layer_attn, image_positions)
        layer_heatmaps.append(hm)

    del fwd
    _clear_cache()

    return generated_text, layer_heatmaps


def save_all_layer_heatmaps(log_dir, step, prev_img, curr_img, layer_heatmaps):
    """
    全34層のヒートマップを保存する

    出力ファイル:
        step{NNN}_layer{LL:02d}_heatmap.png  (34枚)
        step{NNN}_layer_comparison.png        (34層グリッド比較画像)
        step{NNN}_heatmap.png                 (最終層、後方互換)
    """
    # 各層個別保存
    for i, hm in enumerate(layer_heatmaps):
        layer_num = i + 1
        img_out = _save_heatmap_image(hm, prev_img, curr_img)
        img_out.save(log_dir / f"step{step:03d}_layer{layer_num:02d}_heatmap.png")

    # グリッド比較画像（GRID_COLS列×rows行）
    rows = math.ceil(len(layer_heatmaps) / GRID_COLS)
    thumb_w = max(1, curr_img.width // GRID_THUMB_SCALE)
    thumb_h = max(1, curr_img.height // GRID_THUMB_SCALE)
    cell_h = thumb_h + GRID_LABEL_H
    canvas = Image.new("RGB", (GRID_COLS * thumb_w, rows * cell_h), GRID_BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    for i, hm in enumerate(layer_heatmaps):
        col = i % GRID_COLS
        row = i // GRID_COLS
        thumb = overlay_heatmap(curr_img, hm).resize((thumb_w, thumb_h), Image.BILINEAR)
        x, y = col * thumb_w, row * cell_h
        canvas.paste(thumb, (x, y + GRID_LABEL_H))
        draw.text((x + 2, y + 2), f"L{i+1:02d}", fill=GRID_TEXT_COLOR, font=FONT)

    canvas.save(log_dir / f"step{step:03d}_layer_comparison.png")

    # 後方互換: 最終層で既存形式の保存
    hm_last = layer_heatmaps[-1]
    _save_heatmap_image(hm_last, prev_img, curr_img).save(
        log_dir / f"step{step:03d}_heatmap.png"
    )


def image_to_base64(img):
    """PIL Image を base64 エンコード"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _box(title, body, style="double"):
    """ボックス描画文字でセクションを囲む"""
    if style == "double":
        tl, tr, bl, br, h, v = "╔", "╗", "╚", "╝", "═", "║"
    elif style == "bold":
        tl, tr, bl, br, h, v = "┏", "┓", "┗", "┛", "━", "┃"
    else:
        tl, tr, bl, br, h, v = "┌", "┐", "└", "┘", "─", "│"
    width = 72
    lines = body.splitlines()
    bar   = h * (width - 2)
    out   = [f"{tl}{h * ((width - len(title) - 2) // 2)} {title} {h * ((width - len(title) - 1) // 2)}{tr}"]
    for line in lines:
        while len(line) > width - 4:
            out.append(f"{v} {line[:width-4]} {v}")
            line = line[width-4:]
        out.append(f"{v} {line:<{width-4}} {v}")
    out.append(f"{bl}{bar}{br}")
    return "\n".join(out)


def log_llm_exchange(log_path, step, history_text, system_text, user_text, response_text):
    """HISTORY / PROMPT / RESPONSE の3層構造でログ出力"""
    sep   = "━" * 72
    block = f"\n{sep}\n  STEP {step}\n{sep}\n\n"
    if history_text:
        block += _box("📜 HISTORY", history_text, style="single") + "\n\n"
    prompt_body = f"[SYSTEM]\n{system_text}\n\n[USER]\n{user_text}"
    block += _box("★ PROMPT", prompt_body, style="bold") + "\n\n"
    block += _box("💬 RESPONSE", response_text, style="double") + "\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(block + "\n")
    print(block)


def parse_action(llm_output, available_actions):
    m = re.search(r"ACTION\s*:\s*([1-9])", llm_output, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n in available_actions:
            return n
    nums = re.findall(r"\b([1-9])\b", llm_output)
    for n in reversed(nums):
        if int(n) in available_actions:
            return int(n)
    return random.choice(available_actions)


def extract_section(text, heading):
    m = re.search(rf"#+ {heading}\s*\n(.*?)(?=\n#|\Z)", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def write_log(log_path, text):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = Path("logs_vision") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path  = log_dir / "game_log.txt"

    session = requests.Session()
    session.headers.update(HEADERS)

    card    = session.post(f"{BASE_URL}/api/scorecard/open", json={}).json()
    card_id = card["card_id"]

    games   = session.get(f"{BASE_URL}/api/games").json()
    game_id = next(g["game_id"] for g in games if g["game_id"].startswith("ls20"))

    resp = session.post(f"{BASE_URL}/api/cmd/RESET",
                        json={"game_id": game_id, "card_id": card_id}).json()
    guid = resp.get("guid")

    header = f"=== GAME START: {game_id}  [{timestamp}] ==="
    write_log(log_path, header)
    write_log(log_path, "=" * len(header))

    SYSTEM = (
        "You are an agent playing an unknown game. "
        "You are not told the rules — you must infer them purely from observation.\n\n"
        "Structure your response using exactly these sections:\n\n"
        "# Environment Observation\n"
        "Describe what you see in the current game state.\n\n"
        "# Changes from Last Step\n"
        "Describe what changed between BEFORE and AFTER. (Skip on the first step.)\n\n"
        "# Action Hypotheses\n"
        "For each available action, state your current hypothesis about what it does. "
        "Update based on any new evidence from this step.\n\n"
        "# Game Goal\n"
        "State your current hypothesis about the winning condition. "
        "Update if new evidence changes your understanding.\n\n"
        "# Next Action\n"
        "State which action you choose and why.\n\n"
        "End your response with exactly: ACTION: N  (N is the action number)"
    )

    history  = []   # {"step": N, "action": N, "level_cleared": bool,
                    #  "action_hypotheses": str, "game_goal": str}
    prev_img = None

    for step in range(MAX_STEPS):
        state   = resp.get("state", "UNKNOWN")
        actions = resp.get("available_actions", [])

        write_log(log_path, f"\n{'─'*60}")
        write_log(log_path, f"STEP {step + 1}  |  state={state}")
        write_log(log_path, f"{'─'*60}")

        if state in ("WIN", "GAME_OVER"):
            write_log(log_path, f"\n=== GAME END: {state} ===")
            break

        frame = resp["frame"][0]
        img   = frame_to_image(frame)

        img_path = log_dir / f"step{step+1:03d}.png"
        img.save(img_path)
        write_log(log_path, f"[画像保存] {img_path}")

        levels_done = resp.get("levels_completed", 0)
        win_levels  = resp.get("win_levels", 0)

        # ── history テキスト構築 ──────────────────────────────────────
        history_lines = []
        if history:
            history_lines.append("=== Action History ===")
            for h in history[-10:]:
                level_note = "  [LEVEL CLEARED]" if h["level_cleared"] else ""
                history_lines.append(f"Step {h['step']}: ACTION{h['action']}{level_note}")
            last = history[-1]
            if last.get("action_hypotheses"):
                history_lines.append("\n=== Hypotheses carried over from last step ===")
                history_lines.append("# Action Hypotheses")
                history_lines.append(last["action_hypotheses"])
                history_lines.append("# Game Goal")
                history_lines.append(last.get("game_goal", "(unknown)"))
        history_text = "\n".join(history_lines)

        # ── LLMに渡す画像と説明テキスト ──────────────────────────────
        if prev_img:
            send_img = make_comparison_image(prev_img, img)
            img_note = (
                "The attached image shows BEFORE (left) and AFTER (current, right) side by side. "
                "Compare them to understand what the last action changed."
            )
        else:
            send_img = img
            img_note = "The attached image shows the current game state (first step, no previous state)."

        prompt = (
            f"{history_text}\n\n" if history_text else ""
        ) + (
            f"Step {step + 1} | Available actions: {actions} | "
            f"Level progress: {levels_done} / {win_levels}\n\n"
            f"{img_note}"
        )

        # ── 推論 + ヒートマップ ───────────────────────────────────────
        llm_output, layer_heatmaps = infer(SYSTEM, prompt, send_img)

        log_llm_exchange(
            log_path,
            step          = step + 1,
            history_text  = history_text,
            system_text   = SYSTEM,
            user_text     = (
                f"Step {step + 1} | Available actions: {actions} | "
                f"Level progress: {levels_done} / {win_levels}\n\n"
                f"{img_note}\n(+1 image)"
            ),
            response_text = llm_output,
        )

        # ── ヒートマップ付き比較画像の保存（全35層） ───────────────
        save_all_layer_heatmaps(log_dir, step + 1, prev_img, img, layer_heatmaps)
        write_log(log_path, f"[ヒートマップ保存] step{step+1:03d}_layer*.png x{NUM_LAYERS} + comparison")

        action = parse_action(llm_output, actions)
        write_log(log_path, f"\n→ ACTION{action} を実行")

        action_hypotheses = extract_section(llm_output, "Action Hypotheses")
        game_goal         = extract_section(llm_output, "Game Goal")

        resp = session.post(
            f"{BASE_URL}/api/cmd/ACTION{action}",
            json={"game_id": game_id, "guid": guid}
        ).json()
        if resp.get("guid"):
            guid = resp["guid"]

        new_levels    = resp.get("levels_completed", 0)
        level_cleared = new_levels > levels_done
        history.append({
            "step": step + 1, "action": action,
            "level_cleared": level_cleared,
            "action_hypotheses": action_hypotheses,
            "game_goal": game_goal,
        })
        prev_img = img

        write_log(log_path, f"  新しい state: {resp.get('state', 'UNKNOWN')}")

    write_log(log_path, f"\nログ保存先: {log_path}")


if __name__ == "__main__":
    main()
