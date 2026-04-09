#!/usr/bin/env python3
"""
シンプルLLMエージェント（ヒートマップなし）
- Gemma-3-4b-it で推論（transformers）
- M1 MacBook Air 8GB対応
"""

import os
import re
import random
from datetime import datetime
from pathlib import Path

import torch
import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# HuggingFace ログイン
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✓ HuggingFace にログインしました")

MODEL_ID  = "google/gemma-3-4b-it"
MAX_STEPS = 20
API_KEY   = os.environ["ARC_API_KEY"]
BASE_URL  = "https://three.arcprize.org"
HEADERS   = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

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

# ─── transformers モデル初期化 ─────────────────────────────────────
print(f"Loading {MODEL_ID} with transformers...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print(f"✓ Model loaded  (device={next(model.parameters()).device})")
# ───────────────────────────────────────────────────────────────────


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
    """BEFORE / AFTER を横並びに"""
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
    draw.text((6, 4), "BEFORE", fill=(200, 200, 200), font=font)
    draw.rectangle([w + gap, 0, w * 2 + gap - 1, label_h - 1], fill=(40, 80, 40))
    draw.text((w + gap + 6, 4), "AFTER", fill=(180, 255, 180), font=font)
    return canvas


def infer(system_text, user_text, img):
    """
    transformers推論（シンプル版、attentionなし）
    Returns: str (LLM出力)
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

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    generated_text = processor.decode(gen_ids[0, input_len:], skip_special_tokens=True)

    return generated_text


def _box(title, body, style="double"):
    """ボックス描画"""
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
    """ログ出力"""
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
    log_dir   = Path("logs_basic") / timestamp
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

    history  = []
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

        # history テキスト構築
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

        # LLMに渡す画像
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

        # ── 推論 ─────────────────────────────
        llm_output = infer(SYSTEM, prompt, send_img)

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
