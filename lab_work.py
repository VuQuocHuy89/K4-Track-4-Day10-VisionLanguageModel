"""Code thực hành cho lab_day10.ipynb; chạy từ thư mục gốc repository."""

import gc
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import CLASSES, read_jsonl, write_jsonl

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROMPT = (
    "Locate road damage in this image. Detect only pothole, crack, and manhole. "
    "Return only a JSON list of objects with label and bbox_2d coordinates "
    "as integers on a 0-1000 scale. If no damage is found, return []."
)
LLM_TARGETS = r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
VIT_TARGETS = r".*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"


def yolo_to_box(label, yolo):
    xc, yc, width, height = yolo
    values = [xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2]
    box = [round(max(0, min(1, value)) * 1000) for value in values]
    box[2] = max(box[0] + 1, box[2])
    box[3] = max(box[1] + 1, box[3])
    if label not in CLASSES or box[2] > 1000 or box[3] > 1000:
        raise ValueError(f"Invalid object: {label}, {yolo}")
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects):
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    boxes.sort(key=lambda obj: (obj["bbox_2d"][1], obj["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def check_data():
    meta = json.loads(Path("data/splits/meta.json").read_text(encoding="utf-8"))
    sessions = set()
    for split in ("train", "val", "test"):
        rows = read_jsonl(f"data/splits/{split}.jsonl")
        split_sessions = {row["session"] for row in rows}
        assert not sessions.intersection(split_sessions), f"Session leakage in {split}"
        sessions.update(split_sessions)
        assert len(rows) == meta["counts"][split]["images"]
        print(split, len(rows), meta["counts"][split]["boxes"])
    print("dataset_version:", meta["dataset_version"])
    return meta


def draw_boxes(image, gt, pred=None):
    from scripts.score import parse_output

    image = image.copy()
    draw = ImageDraw.Draw(image)
    items = [(obj["label"], yolo_to_box(obj["label"], obj["yolo"])["bbox_2d"], "lime") for obj in gt]
    if pred is not None:
        for obj in parse_output(pred) or []:
            items.append((obj["label"], obj["bbox"], "red"))
    for label, box, color in items:
        xy = [box[0] * image.width / 1000, box[1] * image.height / 1000,
              box[2] * image.width / 1000, box[3] * image.height / 1000]
        draw.rectangle(xy, outline=color, width=3)
        draw.text((xy[0], max(0, xy[1] - 12)), label, fill=color)
    return image


def load_processor():
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return processor


def load_model():
    import torch
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    assert torch.cuda.is_available(), "Colab Runtime phải dùng T4 GPU"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    return Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quantization, dtype=torch.float16,
        device_map={"": 0},
    )


def predict(model, processor, split, path):
    import torch

    model.eval()
    rows = read_jsonl(f"data/splits/{split}.jsonl")
    predictions = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            with Image.open(Path("data") / row["image_path"]) as image:
                message = [{"role": "user", "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": PROMPT},
                ]}]
                inputs = processor.apply_chat_template(
                    message, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to("cuda")
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            raw = processor.tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            predictions.append({"image_id": row["image_id"], "raw": raw})
            if index % 50 == 0:
                print(f"{split}: {index}/{len(rows)}")
    write_jsonl(path, predictions)
    return path


def encode_example(row, processor):
    import torch

    answer = objects_to_answer(row["objects"])
    with Image.open(Path("data") / row["image_path"]) as image:
        user = [{"role": "user", "content": [
            {"type": "image", "image": image.convert("RGB")},
            {"type": "text", "text": PROMPT},
        ]}]
        prompt = processor.apply_chat_template(
            user, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        full = processor.apply_chat_template(
            user + [{"role": "assistant", "content": answer}],
            tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )
    n_prompt = prompt["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][0, :n_prompt], prompt["input_ids"][0])
    labels = full["input_ids"][0].clone()
    labels[:n_prompt] = -100
    assert (labels != -100).any(), "Assistant answer was masked"
    item = {key: full[key][0] for key in ("input_ids", "attention_mask")}
    item["labels"] = labels
    for key in ("mm_token_type_ids", "pixel_values", "image_grid_thw"):
        if key in full:
            item[key] = full[key][0] if key == "mm_token_type_ids" else full[key]
    return item


def collate(batch, pad_token_id):
    import torch

    pad = torch.nn.utils.rnn.pad_sequence
    out = {
        "input_ids": pad([item["input_ids"] for item in batch], batch_first=True,
                         padding_value=pad_token_id),
        "attention_mask": pad([item["attention_mask"] for item in batch], batch_first=True,
                              padding_value=0),
        "labels": pad([item["labels"] for item in batch], batch_first=True,
                      padding_value=-100),
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
    }
    if "mm_token_type_ids" in batch[0]:
        out["mm_token_type_ids"] = pad(
            [item["mm_token_type_ids"] for item in batch], batch_first=True,
            padding_value=0,
        )
    return out


def apply_setup(model, setup, rank=16):
    import torch
    from peft import LoraConfig, get_peft_model

    assert setup in ("E1", "E2", "E3")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_input_require_grads()
    pattern = LLM_TARGETS + ("|" + VIT_TARGETS if setup == "E3" else "")
    matched = [name for name, _ in model.named_modules() if re.fullmatch(pattern, name)]
    expected = 348 if setup == "E3" else 252
    assert len(matched) == expected, f"Matched {len(matched)} LoRA modules, expected {expected}"
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        target_modules=pattern, bias="none", task_type="CAUSAL_LM",
    ))
    if setup in ("E2", "E3"):
        projector_counts = {"visual.merger": 0, "visual.deepstack_merger_list": 0}
        for name, parameter in model.named_parameters():
            for module_name in projector_counts:
                if module_name in name:
                    projector_counts[module_name] += parameter.numel()
                    parameter.data = parameter.data.to(torch.float32)
                    parameter.requires_grad_(True)
        assert all(projector_counts.values()), f"Projector parameters not found: {projector_counts}"
        count = sum(projector_counts.values())
        print("projector params", projector_counts, "total", count)
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters()
               if parameter.requires_grad), "Trainable parameters must be FP32 for GradScaler"
    print(setup, "rank", rank, "LoRA modules", len(matched), "trainable params", trainable)
    return model, trainable


def save_checkpoint(model, name, setup, drive_root=None):
    import torch

    target = Path("work/checkpoints") / name
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    if setup in ("E2", "E3"):
        projector = {key: value.detach().cpu() for key, value in model.named_parameters()
                     if "visual.merger" in key or "visual.deepstack_merger_list" in key}
        assert projector
        torch.save(projector, target / "projector.pt")
    if drive_root:
        shutil.copytree(target, Path(drive_root) / name, dirs_exist_ok=True)
    return target


def load_checkpoint(name, setup, drive_root=None):
    import torch
    from peft import PeftModel

    path = Path("work/checkpoints") / name
    if not path.exists() and drive_root:
        path = Path(drive_root) / name
    assert path.exists(), f"Missing checkpoint: {path}"
    model = PeftModel.from_pretrained(load_model(), path)
    if setup in ("E2", "E3"):
        projector = torch.load(path / "projector.pt", map_location="cpu", weights_only=True)
        current = dict(model.named_parameters())
        assert projector and all(key in current and current[key].shape == value.shape
                                 for key, value in projector.items())
        result = model.load_state_dict(projector, strict=False)
        assert not result.unexpected_keys
    return model


def train(name, setup, processor, train_fraction=1.0, drive_root=None):
    import torch
    from torch.utils.data import DataLoader, Dataset

    assert 0 < train_fraction <= 1
    torch.manual_seed(42)
    random.seed(42)
    rows = read_jsonl("data/splits/train.jsonl")
    if train_fraction < 1:
        rows = random.Random(42).sample(rows, int(len(rows) * train_fraction))

    class Examples(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return encode_example(rows[index], processor)

    loader = DataLoader(
        Examples(), batch_size=2, shuffle=True, num_workers=0,
        collate_fn=lambda batch: collate(batch, processor.tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(42),
    )
    model, trainable = apply_setup(load_model(), setup)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=0)
    warmup = 15
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup else
        0.5 * (1 + math.cos(math.pi * (step - warmup) / (300 - warmup))),
    )
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    iterator = iter(loader)
    for step in range(1, 301):
        total_loss = 0.0
        for _ in range(4):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / 4
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            total_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 25 == 0:
            print(f"{name}: step {step}/300, loss {total_loss:.4f}")
        if step % 100 == 0:
            save_checkpoint(model, name, setup, drive_root)
    metadata = {
        "trainable_params": trainable,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "train_minutes": (time.perf_counter() - start) / 60,
    }
    assert all(value > 0 for value in metadata.values())
    Path(f"work/{name}_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def score(name, prediction_path, metadata=None, out=None):
    out = out or f"report/evidence/{name.lower()}.json"
    command = [sys.executable, "scripts/score.py", "--pred", str(prediction_path),
               "--name", name, "--out", out]
    for key, value in (metadata or {}).items():
        command += ["--meta", f"{key}={value}"]
    subprocess.run(command, check=True)
    return json.loads(Path(out).read_text(encoding="utf-8"))


def sync_outputs(drive_root, restore=False):
    if not drive_root:
        return
    backup = Path(drive_root).parent / "outputs"
    for directory in ("report/evidence", "work"):
        source = backup / directory if restore else Path(directory)
        destination = Path(directory) if restore else backup / directory
        destination.mkdir(parents=True, exist_ok=True)
        if source.exists():
            for file in source.glob("*.json*"):
                shutil.copy2(file, destination / file.name)


def run_trained(name, setup, processor, train_fraction=1.0, drive_root=None):
    import torch

    metadata = train(name, setup, processor, train_fraction, drive_root)
    model = load_checkpoint(name, setup, drive_root)
    prediction_path = predict(model, processor, "test", f"work/{name.lower()}.jsonl")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    result = score(name, prediction_path, metadata)
    sync_outputs(drive_root)
    return result


def save_examples(name, count=6):
    from scripts.score import _gt_boxes, _match, parse_output

    rows = read_jsonl("data/splits/test.jsonl")
    predictions = {row["image_id"]: row["raw"] for row in read_jsonl(f"work/{name.lower()}.jsonl")}
    correct, errors = [], []
    for row in rows:
        parsed = parse_output(predictions[row["image_id"]]) or []
        totals, _ = _match(_gt_boxes(row), parsed)
        tp = sum(value[0] for value in totals.values())
        fp = sum(value[1] for value in totals.values())
        fn = sum(value[2] for value in totals.values())
        if tp:
            correct.append((row, tp, fp, fn))
        if fp or fn:
            errors.append((row, tp, fp, fn))
    selected = correct[:count // 2]
    used = {item[0]["image_id"] for item in selected}
    selected += [item for item in errors if item[0]["image_id"] not in used][:count - len(selected)]
    assert len(selected) == count, "Need more correct/error examples"
    target = Path("report/figures")
    target.mkdir(parents=True, exist_ok=True)
    for index, (row, tp, fp, fn) in enumerate(selected, 1):
        with Image.open(Path("data") / row["image_path"]) as image:
            annotated = draw_boxes(image.convert("RGB"), row["objects"], predictions[row["image_id"]])
            canvas = Image.new("RGB", (annotated.width, annotated.height + 24), "white")
            canvas.paste(annotated, (0, 24))
            ImageDraw.Draw(canvas).text(
                (4, 5), f"{row['image_id']} | GT: green  Pred: red | TP={tp} FP={fp} FN={fn}",
                fill="black",
            )
            canvas.save(target / f"{name.lower()}_{index}_{row['image_id']}.png")
        print(index, row["image_id"], f"TP={tp} FP={fp} FN={fn}")
    print("Green = ground truth; red = prediction")


def self_check():
    assert yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4])["bbox_2d"] == [400, 300, 600, 700]
    assert objects_to_answer([]) == "[]"
    assert objects_to_answer([
        {"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]},
        {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]},
    ]).startswith('[{"label":"pothole"')
