import argparse
import gc

import torch


def gb(x):
    return x / (1024 ** 3)


def mem():
    torch.cuda.synchronize()
    return (
        gb(torch.cuda.memory_allocated()),
        gb(torch.cuda.memory_reserved()),
        gb(torch.cuda.max_memory_allocated()),
        gb(torch.cuda.max_memory_reserved()),
    )


def safe_mem():
    try:
        return tuple(f"{value:.3f}" for value in mem())
    except Exception:
        return ("N/A", "N/A", "N/A", "N/A")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Gemma4-31B LoRA VRAM benchmark reproducer. "
            "WARNING: this script is expected to hit VERIFIED_OOM for some long-context configs. "
            "Requires GPU and compatible ROCm stack."
        )
    )
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--model-id", default="unsloth/gemma-4-31B-it-unsloth-bnb-4bit")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device-map", default="sequential")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CLI/config only. Does not import Unsloth/bitsandbytes or allocate GPU tensors.",
    )
    args = p.parse_args()

    if args.dry_run:
        print("STATUS=DRY_RUN")
        print(f"model_id={args.model_id}")
        print(f"rank={args.rank}")
        print(f"seq_len={args.seq_len}")
        print(f"local_files_only={args.local_files_only}")
        print(f"device_map={args.device_map}")
        return

    import bitsandbytes as bnb
    from unsloth import FastModel

    result = {
        "STATUS": "VERIFIED_ERROR",
        "model_id": args.model_id,
        "rank": args.rank,
        "seq_len": args.seq_len,
        "input_shape": "N/A",
        "trainable_params": "N/A",
        "load_vram_gb": "N/A",
        "lora_vram_gb": "N/A",
        "forward_status": False,
        "backward_status": False,
        "optimizer_step_status": False,
        "peak_alloc_gb": "N/A",
        "peak_reserved_gb": "N/A",
        "oom_phase": "",
        "note": "",
    }

    model = None
    optimizer = None
    phase = "model_load"

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_id,
            dtype=None,
            max_seq_length=args.seq_len,
            load_in_4bit=True,
            full_finetuning=False,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            use_exact_model_name=True,
            fullgraph=False,
        )
        load_alloc, _, _, _ = mem()
        result["load_vram_gb"] = f"{load_alloc:.3f}"

        phase = "lora_apply"
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_gradient_checkpointing="unsloth",
        )
        lora_alloc, _, _, _ = mem()
        result["lora_vram_gb"] = f"{lora_alloc:.3f}"
        result["trainable_params"] = str(sum(p.numel() for p in model.parameters() if p.requires_grad))

        messages = [{"role": "user", "content": [{"type": "text", "text": "Write a long essay about AI."}]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=args.seq_len,
            add_generation_prompt=False,
        )
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        labels = input_ids.clone()

        if input_ids.shape[1] < args.seq_len:
            pad = args.seq_len - input_ids.shape[1]
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            input_ids = torch.cat([input_ids, torch.full((1, pad), pad_id, device="cuda", dtype=torch.long)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros((1, pad), device="cuda", dtype=torch.long)], dim=1)
            labels = torch.cat([labels, torch.full((1, pad), -100, device="cuda", dtype=torch.long)], dim=1)

        result["input_shape"] = str(tuple(input_ids.shape))

        model.train()
        if hasattr(model, "config"):
            model.config.use_cache = False

        optimizer = bnb.optim.AdamW8bit((pp for pp in model.parameters() if pp.requires_grad), lr=2e-4)
        torch.cuda.reset_peak_memory_stats()

        phase = "forward"
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        _ = out.loss
        result["forward_status"] = True

        phase = "backward"
        out.loss.backward()
        result["backward_status"] = True

        phase = "optimizer_step"
        optimizer.step()
        result["optimizer_step_status"] = True

        _, _, peak_alloc, peak_reserved = mem()
        result["peak_alloc_gb"] = f"{peak_alloc:.3f}"
        result["peak_reserved_gb"] = f"{peak_reserved:.3f}"
        result["STATUS"] = "VERIFIED_OK"

    except torch.OutOfMemoryError as e:
        _, _, peak_alloc, peak_reserved = safe_mem()
        result["peak_alloc_gb"] = peak_alloc
        result["peak_reserved_gb"] = peak_reserved
        result["oom_phase"] = phase
        result["note"] = str(e).replace("\n", " ")[:900]
        result["STATUS"] = "VERIFIED_OOM"
    except Exception as e:
        _, _, peak_alloc, peak_reserved = safe_mem()
        result["peak_alloc_gb"] = peak_alloc
        result["peak_reserved_gb"] = peak_reserved
        result["oom_phase"] = phase
        result["note"] = f"{type(e).__name__}: {e}"
        result["STATUS"] = "VERIFIED_ERROR"
    finally:
        for k, v in result.items():
            print(f"{k}={v}")
        try:
            del optimizer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
