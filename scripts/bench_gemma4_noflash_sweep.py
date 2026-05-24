"""
Sanitized noflash reproduction script.

Context from this lab:
- Isolated noflash SDPA tests worked (VERIFIED_OK).
- Gemma4 actual attention signatures used bf16 q/k/v with D=256 and D=512.
- Gemma4 noflash sweep (r8/r16/r32 seq4096 and r8 seq8192) was VERIFIED_OOM.
"""

import argparse
import gc
import json

import torch
import torch.nn.functional as F


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


def install_sdpa_counter():
    state = {"count": 0, "unique": []}
    seen = set()
    original = F.scaled_dot_product_attention
    original_doc = getattr(original, "__doc__", None)

    def wrapped(q, k, v, *args, **kwargs):
        state["count"] += 1
        sig = {
            "q": tuple(q.shape),
            "k": tuple(k.shape),
            "v": tuple(v.shape),
            "dtype": str(q.dtype),
            "is_causal": bool(kwargs.get("is_causal", False)),
            "requires_grad": bool(
                getattr(q, "requires_grad", False)
                or getattr(k, "requires_grad", False)
                or getattr(v, "requires_grad", False)
            ),
        }
        key = json.dumps(sig, sort_keys=True)
        if key not in seen:
            seen.add(key)
            state["unique"].append(sig)
        return original(q, k, v, *args, **kwargs)

    wrapped.__doc__ = original_doc
    F.scaled_dot_product_attention = wrapped
    return state


def run_case(rank, seq_len, model_id, local_files_only, device_map):
    import bitsandbytes as bnb

    result = {
        "config": f"r{rank}_seq{seq_len}",
        "status": "VERIFIED_ERROR",
        "forward": False,
        "backward": False,
        "optimizer_step": False,
        "oom_phase": "",
        "SDPA_CALL_COUNT": 0,
        "SDPA_SIGNATURES_FIRST20": [],
        "peak_alloc_gb": "N/A",
        "peak_reserved_gb": "N/A",
        "note": "",
    }

    model = None
    optimizer = None
    phase = "import"
    sdpa_state = None

    try:
        # Strategy B: restore doc if noflash patch makes it None before Unsloth import.
        orig_doc = F.scaled_dot_product_attention.__doc__
        import noflash_attention  # noqa: F401

        if F.scaled_dot_product_attention.__doc__ is None:
            F.scaled_dot_product_attention.__doc__ = orig_doc or (
                "scaled_dot_product_attention(query, key, value, attn_mask=None, "
                "dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)"
            )

        from unsloth import FastModel

        sdpa_state = install_sdpa_counter()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        phase = "model_load"
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_id,
            dtype=None,
            max_seq_length=seq_len,
            load_in_4bit=True,
            full_finetuning=False,
            device_map=device_map,
            local_files_only=local_files_only,
            use_exact_model_name=True,
            fullgraph=False,
        )

        phase = "lora_apply"
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=rank,
            lora_alpha=rank,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_gradient_checkpointing="unsloth",
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": "Write a long essay about AI."}]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            add_generation_prompt=False,
        )
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        labels = input_ids.clone()
        if input_ids.shape[1] < seq_len:
            pad = seq_len - input_ids.shape[1]
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            input_ids = torch.cat([input_ids, torch.full((1, pad), pad_id, device="cuda", dtype=torch.long)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros((1, pad), device="cuda", dtype=torch.long)], dim=1)
            labels = torch.cat([labels, torch.full((1, pad), -100, device="cuda", dtype=torch.long)], dim=1)

        model.train()
        if hasattr(model, "config"):
            model.config.use_cache = False
        optimizer = bnb.optim.AdamW8bit((p for p in model.parameters() if p.requires_grad), lr=2e-4)

        torch.cuda.reset_peak_memory_stats()
        phase = "forward"
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        result["forward"] = True

        phase = "backward"
        out.loss.backward()
        result["backward"] = True

        phase = "optimizer_step"
        optimizer.step()
        result["optimizer_step"] = True

        _, _, peak_alloc, peak_reserved = mem()
        result["peak_alloc_gb"] = f"{peak_alloc:.3f}"
        result["peak_reserved_gb"] = f"{peak_reserved:.3f}"
        result["SDPA_CALL_COUNT"] = sdpa_state["count"]
        result["SDPA_SIGNATURES_FIRST20"] = sdpa_state["unique"][:20]
        result["status"] = "VERIFIED_OK" if result["SDPA_CALL_COUNT"] > 0 else "INVALID_TEST"

    except torch.OutOfMemoryError as e:
        _, _, peak_alloc, peak_reserved = safe_mem()
        result["peak_alloc_gb"] = peak_alloc
        result["peak_reserved_gb"] = peak_reserved
        result["oom_phase"] = phase
        result["note"] = str(e).replace("\n", " ")[:900]
        result["status"] = "VERIFIED_OOM"
    except Exception as e:
        _, _, peak_alloc, peak_reserved = safe_mem()
        result["peak_alloc_gb"] = peak_alloc
        result["peak_reserved_gb"] = peak_reserved
        result["oom_phase"] = phase
        result["note"] = f"{type(e).__name__}: {e}"
        result["status"] = "VERIFIED_ERROR"
    finally:
        if sdpa_state is not None:
            result["SDPA_CALL_COUNT"] = sdpa_state["count"]
            result["SDPA_SIGNATURES_FIRST20"] = sdpa_state["unique"][:20]
        if result["peak_alloc_gb"] == "N/A":
            _, _, peak_alloc, peak_reserved = safe_mem()
            result["peak_alloc_gb"] = peak_alloc
            result["peak_reserved_gb"] = peak_reserved
        print("=== RESULT ===")
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


def main():
    p = argparse.ArgumentParser(
        description=(
            "Gemma4 noflash sweep reproducer. WARNING: these configs are expected to be OOM in this lab. "
            "Requires GPU and compatible ROCm stack."
        )
    )
    p.add_argument("--model-id", default="unsloth/gemma-4-31B-it-unsloth-bnb-4bit")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device-map", default="sequential")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CLI/config only. Does not import noflash/Unsloth/bitsandbytes or allocate GPU tensors.",
    )
    args = p.parse_args()

    if args.dry_run:
        print("STATUS=DRY_RUN")
        print(f"model_id={args.model_id}")
        print(f"local_files_only={args.local_files_only}")
        print(f"device_map={args.device_map}")
        print("configs=[(8,4096),(16,4096),(32,4096),(8,8192)]")
        return

    configs = [(8, 4096), (16, 4096), (32, 4096), (8, 8192)]
    for rank, seq_len in configs:
        print(f"\n===== RUN r{rank} seq{seq_len} =====")
        run_case(rank, seq_len, args.model_id, args.local_files_only, args.device_map)


if __name__ == "__main__":
    main()
