"""Experimental GPU QLoRA recipe; NOT YET verified end-to-end on Colab."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--out", default="outputs/qwen15b-qlora")
    p.add_argument("--epochs", type=float, default=2)
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this 4-bit training recipe")

    dataset = load_dataset("json", data_files={"train": args.train, "validation": args.validation})
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=cfg, device_map="auto"
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    train_cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        fp16=True,
        bf16=False,
        max_length=512,
        completion_only_loss=True,
        gradient_checkpointing=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=train_cfg,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print("Saved adapter to", args.out)
    print("Next: evaluate on the held-out test set with fixed generation parameters")


if __name__ == "__main__":
    main()
