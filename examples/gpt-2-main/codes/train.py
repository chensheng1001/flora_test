import torch
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    Adafactor,
    )
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from flora_opt import Flora  # 你自定义的优化器
import sys
from transformers import TrainerCallback

# === 实时监控显存和优化器 ===
class GpuMonitorCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)

        optimizer_name = "Unknown"
        if trainer is not None:
            # 有些优化器只有在 trainer.train() 执行中才会创建
            try:
                # create_optimizer 不是公共API，但 Trainer 内部依赖该方法创建优化器
                if trainer.optimizer is None:
                    trainer.create_optimizer()
                optimizer_name = type(trainer.optimizer).__name__
            except Exception as e:
                optimizer_name = f"Error({str(e)})"

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(
                f"[监控] Step {state.global_step} | 显存: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB | "
                f"优化器: {optimizer_name}"
            )


# === 可选优化器的Trainer ===
class CustomTrainer(Trainer):
    def __init__(self, *args, custom_optim=None, **kwargs):
        self.custom_optim = custom_optim
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            if self.custom_optim == "flora":
                self.optimizer = Flora(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    rank=128,  # 你自定义的参数
                )
            elif self.custom_optim == "adafactor":
                self.optimizer = Adafactor(
                    self.model.parameters(),
                    scale_parameter=True,
                    relative_step=False,
                    warmup_init=False,
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                )
            elif self.custom_optim == "adamw":
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                )
            else:
                raise ValueError(f"未知优化器类型: {self.custom_optim}")
        return self.optimizer

# === 预处理 ===
def preprocess(dataset, tokenizer):
    lines = [line for line in dataset['text'] if line.strip()]
    encodings = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=512)
    return [{"input_ids": torch.tensor(ids, dtype=torch.long)} for ids in encodings["input_ids"]]

# === 主函数 ===
def train(model_name, output_dir, overwrite_output_dir, per_device_train_batch_size,
          num_train_epochs, save_steps, optim_type="flora"):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_dataset = preprocess(dataset, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 可选开启 LoRA 支持（如使用 PEFT）
    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(r=128, lora_alpha=16, lora_dropout=0.1, task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=50,
        report_to=[],
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[GpuMonitorCallback()],
        custom_optim=optim_type
    )

    print(f"[启动训练] 当前使用优化器: {optim_type}")
    trainer.train()
    trainer.save_model()
    torch.cuda.empty_cache()

# === 启动 ===
if __name__ == "__main__":
    train(
        model_name="/mnt/self-define/Xinbz/models/llms/gpt2",
        output_dir="./ckpts/adafactor_128/",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=1000,
        optim_type="adafactor"  # 可选："adamw", "adafactor", "flora"
    )
