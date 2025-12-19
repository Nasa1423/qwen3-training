# Qwen3-4B Fine-tuning

Fine-tuning Qwen3-4B (и других Qwen3 моделей) на диалоговых данных с поддержкой LoRA и масштабирования на несколько GPU.

## Возможности

- **Full Fine-tuning** — полное дообучение модели
- **LoRA/QLoRA** — эффективное дообучение с низким потреблением памяти
- **Multi-GPU** — Accelerate + DeepSpeed ZeRO для масштабирования
- **Inference** — интерактивный и пакетный режимы

## Установка

```bash
# Установка зависимостей через uv
uv sync

# Или через pip
pip install -e .
```

## Формат данных

JSON файл со списком диалогов:

```json
[
  {
    "question": "Вопрос пользователя",
    "answer": "Ответ ассистента",
    "toxicity_score": 0.95
  }
]
```

## Использование

### Single GPU — LoRA (рекомендуется)

```bash
# LoRA обучение (потребляет ~16GB VRAM)
python train_lora.py --data_path ./data/dialogs.json

# QLoRA 4-bit (потребляет ~8GB VRAM)
python train_lora.py --data_path ./data/dialogs.json --use_4bit
```

### Single GPU — Full Fine-tune

```bash
# Полное обучение (требует много VRAM)
python train_full.py --data_path ./data/dialogs.json
```

### Multi-GPU с Accelerate

```bash
# Сгенерировать конфиг (интерактивно)
accelerate config

# Или использовать готовый конфиг
accelerate launch --config_file configs/accelerate_config.yaml \
    train_lora.py --data_path ./data/dialogs.json
```

### Multi-GPU с DeepSpeed

```bash
# ZeRO Stage 2 (рекомендуется для 2-4 GPU)
accelerate launch --config_file configs/accelerate_config.yaml \
    --deepspeed_config_file configs/deepspeed_zero2.json \
    train_lora.py --data_path ./data/dialogs.json

# ZeRO Stage 3 + CPU Offload (для ограниченной VRAM)
accelerate launch --config_file configs/accelerate_config.yaml \
    --deepspeed_config_file configs/deepspeed_zero3_offload.json \
    train_full.py --data_path ./data/dialogs.json
```

### torchrun (альтернатива)

```bash
# 2 GPU
torchrun --nproc_per_node=2 train_lora.py --data_path ./data/dialogs.json

# 4 GPU
torchrun --nproc_per_node=4 train_lora.py --data_path ./data/dialogs.json
```

## Inference

```bash
# Интерактивный режим
python inference.py --model_path ./output/lora/final --interactive

# С LoRA адаптером
python inference.py --model_path Qwen/Qwen3-4B --adapter_path ./output/lora/final

# Один промпт
python inference.py --model_path ./output/full/final --prompt "Привет!"

# Пакетная обработка
python inference.py --model_path ./output/full/final \
    --input_file questions.txt --output_file answers.json
```

## Объединение LoRA с базовой моделью

```bash
python train_lora.py --merge_adapter \
    --adapter_path ./output/lora/final \
    --merged_output ./output/merged
```

## Масштабирование на несколько GPU

### Таблица рекомендаций

| Метод | GPU Memory | Кол-во GPU | Рекомендуемый конфиг |
|-------|------------|------------|---------------------|
| LoRA | 16GB+ | 1 | `train_lora.py` |
| QLoRA 4-bit | 8GB+ | 1 | `--use_4bit` |
| LoRA Multi-GPU | 16GB+ × N | 2-8 | Accelerate DDP |
| Full Fine-tune | 40GB+ | 1 | `train_full.py` |
| Full + ZeRO2 | 24GB+ × N | 2-4 | DeepSpeed ZeRO2 |
| Full + ZeRO3 | 16GB+ × N | 4-8 | DeepSpeed ZeRO3 + CPU offload |

### Настройка Accelerate

Отредактируйте `configs/accelerate_config.yaml`:

```yaml
num_processes: 4  # Количество GPU
```

### Примеры команд для разных конфигураций

```bash
# 2x RTX 4090/5090 — LoRA
accelerate launch --num_processes=2 train_lora.py --data_path ./data/dialogs.json

# 4x A100 — Full Fine-tune с ZeRO2  
accelerate launch --num_processes=4 \
    --deepspeed_config_file configs/deepspeed_zero2.json \
    train_full.py --data_path ./data/dialogs.json

# 8x RTX 3090 — Full Fine-tune с ZeRO3 + CPU Offload
accelerate launch --num_processes=8 \
    --deepspeed_config_file configs/deepspeed_zero3_offload.json \
    train_full.py --data_path ./data/dialogs.json \
    --gradient_accumulation 4
```

## Параметры обучения

### LoRA

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--lora_r` | 64 | Ранг LoRA |
| `--lora_alpha` | 128 | Alpha LoRA |
| `--lora_dropout` | 0.05 | Dropout |
| `--learning_rate` | 2e-4 | Скорость обучения |

### Общие

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--epochs` | 3 | Количество эпох |
| `--batch_size` | 1-2 | Batch size на GPU |
| `--gradient_accumulation` | 4-8 | Накопление градиентов |
| `--max_length` | 2048 | Макс. длина последовательности |

## Структура проекта

```
qwen-training-modern/
├── pyproject.toml          # Зависимости
├── config.py               # Конфигурация (dataclasses)
├── data_loader.py          # Загрузка данных
├── train_full.py           # Полное обучение
├── train_lora.py           # LoRA обучение
├── inference.py            # Инференс
├── configs/
│   ├── accelerate_config.yaml
│   ├── deepspeed_zero2.json
│   └── deepspeed_zero3_offload.json
└── README.md
```

## Логирование

```bash
# TensorBoard (по умолчанию)
tensorboard --logdir ./output

# Weights & Biases
python train_lora.py --data_path ./data/dialogs.json --report_to wandb
```
