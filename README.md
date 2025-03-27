# 🚀 Detoxifying Summarization with FLAN-T5 + LoRA + PPO

This project fine-tunes Google's FLAN-T5 using **LoRA adapters** and **Reinforcement Learning (PPO)** to generate **less-toxic summaries** from dialogue data.

We use the [SAMSum dataset](https://huggingface.co/datasets/samsum) for training, and the [Detoxify](https://github.com/unitaryai/detoxify) model as a reward signal during RL.

---

## ✨ Key Highlights

- 🧠 **FLAN-T5** as base summarization model
- 🔧 **LoRA** (Low-Rank Adaptation) for lightweight fine-tuning
- 🎯 **PPO (Proximal Policy Optimization)** reinforcement learning
- 🔍 **Detoxify** reward model to reduce toxicity
- 🗣️ Based on the real-world SAMSum dialogue summarization dataset
- 📉 Achieved **5x lower average toxicity** than original dataset summaries!

---

## 💡 Motivation

Even high-quality language models can generate biased or toxic outputs.  
This project explores how to **fine-tune an LLM to reduce toxicity** while preserving summary quality — using reward-based learning instead of static loss functions.

---

## 🛠️ Tech Stack

- `transformers==4.28.1`
- `trl==0.4.7` (Hugging Face RLHF library)
- `peft==0.2.0` (for LoRA)
- `datasets`, `detoxify`, `accelerate`

---

## 📦 Dataset

We use the **[SAMSum dataset](https://huggingface.co/datasets/samsum)**, which contains thousands of casual human dialogues and professionally written summaries.

---

## 🧪 Training Pipeline

1. **Start with FLAN-T5** (`google/flan-t5-base`)
2. **Apply LoRA adapters** for parameter-efficient fine-tuning
3. **Wrap with a PPO-compatible value head**
4. Generate summaries → compute toxicity with Detoxify
5. Use **PPO** to reward less-toxic outputs
6. Repeat over 50 steps on 500 dialogue samples

---

## 📊 Results

| Metric             | Original Summaries | Fine-Tuned Model |
|--------------------|--------------------|------------------|
| 🤬 Avg Toxicity     | 0.0096             | **0.0020** ✅     |

> A 5x reduction in average toxicity without any direct supervision.


