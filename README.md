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
3. **Apply LoRA adapters** for parameter-efficient fine-tuning
4. **Wrap with a PPO-compatible value head**
5. Generate summaries → compute toxicity with Detoxify
6. Use **PPO** to reward less-toxic outputs
7. Repeat over 50 steps on 500 dialogue samples


---

**Here's everything we did, step by step:**

**🛠️ 1. Setting Things Up:**

We started by installing and importing everything we needed — Hugging Face’s transformers, trl for PPO, peft for LoRA, datasets for the SAMSum dataset, and Detoxify to score toxicity.

**🧠 2. Loading the Model & Adding LoRA:**

We used google/flan-t5-base as the base summarization model.
Instead of fine-tuning the whole thing, we added LoRA adapters — which makes training way faster and more memory-efficient.

Then we wrapped the model with a value head so it could learn from reward signals using PPO.

**📚 3. Preparing the Dataset:**

We used the SAMSum dataset, which is full of everyday conversations and summaries.
We cleaned and tokenized the data and batched it using a data loader to feed it into the model during training.

**🎯 4. Defining the Reward Function:**

Here’s the cool part — instead of using traditional loss, we let the model learn from feedback.
We used Detoxify to measure how toxic a generated summary was, and gave higher rewards to summaries that were more respectful and neutral.

Reward = 1 - toxicity score

**⚙️ 5. Setting Up PPO:**

We configured PPO with a small learning rate and batch size, and connected it to our LoRA-augmented model. This let the model optimize its behavior based on the reward scores from Detoxify.

**🔁 6. Fine-Tuning with PPO:**

This is where the magic happened:

The model generated summaries from dialogues.

We filtered out any empty or super short ones.

Then we checked how toxic they were.

The PPO trainer used this info to improve the model over time.

We ran this loop for 50 training steps.

**📊 7. Evaluating the Results:**

To see if it worked, we compared the toxicity of the generated summaries to the original human-written ones.

---

## 📊 Results

| Metric             | Original Summaries | Fine-Tuned Model |
|--------------------|--------------------|------------------|
| 🤬 Avg Toxicity     | 0.0096             | **0.0020** ✅     |

> A 5x reduction in average toxicity without any direct supervision.


