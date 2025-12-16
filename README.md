# AI-ML-Assignment7

## Victoria Salomon

---

# Fine-Tuning a Pre-trained Language Model with LoRA for Sentiment Analysis

## Overview
This project fine-tunes a pre-trained DistilBERT model for sentiment analysis on the IMDb dataset. Using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA), the model adapts to the task with fewer trainable parameters, reducing resource requirements.

---

## Dataset
- **Name:** IMDb Movie Review Dataset
- **Task:** Sentiment classification (Positive / Negative)
- **Size:** 25,000 training, 25,000 test samples

---

## Model and Methodology

### Base Model
- **Model:** DistilBERT (`distilbert-base-uncased`)
- **Architecture:** Transformer-based language model

### LoRA Configuration
- **Implementation:** PEFT with LoRA (Low-Rank Adaptation)
- **Parameters:**
  - Rank (`r`): 8
  - Alpha: 32
  - Target modules: `"classifier"` (or as applicable)
- **Purpose:** Reduce the number of trainable parameters, enabling efficient fine-tuning

---

## Training
- Fine-tuned the model on the IMDb training data for **X epochs** (specify your epochs)
- Used a learning rate of **Y** (specify your learning rate)
- Loss function: Cross-Entropy Loss
- Optimizer: AdamW

*(Include training details, e.g., batch size, epochs, hardware used)*

---

## Results

## Inference
Sample predictions on new texts:

| Text | Predicted Sentiment | Confidence |
|-------|---------------------|--------------|
| "This movie was fantastic! I loved it." | Positive | 0.83 |
| "The film was boring and too long." | Negative | 0.94 |

*(Include code snippets or mention that predictions are available in the `predict()` function)*

---

## Conclusion
- LoRA enables efficient fine-tuning of large models with fewer parameters.
- Our model achieved an accuracy of **X.XX**, outperforming the baseline.
- This approach reduces computational resource requirements and training time.

---

## Future Work
- Explore other PEFT methods (e.g., adapters)
- Fine-tune on larger datasets or different tasks
- Visualize training curves and confusion matrix

---

## Dependencies
- `transformers`
- `datasets`
- `peft`
- `scikit-learn`
- `torch`

