# NLP Social Engineering Conversation Classifier

This project explores multi-class classification of social engineering (SE) recruiter-style conversations using both traditional and transformer-based natural language processing techniques.

##  Task Description

Given a synthetic recruiter-target conversation, classify the outcome as:
- `successful`: The target discloses sensitive information or accepts a request.
- `partial`: The target shows mild engagement without sharing critical data.
- `failure`: The target refuses, ignores, or ends the interaction.

This task supports research in phishing prevention and conversational AI safety.

---

##  Dataset

- **2,000 labeled conversations** generated via GPT-style language models
- Multi-turn dialogues that simulate LinkedIn-style recruiter messages
- Each conversation labeled manually as `failure`, `partial`, or `successful`
- File: `diverse_se_conversations_2000.csv`

---

##  Models

### 1. Naive Bayes (Baseline)
- Vectorization: `TF-IDF`
- Classifier: `MultinomialNB`
- Evaluation: Accuracy and weighted F1
- Accuracy: ~52–54% (with cross-validation)

### 2. RoBERTa (Transformer)
- Model: `roberta-base` via Hugging Face Transformers
- Training: 5 epochs, weighted cross-entropy, class balancing
- Evaluation: Accuracy = **85.5%**, F1 = **0.85**

---

## Visual Outputs

- `TrainingLossOverTimeForBert.png`: Loss curve showing RoBERTa convergence
- `NaiveBayesConfMatrix1.png`: Confusion matrix for baseline model
- `trainingLossV2.png`: Updated Bert training loss curve

---

##  Key Files

| File                             | Description                                 |
|----------------------------------|---------------------------------------------|
| `Bert_Classification.py`         | Fine-tuning script for RoBERTa              |
| `NaiveBayes-bagOfWords-2000.py`  | Naive Bayes baseline using TF-IDF           |
| `Final_Project_Writeup_NLP.docx` | Full report writeup                         |
| `synthetic_se_conversations_100.txt` | Sample of generated dialogues          |

---

##  Note on Model Weights

The trained `model.safetensors` file was too large for GitHub (475 MB) and is excluded.  
You can re-train the model using `Bert_Classification.py` or contact the author for access.

---

##  Requirements

```bash
transformers
torch
scikit-learn
pandas
evaluate
matplotlib
