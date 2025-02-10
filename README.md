# text_summarization
# Text Summarization using T5 Model

## 📌 Overview
This project implements **text summarization** using the **T5 (Text-To-Text Transfer Transformer) model**. The model is trained and fine-tuned to generate concise summaries from input text using **PyTorch Lightning** and **Transformers library**.

## ⚙️ Setup and Installation
### 1️⃣ Install Dependencies
```sh
pip install --quiet transformers pytorch-lightning torch pandas numpy scikit-learn keras matplotlib seaborn tqdm
```

### 2️⃣ Load the Dataset
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('news_summary.csv', encoding='latin-1')
df = df[['text', 'ctext']]
df.columns = ["summary", "text"]
df.dropna(inplace=True)
train_df, test_df = train_test_split(df, test_size=0.1)
```

### 3️⃣ Initialize the T5 Model
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
```

## 🚀 Training and Summarization
### 1️⃣ Training the Model
```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

trainer = Trainer(max_epochs=5, callbacks=[ModelCheckpoint(dirpath='models/', save_top_k=1)])
trainer.fit(model)
```

### 2️⃣ Generate Summaries
```python
def generate_summary(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

sample_text = "Your long text here..."
summary = generate_summary(sample_text)
print(summary)
```

## 📊 Key Features
- **Uses T5 Model**: A powerful transformer for text-to-text tasks.
- **PyTorch Lightning Integration**: Simplifies model training and checkpointing.
- **Customizable Hyperparameters**: Fine-tune text summarization performance.
- **Handles Large Text Inputs**: Efficient text processing with tokenization.

## 📌 Contributors
- **Project Lead**: [Quoc Hung]
- **Instructor**: [Assoc. Prof. Dr. Le Anh Cuong]

## 📝 License
This project is for educational and research purposes only.

## 📬 Contact
For any inquiries, please reach out to [beaufull2002@gmail.com].


