# ocr-ner-pipeline

---

# 🧠 OCR + NER Pipeline for Handwritten Text

## 📌 Project Overview

This project is an end-to-end pipeline that extracts structured information from handwritten images.

It converts images into text using OCR, cleans the text, and applies Named Entity Recognition (NER) to extract meaningful entities such as:

* 👤 Persons
* 📍 Locations
* 📅 Dates

Finally, it visualizes the results using an interactive Streamlit dashboard.

---

# 🚀 Features

* 📷 Handwritten image input
* 🔍 OCR text extraction (Tesseract)
* 🧹 Text cleaning & correction
* 🧠 Named Entity Recognition (spaCy)
* 📊 Interactive Streamlit dashboard
* 📁 Batch processing from folder
* 📥 CSV export of results

---

# 🏗️ Pipeline Architecture

```
Image
  ↓
OCR (Tesseract)
  ↓
Text Cleaning
  ↓
NER (spaCy)
  ↓
Structured Output
  ↓
Streamlit Dashboard
```

---

# 🧪 Example Output

### Input Image Text:

```
I met Ahmed in Amman on Monday
```

### Extracted Entities:

* Ahmed → PERSON
* Amman → GPE
* Monday → DATE

---

# 🛠️ Tech Stack

* Python 🐍
* Tesseract OCR 🔍
* spaCy NLP 🧠
* OpenCV 🖼️
* Streamlit 🌐
* Pandas 📊

---

# 📁 Project Structure

```
project/
│
├── data/                  # Handwritten images
├── app_streamlit.py       # Streamlit dashboard
├── main.py                # OCR + NER pipeline (CLI version)
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone repository

```bash
git clone https://github.com/your-username/ocr-ner-project.git
cd ocr-ner-project
```

## 2. Create virtual environment

```bash
python -m venv venv
source venv\Scripts\activate   # Windows
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

---

# ▶️ Run Project

## Run Streamlit Dashboard

```bash
streamlit run app_streamlit.py
```

## Run CLI Pipeline

```bash
python main.py
```

---

# 📊 Dashboard Features

* Upload image OR process batch folder
* View extracted text
* View detected entities
* Download results as CSV
* View statistics (entity distribution)

---

# ⚠️ Limitations

* OCR accuracy depends on handwriting quality
* Some characters may be misread
* spaCy model is not trained specifically on handwritten text

---

# 💡 Future Improvements

* Fine-tuned handwriting OCR model
* Transformer-based NER (BERT-based)
* Better noise handling
* Image preprocessing enhancement
* Deployment to cloud (Streamlit Cloud / HuggingFace Spaces)

---

# 🎯 Use Cases

* Document digitization
* Form processing systems
* Healthcare records extraction
* Banking document analysis

---