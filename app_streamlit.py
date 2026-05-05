import streamlit as st
import pytesseract
import spacy
import cv2
import os
import numpy as np
import pandas as pd

# =========================
# Setup
# =========================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
nlp = spacy.load("en_core_web_sm")

DATA_FOLDER = "data"

st.set_page_config(page_title="OCR + NER Pipeline", layout="wide")

st.title("🧠 OCR + NER Pipeline Dashboard")
st.write("Handwritten Images → OCR → Named Entity Recognition")

# =========================
# Helpers
# =========================
def clean_text(text):
    text = text.replace("_", " ")
    return text.strip()

def correct_text(text):
    return text.replace("Moncay", "Monday").replace("Pacis", "Paris")

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    text = clean_text(text)
    text = correct_text(text)

    doc = nlp(text)

    return text, doc

# =========================
# Batch Processing
# =========================
st.header("📁 Batch Processing (data folder)")

results = []

if st.button("🚀 Run Pipeline on All Images"):

    if not os.path.exists(DATA_FOLDER):
        st.error("Data folder not found!")
    else:
        files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(("png","jpg","jpeg"))]

        if not files:
            st.warning("No images found!")
        else:

            for file in files:
                path = os.path.join(DATA_FOLDER, file)
                image = cv2.imread(path)

                text, doc = process_image(image)

                entities = [(ent.text, ent.label_) for ent in doc.ents]

                if not entities:
                    entities = [("None", "None")]

                for ent_text, ent_label in entities:
                    results.append({
                        "image": file,
                        "text": text,
                        "entity": ent_text,
                        "label": ent_label
                    })

            df = pd.DataFrame(results)

            st.success("Pipeline Completed!")

            # =========================
            # IMAGE PREVIEW + RESULTS
            # =========================
            st.subheader("📊 Results Table")
            st.dataframe(df, use_container_width=True)

            # =========================
            # DOWNLOAD CSV
            # =========================
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "📥 Download Results CSV",
                csv,
                "ner_results.csv",
                "text/csv"
            )

            # =========================
            # STATS DASHBOARD
            # =========================
            st.subheader("📈 Entity Statistics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Images", len(files))
            col2.metric("Total Entities", len(df))
            col3.metric("Unique Labels", df["label"].nunique())

            # Label distribution chart
            st.bar_chart(df["label"].value_counts())

# =========================
# SINGLE IMAGE MODE
# =========================
st.header("📤 Test Single Image")

uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded is not None:

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    text, doc = process_image(image)

    st.subheader("📄 Extracted Text")
    st.write(text)

    st.subheader("🏷️ Entities")

    if doc.ents:
        for ent in doc.ents:
            st.write(f"**{ent.text}** → {ent.label_}")
    else:
        st.write("No entities found")