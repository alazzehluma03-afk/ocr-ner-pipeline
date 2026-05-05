import pytesseract
import spacy
import cv2
import os
import re

# ================================
# 1. Tesseract Path
# ================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ================================
# 2. Load spaCy Model
# ================================
nlp = spacy.load("en_core_web_sm")

# ================================
# 3. Text Cleaning (SAFE)
# ================================
def clean_text(text):
    text = text.replace("_", " ")
    text = re.sub(r'[^\w\s]', '', text)  # remove symbols only
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ================================
# 4. OCR Error Correction (lightweight)
# ================================
def correct_common_errors(text):
    corrections = {
        "T met": "I met",
        "Moncay": "Monday",
        "Pacis": "Paris",
        "meeking": "meeting",
        "withh": "with"
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    return text

# ================================
# 5. Entity Fix (simple rule boost)
# ================================
def fix_entities(doc):
    fixed = []

    for ent in doc.ents:
        label = ent.label_

        # force correct PERSON detection for known names
        if ent.text.lower() in ["sara", "ahmed"]:
            label = "PERSON"

        fixed.append((ent.text, label))

    return fixed

# ================================
# 6. Pipeline
# ================================
folder_path = 'data'

print("\n--- Starting OCR & NER Pipeline ---")

if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' not found!")

else:
    for filename in os.listdir(folder_path):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            img_path = os.path.join(folder_path, filename)

            print(f"\n[ File: {filename} ]")

            # ============================
            # OCR preprocessing
            # ============================
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # ============================
            # OCR
            # ============================
            raw_text = pytesseract.image_to_string(thresh)

            # ============================
            # Cleaning pipeline (IMPORTANT ORDER)
            # ============================
            cleaned = clean_text(raw_text)
            corrected = correct_common_errors(cleaned)

            print(f"Raw Text: {raw_text.strip()}")
            print(f"Final Text: {corrected}")

            # skip empty/noisy outputs
            if len(corrected.strip()) < 5:
                print("⚠️ OCR failed or text too noisy")
                continue

            # ============================
            # NER
            # ============================
            doc = nlp(corrected)
            entities = fix_entities(doc)

            print("Detected Entities:")

            if not entities:
                print("No entities found.")

            for text, label in entities:
                print(f" - Text: {text} | Label: {label}")

print("\n--- Pipeline Process Finished ---")