import openai
import spacy
from flask import Flask, json, request, Response, jsonify
from flask_cors import CORS
import os
import pytesseract
import cv2
import numpy as np
import re
from PIL import Image
from pdf2image import convert_from_path
from datetime import datetime
from collections import defaultdict
from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv
import pytesseract
from werkzeug.utils import secure_filename
import uuid
import extract_msg
import shutil


app = Flask(__name__)
CORS(app)

load_dotenv()

nlp = spacy.load("en_core_web_sm")

openai.api_key = os.getenv("OPENAI_API_KEY")

credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Check if the environment variable is set correctly
if credentials_json is None:
    raise ValueError(
        "The environment variable 'GOOGLE_APPLICATION_CREDENTIALS_JSON' is not set.")

# Convert the JSON string to a Python dictionary
credentials_info = json.loads(credentials_json)

# Create credentials from the dictionary
credentials = service_account.Credentials.from_service_account_info(
    credentials_info)

# Initialize Google Vision client
client = vision.ImageAnnotatorClient(credentials=credentials)

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# print("Tesseract path:", pytesseract.pytesseract.tesseract_cmd)
# print("Is Tesseract installed?", os.path.exists(
#     pytesseract.pytesseract.tesseract_cmd))


def correct_image_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angle = 0
    if lines is not None:
        for rho, theta in lines[0]:
            angle = np.degrees(theta) - 90
            break

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image


def clean_temp_folder(folder_path):
    try:
        # Remove all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Temporary folder cleaned.")
    except Exception as e:
        print(f"Error cleaning temporary folder: {e}")


def process_image_with_vision(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description
    else:
        return ""


def process_image(image_path):
    # img = cv2.imread(image_path)
    # img = correct_image_rotation(img)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # denoised = cv2.fastNlMeansDenoising(gray)
    # _, binary = cv2.threshold(
    #     denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    vision_text = process_image_with_vision(image_path)

    if vision_text.strip():
        return vision_text
    else:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image_path, config=custom_config)
        return text


def process_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text

def process_msg(msg_path):
    text = ""
    try:
        # Use extract_msg to read the MSG file
        msg = extract_msg.Message(msg_path)
        text += f"Subject: {msg.subject}\n"
        text += f"Date: {msg.date}\n"
        text += f"Sender: {msg.sender}\n"
        text += f"To: {msg.to}\n\n"
        text += f"Body:\n{msg.body}\n"
        msg.close()
    except Exception as e:
        print(f"Error processing MSG: {e}")
    return text

def process_file(file_path):
    _, ext = os.path.splitext(file_path.lower())
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return process_image(file_path)
    elif ext == '.pdf':
        return process_pdf(file_path)
    elif ext == '.msg':
        return process_msg(file_path)
    else:
        return f"Unsupported file type: {file_path}"


def refine_text_with_spacy(ocr_text):
    """
    Refine the OCR output to remove noise and extract structured data.
    """
    # Pre-clean the text
    ocr_text = re.sub(r"[✓✔]+", "", ocr_text)  # Remove tick marks
    ocr_text = re.sub(r"[←→«»]+", "", ocr_text)  # Remove arrow symbols
    # Remove "Edited" timestamps
    ocr_text = re.sub(r"\bEdited\b.*\b\d{1,2}:\d{2} (?:AM|PM)", "", ocr_text)
    # Remove standalone timestamps
    ocr_text = re.sub(r"\b\d{1,2}:\d{2} (?:AM|PM)\b", "", ocr_text)
    # Remove network indicators
    ocr_text = re.sub(r"\b[4G|LTE]+\b", "", ocr_text)
    # Remove extra spaces and line breaks
    ocr_text = re.sub(r"\s+", " ", ocr_text).strip()

    # Remove duplicates
    lines = ocr_text.splitlines()
    seen_lines = set()
    deduplicated_lines = []
    for line in lines:
        if line not in seen_lines:
            deduplicated_lines.append(line)
            seen_lines.add(line)

    # Join deduplicated lines
    deduplicated_text = "\n".join(deduplicated_lines)

    # Process text with spaCy
    doc = nlp(deduplicated_text)

    # Extract dates using Named Entity Recognition
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    # Join cleaned tokens
    cleaned_text = " ".join([token.text for token in doc])

    return {
        "cleaned_text": cleaned_text.strip(),
        "dates": dates
    }


def refine_text_with_chatgpt(refined_data):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant that refines OCR text and writes a summary in the end. The text can be of many different platforms like teams, gmail, slack, etc. Remove the noise accordingly. Return the text as well as the summary."
            }, {
                "role": "user",
                "content": f"Refine the following text imporve ocr and improve clarity:\n\n{refined_data}"
            }],
            max_tokens=1000
        )

        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error refining text with ChatGPT: {e}")
        return refined_data


def extract_dates_with_chatgpt(refined_data):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant that extracts dates from the ocr text. There might be multiple dates, find the most appropriate one. Only return the date in dd-mm-yyyy format."
            }, {
                "role": "user",
                "content": f"Find the date from this data :\n\n{refined_data}. And return only the date in dd-mm-yyyy format, no other text, nothing. The output should only be of 10 characters or less."
            }],
            max_tokens=1000
        )

        # print(response['choices'][0]['message']['content'].strip())

        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        print(f"Error refining date with ChatGPT: {e}")
        return refined_data


def process_folder(folder_path):
    text_by_date = defaultdict(list)
    total_files = sum([len(files) for r, d, files in os.walk(folder_path)])
    processed_files = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            processed_files += 1
            progress = int((processed_files / total_files) * 100)

            text = process_file(file_path)

            refined_data = refine_text_with_spacy(text)

            refined_data_with_chatgpt = refine_text_with_chatgpt(refined_data)

            date = extract_dates_with_chatgpt(refined_data_with_chatgpt)
            # date = dates[0] if dates else "No Date"

            print(date)

            text_by_date[date].append((file_path, refined_data_with_chatgpt))

            print(f"{progress}|{date}|{file_path}|{refined_data_with_chatgpt}")

            yield f"{progress}|{date}|{file_path}|{refined_data_with_chatgpt}"

    def parse_date(date_str):
        try:
            # First, try parsing as "%B %d, %Y" (e.g., "August 19, 2024")
            return datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            pass

        try:
            # If that fails, try parsing as "%d/%m/%Y" (e.g., "19/08/2024")
            return datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            pass

        # If both fail, return None
        return None

    # After processing all files, yield the sorted results
    sorted_dates = sorted(
        text_by_date.keys(),
        key=lambda x: parse_date(x) if parse_date(
            x) else datetime.max  # Treat None as max date
    )

    # for date in sorted_dates:
    #     yield f"100|RESULT|{date}|" + "|".join([f"{file_path}:{text}" for file_path, text in text_by_date[date]])


@app.route('/api/process-ocr', methods=['POST'])
def process_ocr():
    print("Received POST request to /api/process-ocr")

    if request.method == 'POST':
        folder_path = request.form.get('folderPath', None)
        if not folder_path:
            return jsonify({"error": "No folder path provided"}), 400

            # Ensure there are files in the request
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided"}), 400

            # Create a directory to save the files temporarily
        temp_folder = 'temp'  # Relative path

# Ensure the directory exists
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
            print("Created temporary folder:", temp_folder)

        for file in files:
            if not file.filename:
                return jsonify({"error": "One of the uploaded files has an empty filename"}), 400

            # Sanitize and make the filename unique
            sanitized_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{sanitized_filename}"
            file_path = os.path.join(temp_folder, unique_filename)

            try:
                file.save(file_path)
                print(f"Saved file: {unique_filename} at {file_path}")
            except Exception as e:
                print(
                    f"Error saving file {unique_filename} to {file_path}: {e}")
                return jsonify({"error": f"Failed to save file {unique_filename}"}), 500

            # Now process the files (implement your OCR processing)
        def generate():
            try:
                for result in process_folder(temp_folder):
                    # Process the file (this is just an example; replace with actual OCR logic)
                    yield f"data: {result}\n\n"
            finally:
                # Clean up the temporary folder after processing
                clean_temp_folder(temp_folder)

            # Return the streaming response with OCR results
        return Response(generate(), mimetype='text/event-stream')

    return jsonify({"error": "Method not allowed"}), 405


@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running"}), 200


if __name__ == '__main__':
    print("Starting Flask server on port 8000")
    # Default to 5000 if no PORT is set
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
