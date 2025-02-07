from docx import Document
import openai
from flask import Flask, json, request, Response, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import re
from datetime import datetime
from collections import defaultdict
from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import extract_msg
import gc

import fitz

app = Flask(__name__)
CORS(app)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if credentials_json is None:
    raise ValueError(
        "The environment variable 'GOOGLE_APPLICATION_CREDENTIALS_JSON' is not set.")

credentials_info = json.loads(credentials_json)
credentials = service_account.Credentials.from_service_account_info(
    credentials_info)
client = vision.ImageAnnotatorClient(credentials=credentials)


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
    vision_text = process_image_with_vision(image_path)

    if vision_text.strip():
        return vision_text


def process_pdf(pdf_path, chunk_size=5):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    text = ""

    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        chunk_text = ""

        for i in range(start_page, end_page):
            page = doc[i]
            page_text = page.get_text()
            cleaned_result = clean_text(page_text)
            cleaned_result = normalize_text(cleaned_result)
            chunk_text += f"\r\n{'='*40}\r\nPage {i+1}\r\n{'='*40}\r\n\n{cleaned_result}\r\n"

        text += chunk_text

        # Yield progress and chunk text
        progress = int((end_page / total_pages) * 100)
        yield progress, chunk_text

    doc.close()
    return text


def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common OCR artifacts
    text = re.sub(r'[|]', '', text)
    # Merge split words
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Add line breaks after sentences
    text = re.sub(r'(\.\s)([A-Z])', r'\1\n\2', text)
    return text


def normalize_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove unnecessary spaces around punctuation
    text = re.sub(r'\s([.,:;?!])', r'\1', text)
    # Ensure line breaks after headers and sections
    text = re.sub(r'(?<=\.)\s*(?=\S)', '\n', text)
    return text.strip()


def process_msg(msg_path):
    text = ""
    try:
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


def process_doc(doc_path):
    doc = Document(doc_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def process_file(file_path):
    _, ext = os.path.splitext(file_path.lower())
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return process_image(file_path)
    elif ext == '.pdf':
        return process_pdf(file_path)
    elif ext == '.msg':
        return process_msg(file_path)
    elif ext in ['.doc', '.docx']:
        return process_doc(file_path)
    else:
        return f"Unsupported file type: {file_path}"


def refine_text_with_chatgpt(refined_data):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Improve the text. Try to be verbatim."
            }, {
                "role": "user",
                "content": f"Refine the text verbatim.\n\n{refined_data}"
            }],
            max_tokens=2000
        )

        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error refining text with ChatGPT: {e}")
        return refined_data
    finally:
        del response
        gc.collect()


def extract_date_with_regex(text):
    # This regex pattern looks for common date formats
    date_patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # dd/mm/yyyy or dd-mm-yyyy
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # yyyy/mm/dd or yyyy-mm-dd
        # Month DD, YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',
        # DD Month YYYY
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b',
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()

    return "No date found"


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
            max_tokens=2000
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
            refined_data_with_chatgpt = ""
            final_text = ""

            if file_path.lower().endswith('.pdf'):

                date = "No date found"

                doc = fitz.open(file_path)
                total_pages = len(doc)

                if (total_pages < 5):
                    for progress, text in process_pdf(file_path):

                        refined_data_with_chatgpt = refine_text_with_chatgpt(
                            text)

                        final_text += refined_data_with_chatgpt

                    if date == "No date found":
                        date = extract_dates_with_chatgpt(
                            final_text[:500])

                    # Ensure final_text is complete before yielding
                    yield f"{progress}|{date}|{file_path}|{final_text}\n\n"
                  # Initialize date

                else:
                    for progress, text in process_pdf(file_path):

                        final_text += text

                    if date == "No date found":
                        date = extract_dates_with_chatgpt(
                            final_text[:500])

                    yield f"{progress}|{date}|{file_path}|{final_text}\n\n"

            else:
                text = process_file(file_path)
                word_count = len(text.split())

                print(f"Word count: {word_count}")

                if word_count <= 1000:
                    refined_data = text
                    refined_data_with_chatgpt = refine_text_with_chatgpt(
                        refined_data)
                    date = extract_dates_with_chatgpt(
                        refined_data_with_chatgpt)
                else:
                    refined_data = text
                    refined_data_with_chatgpt = text
                    date = extract_date_with_regex(text.split('\n', 1)[0])

                text_by_date[date].append(
                    (file_path, refined_data_with_chatgpt))

                yield f"{progress}|{date}|{file_path}|{refined_data_with_chatgpt}\n\n"

            if (refined_data_with_chatgpt is not None):
                del refined_data_with_chatgpt
            if (final_text is not None):
                del final_text
            gc.collect()

    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            pass

        try:
            return datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            pass

        return None

    sorted_dates = sorted(
        text_by_date.keys(),
        key=lambda x: parse_date(x) if parse_date(x) else datetime.max
    )


@app.route('/api/process-ocr', methods=['POST'])
def process_ocr():
    print("Received POST request to /api/process-ocr")

    if request.method == 'POST':
        folder_path = request.form.get('folderPath', None)
        if not folder_path:
            return jsonify({"error": "No folder path provided"}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided"}), 400

        temp_folder = 'temp'

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
            print("Created temporary folder:", temp_folder)

        for file in files:
            if not file.filename:
                return jsonify({"error": "One of the uploaded files has an empty filename"}), 400

            sanitized_filename = secure_filename(file.filename)
            unique_filename = f"{sanitized_filename}"
            file_path = os.path.join(temp_folder, unique_filename)

            try:
                file.save(file_path)
                print(f"Saved file: {unique_filename} at {file_path}")
            except Exception as e:
                print(
                    f"Error saving file {unique_filename} to {file_path}: {e}")
                return jsonify({"error": f"Failed to save file {unique_filename}"}), 500

        def generate():
            try:
                for result in process_folder(temp_folder):
                    yield f"data: {result}\n\n"
            finally:
                clean_temp_folder(temp_folder)

        return Response(generate(), mimetype='text/event-stream')

    return jsonify({"error": "Method not allowed"}), 405


@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running"}), 200


if __name__ == '__main__':
    print("Starting Flask server on port 8000")
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
