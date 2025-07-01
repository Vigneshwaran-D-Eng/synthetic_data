# core_logic_new.py
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import namedtuple
from typing import Optional
import os
# import instructor
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json

# Azure and OpenAI SDKs
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient, DocumentAnalysisClient

# Local configuration
import config
# --- Application Configuration from Environment Variables ---
# LLM related configurations
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("API_VERSION") # Note: Env var name is API_VERSION
GPT_MODEL_NAME = os.getenv("GPT_MODEL_4O", "gpt-4o") # Default to gpt-4o if not set
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS_DEFAULT = int(os.getenv("LLM_MAX_TOKENS_DEFAULT", "5000"))
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler()  # Outputs logs to console
        # For production, consider adding a FileHandler:
        # logging.FileHandler(os.getenv("LOG_FILE_PATH", "app.log"))
    ]
)
logger = logging.getLogger(__name__)

# --- Azure OpenAI Client Initialization with Instructor Patching ---
# The client is initialized once and reused.
AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION
        )
# client: Optional[instructor.Instructor] = None
try:
    if not all([AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION]):
        logger.error("Azure API Key, Endpoint, or Version is missing in environment variables. LLM client cannot be initialized.")
        # client remains None
    else:
        azure_client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION
        )
        client = azure_client
        if client is None:
            # This case should ideally not be hit if AzureOpenAI and instructor.patch work as expected
            # but it's a safeguard based on the original code's check.
            raise ValueError("instructor.patch(AzureOpenAI(...)) returned None.")
        logger.info("Successfully initialized and patched AzureOpenAI client with instructor.")
except Exception as e_client_init: # Renamed e
    logger.critical(f"Critical error initializing AzureOpenAI client: {e_client_init}", exc_info=True)
    client = None # Ensure client is None if initialization fails

# --- Helper Functions ---

# --- Step 1 & 3 Combined: Analyze Document and Extract Clean Key-Value Pairs ---
def extract_text_from_jpg(jpg_bytes, endpoint, key):
    # Create a Form Recognizer client
    client = FormRecognizerClient(endpoint, AzureKeyCredential(key))
    poller = client.begin_recognize_content(jpg_bytes)
    result = poller.result()

    # Extract text and bounding boxes
    extracted_text = []
    bounding_boxes = []
    for page in result:
        for line in page.lines:
            extracted_text.append(line.text)
            bounding_boxes.append({
                "text": line.text,
                "bounding_box": line.bounding_box
            })

    # Create the Document Analysis client for key-value extraction
    doc_client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
    poller = doc_client.begin_analyze_document("prebuilt-document", document=jpg_bytes)
    doc_result = poller.result()

    key_value_pairs = []
    for kv_pair in doc_result.key_value_pairs:
        key_content = kv_pair.key.content if kv_pair.key else "N/A"
        value_content = kv_pair.value.content if kv_pair.value else "N/A"
        key_value_pairs.append({
            "key": key_content,
            "value": value_content
        })

    return extracted_text, bounding_boxes, key_value_pairs

# --- Step 4 Part 1: Generate Synthetic Data with GPT ---
def generate_synthetic_data(fields_for_gpt: dict) -> list:
    """
    Uses GPT-4o to generate multiple new, fictional data entries based on extracted fields.
    """
    print(f"Step 4 (Part 1): Generating data sets with GPT-4o...")

    prompt = f"""
    You are a highly accurate synthetic data generation assistant.
    Your task is to create exactly similar but different data entries for a form. word must be in similar context
    
    **Fields and Examples:**
    {json.dumps(fields_for_gpt, indent=2)}

    **Required same JSON as Output Format but change the value :**
    """
    messages = [
        {"role": "system",
         "content": "You are a data generation assistant that only outputs valid JSON matching the required format."},
        {"role": "user", "content": prompt}
    ]
    temperature = 0.0
    response_format = {"type": "json_object"}

    response = client.chat.completions.create(
        model=GPT_MODEL_NAME,
        messages=messages,
        temperature=temperature,
        response_format=response_format
    )

    generated_json = json.loads(response.choices[0].message.content)
    print("Successfully generated synthetic data from GPT-4o.")
    return generated_json


# Assume these are defined elsewhere in your file
Point = namedtuple('Point', ['x', 'y'])

# def _get_font_for_box(text, box_width, box_height, font_path="arial.ttf"):
#     """Calculates the largest font size that fits the text within a given box."""
#     font_size = int(box_height)
#     try:
#         font = ImageFont.truetype(font_path, font_size)
#     except IOError:
#         font = ImageFont.load_default()
#     while font.getbbox(text)[2] > box_width and font_size > 1:
#         font_size -= 1
#         try:
#             font = ImageFont.truetype(font_path, font_size)
#         except IOError:
#             font = ImageFont.load_default()
#             break
#     return font

# --- MODIFIED FUNCTION ---
# def fill_form_and_save(template_image_path, bounding_boxes_with_new_text, output_image_path):
#     """
#     Fills a template image with new data and saves it.
#     The new text is expected to be inside the 'text' field of the bounding_boxes list itself.
#
#     Args:
#         template_image_path (str): Path to the background image.
#         bounding_boxes_with_new_text (list): A list of dictionaries, each containing the
#                                              NEW 'text' to write and its 'bounding_box' location.
#         output_image_path (str): The path where the new image will be saved.
#     """
#     # 1. Read the template image using OpenCV
#     image_cv = cv2.imread(template_image_path)
#     if image_cv is None:
#         raise FileNotFoundError(f"Could not read image from path: {template_image_path}")
#
#     # 2. Erase old text by filling bounding boxes with white
#     # This step only needs the coordinates, so it works perfectly.
#     for box_info in bounding_boxes_with_new_text:
#         polygon_points = np.array([(p.x, p.y) for p in box_info["bounding_box"]], dtype=np.int32)
#         cv2.fillPoly(image_cv, [polygon_points], (255, 255, 255))
#
#     # 3. Convert image from OpenCV (BGR) to PIL (RGB) for better text rendering
#     image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(image_pil)
#
#     # 4. Write new synthetic data into the cleared boxes
#     # The lookup in `new_data_set` is no longer needed.
#     for box_info in bounding_boxes_with_new_text:
#         # The 'text' field now directly contains the NEW text to write.
#         text_to_write = str(box_info["text"])
#         polygon = box_info["bounding_box"]
#
#         # Get the coordinates to define the box area
#         x_coords = [p.x for p in polygon]
#         y_coords = [p.y for p in polygon]
#         x_min, y_min = int(min(x_coords)), int(min(y_coords))
#         box_width = int(max(x_coords) - x_min)
#         box_height = int(max(y_coords) - y_min)
#
#         # Ensure the box has a valid area before trying to write text
#         if box_width > 0 and box_height > 0:
#             font = _get_font_for_box(text_to_write, box_width, box_height)
#             draw.text((x_min, y_min), text_to_write, font=font, fill=(0, 0, 0)) # Black text
#
#     # 5. Save the modified image to the specified local directory
#     image_pil.save(output_image_path)
#     print(f"Successfully created and saved new image to: {output_image_path}")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Helper function assumed to exist for this example
def _get_font_for_box(text, box_width, box_height):
    """A dummy function to get a font. A real implementation would be more complex."""
    # This is a simplified placeholder.
    # A real version would dynamically find the best font size.
    try:
        # Try to use a common font, fallback to default
        font = ImageFont.truetype("arial.ttf", size=int(box_height * 0.7))
    except IOError:
        font = ImageFont.load_default()
    return font


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# A palette of pleasant, semi-transparent colors for the sticky notes/highlights.
# The format is (R, G, B, Alpha). Alpha=100 (out of 255) provides good transparency.
STICKY_NOTE_COLORS = [
    (255, 255, 153, 100),  # Light Yellow
    (173, 216, 230, 100),  # Light Blue
    (144, 238, 144, 100),  # Light Green
    (255, 182, 193, 100),  # Light Pink
    (255, 204, 153, 100),  # Light Orange
    (221, 160, 221, 100),  # Plum/Light Purple
]


# Helper function assumed to exist for this example
def _get_font_for_box(text, box_width, box_height):
    """A dummy function to get a font. A real implementation would be more complex."""
    try:
        # Using a slightly smaller font size to ensure it fits well within the box
        font_size = int(box_height * 0.7)
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()
    return font


def fill_form_and_save(template_image_path, bounding_boxes_with_new_text, output_image_path):
    """
    Fills a template image with new data, adding a transparent colored
    "sticky note" behind each new text entry, and saves the result.

    Args:
        template_image_path (str): Path to the background image.
        bounding_boxes_with_new_text (list): A list of dictionaries, each containing the
                                             NEW 'text' to write and its 'bounding_box' location.
        output_image_path (str): The path where the new image will be saved.
    """
    # 1. Read the template image using OpenCV
    image_cv = cv2.imread(template_image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Could not read image from path: {template_image_path}")

    # 2. Erase old text by filling bounding boxes with the background color (e.g., white)
    for box_info in bounding_boxes_with_new_text:
        polygon_points = np.array([(p.x, p.y) for p in box_info["bounding_box"]], dtype=np.int32)
        cv2.fillPoly(image_cv, [polygon_points], (255, 255, 255))

    # 3. Convert image to PIL's RGBA format to support transparency
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # +------------------------------------------------------------------+
    # | STICKY NOTE LOGIC: Create and apply a transparent color overlay  |
    # +------------------------------------------------------------------+

    # 4. Create a new, fully transparent overlay image of the same size
    overlay = Image.new("RGBA", image_pil.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # 5. Draw the colored, transparent polygons (sticky notes) on the overlay
    for i, box_info in enumerate(bounding_boxes_with_new_text):
        polygon_points = [(p.x, p.y) for p in box_info["bounding_box"]]

        # Cycle through the predefined colors for variety
        color = STICKY_NOTE_COLORS[i % len(STICKY_NOTE_COLORS)]

        # Draw the transparent polygon on the overlay
        draw_overlay.polygon(polygon_points, fill=color)

    # 6. Composite the overlay (with sticky notes) onto the main image
    combined_image = Image.alpha_composite(image_pil, overlay)

    # 7. Now, draw the new text on top of the combined image
    draw_text = ImageDraw.Draw(combined_image)
    for box_info in bounding_boxes_with_new_text:
        text_to_write = str(box_info["text"])
        polygon = box_info["bounding_box"]

        x_coords = [p.x for p in polygon]
        y_coords = [p.y for p in polygon]
        x_min, y_min = int(min(x_coords)), int(min(y_coords))
        box_width = int(max(x_coords) - x_min)
        box_height = int(max(y_coords) - y_min)

        if box_width > 0 and box_height > 0:
            font = _get_font_for_box(text_to_write, box_width, box_height)
            # Draw black text on top of the sticky note
            draw_text.text((x_min, y_min), text_to_write, font=font, fill=(0, 0, 0))

    # 8. Convert back to RGB for saving (especially for JPG compatibility) and save
    final_image = combined_image.convert("RGB")
    final_image.save(output_image_path)
    print(f"Successfully created and saved new image with sticky notes to: {output_image_path}")