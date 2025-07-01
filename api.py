# api.py
import os
import uuid
import uvicorn
import copy
from fastapi import FastAPI, File, UploadFile, HTTPException
from core_logic_new import extract_text_from_jpg, generate_synthetic_data, fill_form_and_save
from config import FR_ENDPOINT, FR_KEY
# --- FastAPI App Initialization ---
app = FastAPI(
    title="REPLICA: Synthetic Form Data API",
    description="Upload a form template to generate and save 10 synthetic variations.",
    version="1.0.0"
)
temp_template_path = r'C:\Users\prince253243\Documents\Image_data_generation\image.jpg'
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/generate-images/", summary="Generate 10 synthetic form images")
async def generate_images_endpoint(file: UploadFile = File(...)):
    """
    Accepts a form image, processes it through the full pipeline,
    and saves 10 synthetic images to the local 'generated_images' directory.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPG or PNG.")

    template_image_bytes = await file.read()

    try:

        extracted_text, bounding_boxes, key_value_pairs= extract_text_from_jpg(
            template_image_bytes, FR_ENDPOINT, FR_KEY
        )
        if not key_value_pairs:
            raise HTTPException(status_code=422, detail="Could not extract usable fields.")
        # Step 4 (Part 1): Generate Data
        saved_files = []
        for _ in range(10):
            fields_for_gpt = {i: box['text'] for i, box in enumerate(bounding_boxes)}
            synthetic_data_set = generate_synthetic_data(fields_for_gpt)
            if not synthetic_data_set:
                raise HTTPException(status_code=500, detail="GPT-4o failed to generate data.")

            # If generate_synthetic_data returns a dict, use as is; if list, get first element
            if isinstance(synthetic_data_set, list):
                synthetic_data_set = synthetic_data_set[0]

            # Create a new list of bounding boxes with updated text
            # updated_boxes = copy.deepcopy(bounding_boxes)
            for k, v in synthetic_data_set.items():
                bounding_boxes[int(k)]['text'] = v

            filename = f"synthetic_{uuid.uuid4().hex[:10]}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)

            fill_form_and_save(
                template_image_path=temp_template_path,
                bounding_boxes_with_new_text=bounding_boxes,
                output_image_path=output_path
            )
            saved_files.append(output_path)

        return {
            "message": f"Successfully generated and saved {len(saved_files)} images.",
            "saved_files": saved_files
        }
    except Exception as e:
        print(f"An internal error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- Main execution block to run the server ---
if __name__ == "__main__":
    # --- FIX: Changed HOST to "127.0.0.1" for Windows compatibility ---
    HOST = "127.0.0.1"
    PORT = 8000

    print(f"Starting Uvicorn server on http://{HOST}:{PORT}")

    uvicorn.run("api:app", host=HOST, port=PORT, reload=True)