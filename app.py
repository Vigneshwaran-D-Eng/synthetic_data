import streamlit as st
import requests
import os
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="REPLICA | Synthetic Form Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Configuration ---
# This should be the address where your FastAPI app is running
API_BASE_URL = "http://127.0.0.1:8000"
API_ENDPOINT = f"{API_BASE_URL}/generate-images/"

# --- Sidebar ---
with st.sidebar:
    st.image("https://i.imgur.com/g0hM621.png", width=150)  # A sample logo
    st.title("REPLICA")
    st.info(
        """
        **Welcome to the Synthetic Form Generator!**

        1. **Upload** a blank form template (JPG or PNG).
        2. **Click** the 'Generate' button.
        3. **View** 10 synthetically filled-in versions of your form.
        """
    )
    st.warning(
        "Ensure the FastAPI backend server is running before you start."
    )

# --- Main Page ---
st.title("✨ REPLICA: Synthetic Form Generator")
st.markdown(
    "Upload a form template and our AI-powered backend will generate 10 unique, synthetically-filled versions for you. Perfect for training and testing document processing models."
)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload your form template",
    type=['jpg', 'png'],
    help="Please upload a clear image of the form you want to use as a template."
)

if uploaded_file is not None:
    # Display the uploaded template
    image = Image.open(uploaded_file)
    st.subheader("Your Uploaded Template")
    # MODIFIED: Replaced `use_column_width` with `use_container_width`
    st.image(image, caption="This is the template that will be used for generation.", use_container_width=False, width=400)

    # --- Generate Button ---
    if st.button("🚀 Generate 10 Synthetic Images", type="primary"):
        # Prepare the file for the API request
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        # Show a spinner while processing
        with st.spinner(
                "🤖 AI at work... Analyzing template, generating data, and creating images. This may take a minute."):
            try:
                response = requests.post(API_ENDPOINT, files=files, timeout=180)  # 3 minute timeout

                if response.status_code == 200:
                    data = response.json()
                    saved_files = data.get("saved_files", [])

                    st.success(f"✅ Success! {data.get('message')}")
                    st.balloons()

                    st.subheader("Generated Images")

                    # Create a grid to display images
                    # We'll do 5 columns, so 2 rows for 10 images
                    num_columns = 5
                    cols = st.columns(num_columns)

                    for i, file_path in enumerate(saved_files):
                        # Construct the full URL to the image
                        image_url = f"{API_BASE_URL}/{file_path}"
                        col_index = i % num_columns

                        with cols[col_index]:
                            st.image(
                                image_url,
                                caption=f"Variant {i + 1}",
                                # MODIFIED: Replaced `use_column_width` with `use_container_width`
                                use_container_width=True
                            )

                else:
                    # Handle API errors
                    error_detail = response.json().get("detail", "No detail provided.")
                    st.error(f"❌ An error occurred (Status {response.status_code}): {error_detail}")

            except requests.exceptions.RequestException as e:
                # Handle network errors
                st.error(
                    f"🔌 Network Error: Could not connect to the API. Please ensure the backend is running at {API_BASE_URL}. Details: {e}")

else:
    st.info("Please upload an image to begin.")