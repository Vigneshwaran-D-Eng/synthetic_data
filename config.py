# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Azure Form Recognizer Configuration ---
FR_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
FR_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

# --- Azure OpenAI Configuration ---
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AOAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- Model Configuration ---
MODEL_NAME = "gpt-4o"

# --- Validation ---
def validate_config():
    """Checks if all necessary environment variables are set."""
    required_vars = {
        "Form Recognizer Endpoint": FR_ENDPOINT,
        "Form Recognizer Key": FR_KEY,
        "Azure OpenAI Endpoint": AOAI_ENDPOINT,
        "Azure OpenAI Key": AOAI_KEY,
        "Azure OpenAI API Version": AOAI_API_VERSION,
        "Azure OpenAI Deployment Name": AOAI_DEPLOYMENT_NAME,
    }
    print(required_vars)
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")

# Run validation when the module is imported
validate_config()