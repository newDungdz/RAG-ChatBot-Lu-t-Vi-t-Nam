import google.generativeai as genai
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
# Set your API key (replace with your actual key)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# Generate a response
response = model.generate_content("Explain the importance of fast language models")
print(response.text)