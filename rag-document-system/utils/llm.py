import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-3.1-pro-preview")

def generate_answer(context, query):
    prompt = f"""
You are an expert in call center analytics and systems like Genesys.

Use the provided context to answer the question.

Context:
{context}

Question:
{query}

Instructions:
- First give a direct answer
- Then explain the reasoning clearly
- If possible, include a real-world example
- Keep answer structured and easy to understand

Answer format:

Answer:
<direct answer>

Explanation:
<detailed explanation>

Example:
<optional example>
"""

    response = model.generate_content(prompt)
    return response.text