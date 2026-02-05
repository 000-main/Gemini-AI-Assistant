#pip install --upgrade google-genai gradio -- install before running with python3 main.py in terminal -- and Open your browser and go to http://127.0.0.1:7860 to access the web app

import os
import gradio as gr
from google import genai
from google.genai import types

# --- 1. Client Setup ---
# It's best practice to set GEMINI_API_KEY in your system environment variables.
# If not set, you can replace os.environ.get(...) with "YOUR_ACTUAL_KEY" temporarily.
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("⚠️ Warning: GEMINI_API_KEY not found in environment variables.")
    # Optional: api_key = "AIza..." 

client = genai.Client(api_key=api_key)

# --- 2. Chat Logic ---
def stream_gemini(message, history):
    # Search tool for grounded answers
    tools = [types.Tool(googleSearch=types.GoogleSearch())]
    
    config = types.GenerateContentConfig(
        temperature=0.7,
        tools=tools,
    )

    full_response = ""
    try:
        # Initializing the stream
        response_stream = client.models.generate_content_stream(
            model="gemini-2.0-flash", 
            contents=message,
            config=config
        )
        
        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                yield full_response # Streams text to the UI
                
    except Exception as e:
        yield f"Error: {str(e)}"

# --- 3. Gradio Interface ---
demo = gr.ChatInterface(
    fn=stream_gemini,
    title="Local Gemini Assistant",
    description="Running locally with Google Search integration.",
    theme="ocean", # Professional local look
    type="messages"
)

if __name__ == "__main__":
    # share=False for local-only, share=True if you want a temporary public link
    demo.launch(share=False)
