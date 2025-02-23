import gradio as gr
import os
import requests

SERVICE_HOST = os.getenv("SERVICE_HOST", "localhost")
SERVICE_PORT = os.getenv("SERVICE_PORT", 5001)
API_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"

def chatbot_response(message):
    """Get chatbot response from API"""
    payload = {"text": message}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Welcome to House MD Practice!")
            chatbox = gr.Chatbot()
            message = gr.Textbox(placeholder="Type your message to Dr House here...")
            clear = gr.Button("Clear")

            def user_interaction(user_message, history):
                if user_message.strip() == "":
                    return history
                response = chatbot_response(user_message)
                history.append((user_message, response))
                return history

            message.submit(fn=user_interaction, inputs=[message, chatbox], outputs=chatbox)
            clear.click(lambda: [], None, chatbox)
        gr.Image(
            "/house.jpg",
            label="Dr House"
        )
app.launch(server_name="0.0.0.0", server_port=5002)
