import random
import gradio as gr
import os
import shutil

files = []

def random_response(message, history):
    return random.choice(["Yes", "No"])

def process_file(fileobj):
    path = "./docs/" + os.path.basename(fileobj)
    shutil.copyfile(fileobj.name, path)

with gr.Blocks() as demo:
    chatInterface = gr.ChatInterface(random_response)
    gr.Interface(
        fn=process_file,
        inputs = [
            "file",
        ]
    )
demo.launch(server_name='0.0.0.0')