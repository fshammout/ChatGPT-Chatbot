import os
import gradio as gr
import pandas as pd
import tempfile
from gpt_index import LLMPredictor, PromptHelper, ServiceContext, GPTSimpleVectorIndex
from langchain import OpenAI
from llama_index import download_loader
import openai

os.environ["OPENAI_API_KEY"] = ""

vectorIndex = None

def createVectorIndex(uploaded_file):
    max_input = 4000
    tokens = 256
    chunk_size = 3000
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(
        max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    llmPredictor = LLMPredictor(llm=OpenAI(
        temperature=0, model_name="text-davinci-003", max_tokens=tokens))

    PandasCSVReader = download_loader("PandasCSVReader")
    loader = PandasCSVReader()
    file_path = uploaded_file.name
    docs = loader.load_data(file=file_path)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llmPredictor, prompt_helper=prompt_helper)
    
    global vectorIndex
    vectorIndex = GPTSimpleVectorIndex.from_documents(
        documents=docs, service_context=service_context)

    return('Index saved successfully!')


def chat(chat_history, user_input):
    bot_response = vectorIndex.query(user_input, response_mode="compact")
    response = ""
    for letter in ''.join(bot_response.response):
        response += letter + ""
        yield chat_history + [(user_input, response)]



with gr.Blocks() as demo:
    gr.Markdown("Feras' ChatBot with ChatGPT Models")
    with gr.Tab("Input Text Document"):
        file_upload = gr.File(label="Upload CSV")
        text_output = gr.Textbox()
        text_button = gr.Button("Build the Bot!!!")
        text_button.click(createVectorIndex, file_upload, text_output)
    with gr.Tab("Knowledge Bot"):
        #          inputbox = gr.Textbox("Input your text to build a Q&A Bot here.....")
        chatbot = gr.Chatbot()
        message = gr.Textbox("What is this dataset about?")
        message.submit(chat, [chatbot, message], chatbot)

demo.queue().launch(debug = True)

