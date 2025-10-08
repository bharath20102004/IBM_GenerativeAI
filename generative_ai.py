# -*- coding: utf-8 -*-
"""IBM_GenerativeAI.ipynb"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=800, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def text_generation(prompt):
    full_prompt = f"Generate a creative and coherent response for the following prompt:\n\n{prompt}\n\nAnswer:"
    return generate_response(full_prompt, max_length=600)

def summarize_text(input_text):
    prompt = f"Summarize the following text clearly and concisely:\n\n{input_text}\n\nSummary:"
    return generate_response(prompt, max_length=400)

def chatbot_response(user_message, history):
    history = history or []
    chat_prompt = "The following is a conversation between a helpful AI assistant and a user.\n\n"
    for turn in history:
        chat_prompt += f"User: {turn[0]}\nAssistant: {turn[1]}\n"
    chat_prompt += f"User: {user_message}\nAssistant:"
    response = generate_response(chat_prompt, max_length=300)
    history.append((user_message, response))
    return response, history

with gr.Blocks() as app:
    gr.Markdown("# 🤖 IBM Generative AI Playground")

    with gr.Tabs():
        with gr.TabItem("📝 Text Generation"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Enter a prompt", lines=4)
                    gen_button = gr.Button("Generate")
                with gr.Column():
                    text_output = gr.Textbox(label="Generated Text", lines=20)
            gen_button.click(text_generation, inputs=text_input, outputs=text_output)

        with gr.TabItem("📄 Summarization"):
            with gr.Row():
                with gr.Column():
                    summary_input = gr.Textbox(label="Enter text to summarize", lines=8)
                    sum_button = gr.Button("Summarize")
                with gr.Column():
                    summary_output = gr.Textbox(label="Summary", lines=15)
            sum_button.click(summarize_text, inputs=summary_input, outputs=summary_output)

        with gr.TabItem("💬 Chat Assistant"):
            chatbot = gr.Chatbot(label="IBM Generative Chatbot")
            user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
            clear = gr.Button("Clear Chat")

            def respond(user_message, history):
                response, history = chatbot_response(user_message, history)
                return history, history

            user_input.submit(respond, [user_input, chatbot], [chatbot, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

app.launch(share=True)
