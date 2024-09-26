from transformers import AutoModel, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration
import streamlit as st
import os
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
import base64
import random

def init_model():
    tokenizer = AutoTokenizer.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True)
    model = AutoModel.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval()
    return model, tokenizer

def init_gpu_model():
    tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval().cuda()
    return model, tokenizer

def init_qwen_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", device_map="cpu", torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor

def get_quen_op(image_file, model, processor):
    try: 
        image = Image.open(image_file).convert('RGB')
        conversation = [
            {
                "role":"user",
                "content":[
                    {
                        "type":"image",
                    },
                    {
                        "type":"text",
                        "text":"Extract text from this image."
                    }
                ]
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(torch.float32) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": 32,
            "do_sample": False,
            "top_k": 20,
            "top_p": 0.90,
            "temperature": 0.4,
            "num_return_sequences": 1,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }

        output_ids = model.generate(**inputs, **generation_config)
        if 'input_ids' in inputs:
                generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        else:
            generated_ids = output_ids
            
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
        return output_text[:] if output_text else "No text extracted from the image."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

def init_llama():
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.getenv("access_token")
    )
    processor = AutoProcessor.from_pretrained(model_id, token=os.getenv("access_token"))
    return model, processor

def get_llama_op(image_file, model, processor):

    with open(image_file, "rb") as f:
        image = base64.b64encode(f.read()).decode('utf-8')

    image = Image.open(image_file)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "You are an accurate OCR engine. From the given image, extract the text."}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=input_text, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=128)
    return processor.decode(output[0])

def get_text(image_file, model, tokenizer):
    res = model.chat(tokenizer, image_file, ocr_type='ocr')
    return res

st.title("Image - Text OCR")
st.write("Upload an image for OCR")

# MODEL, PROCESSOR = init_llama()
random_value = random.randint(0, 100)
st.write(f"Model loaded: build number - {random_value}")

image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if image_file:

    if not os.path.exists("images"):
        os.makedirs("images")
    with open(f"images/{image_file.name}", "wb") as f:
        f.write(image_file.getbuffer())

    image_file = f"images/{image_file.name}"

    # model, tokenizer = init_gpu_model()
    # model, tokenizer = init_model()
    # text = get_text(image_file, model, tokenizer)

    model, processor = init_llama()
    text = get_llama_op(image_file, model, processor)

    # model, processor = init_qwen_model()
    # text = get_quen_op(image_file, model, processor)
    print(text)
    st.write(text)