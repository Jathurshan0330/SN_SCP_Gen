import re
from openai import AzureOpenAI
from typing import OrderedDict
from dotenv import dotenv_values, load_dotenv
import os

import torch
from transformers import pipeline
import urllib.request
import json
import os
import ssl


def load_env_vars() -> OrderedDict:
    if os.path.exists(".env"):
        load_dotenv(override=True)
        config = dotenv_values(".env")

    return config

def azure_open_ai_call(config, 
                       prompt,
                       prompt_preamble = "You are an oncologist. Answer based on the given clinical note for a patient.",
                       temperature = 0 ):
    api_base = config["OPENAI_API_BASE"]
    api_version = config["OPENAI_API_VERSION"]
    api_key = config["AZURE_OPENAI_API_KEY"]
    chat_completion_deployment = config["AZURE_OPENAI_DEPLOYMENT"]


    client = AzureOpenAI(
        api_key=api_key, 
        azure_endpoint=api_base,
        azure_deployment=chat_completion_deployment,
        api_version=api_version,
        )
    

    completion = client.chat.completions.create(
        model=chat_completion_deployment,
        messages=[{
            "role": "system",
            "content":prompt_preamble,
        },
                    {
                        "role": "user",
                        "content": prompt,
                    }],
        temperature=temperature,
    )
            
    
    response = completion.choices[0].message.content

    return response


    

def get_azure_openai_embedding_model(config):
    api_base = config["OPENAI_API_BASE"]
    api_version = config["OPENAI_API_VERSION"]
    api_key = config["AZURE_OPENAI_API_KEY"]
    embeddings_deployment = config["AZURE_EMBEDDING_DEPLOYMENT"]
    
    embedding_model = AzureOpenAIEmbedding(
        api_key=api_key,
        azure_endpoint=api_base,
        deployment_name=embeddings_deployment,
        api_version=api_version)
    
    return embedding_model



class hugging_face_models():
    def __init__(self, model_name_or_path, temperature = 0, max_tokens=1024, device = 'cuda'):
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name_or_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=self.device
        )
        
    def generate_response(self, task_prompt):
        prompt_preamble="You are an oncologist. Answer based on the given clinical note for a patient."
        
        task_prompt = prompt_preamble + '\n' + task_prompt
        messages = [
            {"role": "user", "content": task_prompt},
        ]
        
        outputs = self.pipe(messages, max_new_tokens=self.max_tokens, temperature=self.temperature,do_sample=True)
        
        response = outputs[0]["generated_text"][-1]["content"].strip()
        
        return response