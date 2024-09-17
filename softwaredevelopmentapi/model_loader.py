from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
#from databrickschatbotapi.DatascientistPipeline.Graph_description import GraphDescriptionPipeline
import torch
#from transformers import BitsAndBytesConfig, pipeline
# model_loader.py
# global llm_model
#global img_to_text_pipe
#global llm_model
#global tokenizer
def load_all_models():

    # Clears cache if you're using CUDA

    global llm_model,tokenizer
    #global img_to_text_pipe
    # Load your model here, e.g., Hugging Face model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True, device=device)
    model_name="meta-llama/Meta-Llama-3-8B-Instruct"
    #model_name="meta-llama/Llama-2-7b-chat-hf"
    llm_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16,use_auth_token="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")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True,
                                  use_auth_token="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")
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

#    quantization_config = BitsAndBytesConfig(
#            load_in_4bit=True,
#            bnb_4bit_compute_dtype=torch.float16,
#            use_nested_quant=False
#        )
    #img_to_text_pipe = pipeline("image-to-text", model="llava-hf/llava-1.5-7b-hf", model_kwargs={"quantization_config": quantization_config})

    print("Models loaded successfully!")

