from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# تحميل النموذج من المستودع المحلي
MODEL_DIR = "/content/my-ai-bot/model"  # غيّر له حسب مكان النموذج

print("🔥 Loading Phi-2 model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(data: Prompt):
    prompt = data.prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        do_sample=True,
        top_p=0.9
    )
    
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": text}
