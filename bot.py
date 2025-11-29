import os
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.environ["BOT_TOKEN"]
MODEL_NAME = os.environ["MODEL_NAME"]

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoGPTQForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)

def infer(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=350)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def start(update, context):
    update.message.reply_text("مرحبًا! أنا بوت ذكاء اصطناعي متخصص بالرياضيات. اسأل أي سؤال!")

def reply(update, context):
    q = update.message.text
    update.message.chat.send_action("typing")
    prompt = f"حل المسألة التالية بالتفصيل:\n{q}\n\nالشرح خطوة بخطوة:"
    answer = infer(prompt)
    update.message.reply_text(answer)

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text, reply))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
