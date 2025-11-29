import os
import json
from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import telebot

# -----------------------------
# 1. تحميل المتغيرات من Railway
# -----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if TELEGRAM_TOKEN == "":
    raise ValueError("❌ لم يتم تعيين TELEGRAM_BOT_TOKEN في Railway!")

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="Markdown")

# -----------------------------
# 2. تحميل النموذج المحلي من GitHub
# -----------------------------
MODEL_PATH = "phi2-4bit"

print("🔄 Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("✅ Model Loaded Successfully!")

# -----------------------------
# 3. دالة توليد الردود
# -----------------------------
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=256,
        temperature=0.2
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


# -----------------------------
# 4. بوت تيليغرام
# -----------------------------
@bot.message_handler(func=lambda message: True)
def handle_message(message):

    user_input = message.text

    bot.send_message(message.chat.id, "⏳ *جارٍ التفكير...*")

    try:
        reply = generate_answer(user_input)
    except Exception as e:
        reply = f"⚠️ حدث خطأ في النموذج:\n`{str(e)}`"

    bot.send_message(message.chat.id, reply)


# -----------------------------
# 5. Flask للاحتفاظ بالخدمة نشطة
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 AI Math Bot is running!"

# -----------------------------
# 6. تشغيل تيليغرام
# -----------------------------
if __name__ == "__main__":
    print("🚀 Bot Started on Railway!")
    bot.polling(non_stop=True)
