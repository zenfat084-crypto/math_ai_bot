import os
import json
import requests
from flask import Flask, request
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, filters

# متغيرات البيئة من Render
BOT_TOKEN = os.environ.get("BOT_TOKEN")
COLAB_API_URL = os.environ.get("COLAB_API_URL")  # رابط ngrok من Colab

if not BOT_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN غير موجود في Environment Variables")

app = Flask(__name__)
bot = Bot(BOT_TOKEN)

# dispatcher لمعالجة الرسائل
dispatcher = Dispatcher(bot, None, workers=0, use_context=True)

# أمر /start
def start(update, context):
    update.message.reply_text("🚀 البوت يعمل! أرسل أي رسالة لتوليد رد بالذكاء الاصطناعي.")

# عند استقبال رسالة عادية
def handle_message(update, context):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # عرض "typing..." للمستخدم
    bot.send_chat_action(chat_id=chat_id, action="typing")

    if not COLAB_API_URL:
        bot.send_message(chat_id, "⚠️ خادم الذكاء الاصطناعي غير متصل حاليًا.")
        return

    # طلب للذكاء الاصطناعي في Colab
    try:
        response = requests.post(
            f"{COLAB_API_URL.rstrip('/')}/generate",
            json={"prompt": user_text},
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            ai_reply = data.get("response", "لم يصل رد من النموذج.")
            bot.send_message(chat_id, ai_reply)
        else:
            bot.send_message(chat_id, f"⚠️ خطأ في خادم AI (رمز {response.status_code})")

    except requests.exceptions.RequestException:
        bot.send_message(chat_id, "❌ لم أستطع التواصل مع خادم الذكاء الاصطناعي.")
        

# ربط الأوامر
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Webhook endpoint
@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

# Health check
@app.route("/healthz")
def health():
    return "ok"

# تشغيل السيرفر على Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    render_url = os.environ.get("RENDER_EXTERNAL_URL")  # Render قد يضبطها تلقائيًا

    if render_url:
        webhook_url = f"{render_url.rstrip('/')}/{BOT_TOKEN}"
        try:
            bot.set_webhook(webhook_url)
            print("Webhook set to:", webhook_url)
        except Exception as e:
            print("Webhook setup failed:", e)
    else:
        print("⚠️ تعذّر العثور على RENDER_EXTERNAL_URL — اضبط Webhook يدويًا لاحقًا.")

    app.run(host="0.0.0.0", port=port)
