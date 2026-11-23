import os
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

API_KEY = os.environ.get("GROQ_API_KEY", "")

MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]

client = Groq(api_key=API_KEY)
app = Flask(__name__)
CORS(app)

SECTIONS = ["CAUSE", "SOLUTION", "FERTILIZER", "IRRIGATION", "PREVENTION", "SUBSIDY"]


def call_model(messages, temperature=0.3, max_tokens=2048):
    last_error = None
    for model in MODELS:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                msg = str(e)
                if "429" in msg or "503" in msg or "timeout" in msg.lower():
                    time.sleep(2 * (attempt + 1))
                    continue
                break
    raise last_error


def parse_report(text):
    out = {"en": {}, "pa": {}}
    for lang in ["EN", "PA"]:
        for sec in SECTIONS:
            m = re.search(rf"@@@{lang}_{sec}@@@(.*?)(?=@@@|$)", text, re.S)
            value = m.group(1).replace("*", "").strip() if m else ""
            out[lang.lower()][sec.lower()] = value
    return out


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    system = data.get("system") or ""
    temperature = data.get("temperature", 0.3)
    max_tokens = data.get("maxTokens", 2048)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for item in data.get("contents", []):
        role = "assistant" if item.get("role") in ("assistant", "model") else "user"
        messages.append({"role": role, "content": item.get("text", "")})

    try:
        return jsonify({"text": call_model(messages, temperature, max_tokens)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/report", methods=["POST"])
def report():
    d = request.get_json()
    prompt = f"""You are an expert agricultural advisor for farmers in India.
Farmer inputs: Crop: {d.get('crop', '')}, Problem: {d.get('problem', '')}, Location: {d.get('location', '')}, Season: {d.get('season', '')}.

Write the SAME advice twice: first in English, then in Punjabi (Gurmukhi script only).
Each section must be 2-3 short, complete sentences that match between the two languages.
Do not use markdown or asterisks.
Use exactly these tags in this order:

@@@EN_CAUSE@@@
@@@EN_SOLUTION@@@
@@@EN_FERTILIZER@@@
@@@EN_IRRIGATION@@@
@@@EN_PREVENTION@@@
@@@EN_SUBSIDY@@@
@@@PA_CAUSE@@@
@@@PA_SOLUTION@@@
@@@PA_FERTILIZER@@@
@@@PA_IRRIGATION@@@
@@@PA_PREVENTION@@@
@@@PA_SUBSIDY@@@
"""
    messages = [
        {"role": "system", "content": "You are an expert AI farming assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        text = call_model(messages, 0.3, 4096)
        return jsonify(parse_report(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
