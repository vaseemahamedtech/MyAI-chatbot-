import os
import time
import logging
import threading
import multiprocessing
from functools import wraps
from dotenv import load_dotenv
import google.genai as genai
from serpapi import GoogleSearch
from flask import Flask, render_template, request, jsonify

load_dotenv()
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

tts_process = None
tts_lock = threading.Lock()

def speak_in_process(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        logger.warning(f"TTS error: {e}")

def start_tts(text):
    global tts_process
    with tts_lock:
        stop_tts()
        tts_process = multiprocessing.Process(target=speak_in_process, args=(text,))
        tts_process.daemon = True
        tts_process.start()

def stop_tts():
    global tts_process
    if tts_process and tts_process.is_alive():
        tts_process.terminate()
        tts_process.join(timeout=1)
        if tts_process.is_alive():
            tts_process.kill()
        tts_process = None

def rate_limit(max_per_second=3):
    def decorator(f):
        f.last_called = 0
        f.min_interval = 1.0 / max_per_second

        @wraps(f)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - f.last_called
            left_to_wait = f.min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = f(*args, **kwargs)
            f.last_called = time.time()
            return ret
        return wrapper
    return decorator

def get_gemini_response_with_custom_search(question):
    search = GoogleSearch({
        "q": question,
        "api_key": SERPAPI_API_KEY,
        "gl": "us",
        "hl": "en",
    })
    result = search.get_dict()
    snippets = []
    sources = []
    if "organic_results" in result:
        for res in result["organic_results"][:3]:
            snippet = res.get("snippet") or res.get("title") or ""
            url = res.get("link") or ""
            if snippet and url:
                snippets.append(snippet)
                sources.append({"title": snippet, "url": url})
    prompt = question + "\n\nContext:\n" + "\n".join(snippets)
    response = genai.generate_text(
        model="gemini-2.5-flash",
        prompt=prompt
    )
    return {
        "response": response.text,
        "sources": sources
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
@rate_limit(max_per_second=3)
def ask():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        question = data.get("message", "").strip()
        
        if not question:
            return jsonify({"response": "⚠️ No question received.", "sources": []}), 400
        
        if len(question) > 500:
            return jsonify({"response": "⚠️ Message too long. Please keep it under 500 characters.", "sources": []}), 400
        
        result = get_gemini_response_with_custom_search(question)
        
        threading.Thread(target=start_tts, args=(result["response"],), daemon=True).start()

        return jsonify({
            "response": result["response"],
            "sources": result["sources"],
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}")
        return jsonify({"response": f"⚠️ An error occurred: {e}", "sources": []}), 500

@app.route("/stop_speech", methods=["POST"])
def stop_speech():
    try:
        stop_tts()
        return jsonify({"status": "success", "message": "Speech stopped"})
    except Exception as e:
        logger.error(f"Error stopping speech: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
