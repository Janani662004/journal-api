import psycopg2
import select
import time
from supabase import create_client, Client
from transformers import XLNetTokenizer, XLNetForSequenceClassification, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import torchaudio
import numpy as np
from datetime import datetime

# ========== SUPABASE SETUP ==========
SUPABASE_URL = "https://cfdtkaiekghgymciyqxd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNmZHRrYWlla2doZ3ltY2l5cXhkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MzY5NzE1MSwiZXhwIjoyMDU5MjczMTUxfQ.-XBNHaPzvLgfU8jneukdfdoHG-GUjBi514vSD5c8jzI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== MODEL SETUP ==========
text_model_path = r"D:\project\models\reduced_xlnet"
tokenizer = XLNetTokenizer.from_pretrained(text_model_path)
text_model = XLNetForSequenceClassification.from_pretrained(text_model_path)
text_model.eval()

voice_model_path = r"D:\project\models\wav2vec2"
processor = Wav2Vec2Processor.from_pretrained(voice_model_path)
voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(voice_model_path)
voice_model.eval()

# ========== TEXT EMOTION ANALYSIS ==========
def analyze_text(text):
    if not text or text.strip() == "":
        return "Neutral", []

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = text_model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    pred_id = torch.argmax(probs).item()

    label = text_model.config.id2label.get(pred_id, "Unknown")
    if label == "Unknown":
        print(f"⚠️ Unknown text label for ID {pred_id}: {text[:50]}...")

    return label, probs.tolist()

# ========== AUDIO EMOTION ANALYSIS ==========
def analyze_audio(audio_path):
    try:
        speech, rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"❌ Failed to load audio: {audio_path} – {e}")
        return "Unknown", []

    if rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
        speech = resampler(speech)

    inputs = processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = voice_model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    pred_id = torch.argmax(probs).item()

    label = voice_model.config.id2label.get(pred_id, "Unknown")
    if label == "Unknown":
        print(f"⚠️ Unknown audio label for ID {pred_id} – from file: {audio_path}")

    return label, probs.tolist()

# ========== PIPELINE ==========
def run_pipeline():
    print("🚀 Running journal pipeline...")

    response = supabase.table("journal_entries").select("*").execute()
    entries = response.data

    for entry in entries:
        user_id = entry["user_id"]
        timestamp = entry["timestamp"]
        text_entry = entry.get("text_entry")
        audio_path = entry.get("audio_entry")

        text_label, text_scores = "Neutral", []
        audio_label, audio_scores = "Neutral", []

        if text_entry:
            text_label, text_scores = analyze_text(text_entry)

        if audio_path:
            audio_label, audio_scores = analyze_audio(audio_path)

        # Compute day label (priority: text > audio)
        day_label = text_label if text_label != "Unknown" else audio_label
        day_score = 0.0
        all_scores = text_scores + audio_scores
        if all_scores:
            day_score = max(all_scores)

        # Skip insert if both labels are Unknown
        if text_label == "Unknown" and audio_label == "Unknown":
            print(f"⚠️ Skipping entry for user {user_id} – both labels Unknown")
            continue

        data = {
            "user_id": user_id,
            "timestamp": timestamp,
            "text_emotion_label": text_label,
            "text_scores": text_scores,
            "audio_emotion_label": audio_label,
            "audio_scores": audio_scores,
            "day_label": day_label,
            "day_score": day_score
        }

        print("🔄 Inserting into ai_analysis:", data)
        supabase.table("ai_analysis").insert(data).execute()
        print(f"✅ Entry for user {user_id} processed.")

# ========== LISTENER SETUP ==========
def listen_for_new_entries():
    conn = psycopg2.connect(
        host="your-database-host",
        database="postgres",
        user="your-db-username",
        password="your-db-password"
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute("LISTEN journal_entries_insert;")
    print("👂 Listening for new journal entries...")

    while True:
        if select.select([conn], [], [], 5) == ([], [], []):
            print("⏳ Waiting for notifications...")
        else:
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop()
                print(f"🔔 Received notification: {notify.payload}")
                run_pipeline()

if __name__ == "__main__":
    listen_for_new_entries()
