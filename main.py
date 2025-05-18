from fastapi import FastAPI, Request
from pydantic import BaseModel
from pipeline import analyze_text, analyze_audio
from datetime import datetime
from supabase import create_client, Client
import logging

# ========== SUPABASE CONNECTION ==========
SUPABASE_URL = "https://cfdtkaiekghgymciyqxd.supabase.co"  # <- replace with your actual Supabase project URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNmZHRrYWlla2doZ3ltY2l5cXhkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM2OTcxNTEsImV4cCI6MjA1OTI3MzE1MX0.dGDqSh2ZsNsX88U6BuWgyWtGfwa1dxlSZfP_uGdzkyY"        # <- replace with your Supabase anon key or service key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== FASTAPI APP ==========
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ========== EMOTION LABEL MAPPING ==========
emotion_label_mapping = {
    "LABEL_0": "anger",
    "LABEL_1": "joy",
    "LABEL_2": "fear",
    "LABEL_3": "sadness",
    "LABEL_4": "surprise",
    "LABEL_5": "disgust",
    "LABEL_6": "trust",
    "LABEL_7": "anticipation",
    "LABEL_8": "love",
    "LABEL_9": "optimism",
    "LABEL_10": "pessimism",
    "LABEL_11": "contentment",
    "LABEL_12": "confusion",
    "LABEL_13": "boredom",
    "LABEL_14": "excitement",
    "LABEL_15": "pride",
    "LABEL_16": "guilt",
    "LABEL_17": "fear"
}

# ========== REQUEST MODEL ==========
class UserRequest(BaseModel):
    user_id: str

# ========== MIDDLEWARE FOR LOGGING ==========
@app.middleware("http")
async def log_request(request: Request, call_next):
    body = await request.body()
    print(f"\n🌐 [MIDDLEWARE] Incoming request: {request.method} {request.url}")
    print(f"📦 Body: {body}")
    response = await call_next(request)
    return response

# ========== EMOTION ANALYSIS ENDPOINT ==========
@app.post("/analyze")
def analyze(user: UserRequest):
    print(f"\n🛠️ [START] Running analysis for user_id: {user.user_id}")

    # --- Step 1: Fetch entries ---
    print("📥 Fetching journal entries...")
    entries_response = supabase.table("journal_entries")\
        .select("*")\
        .eq("user_id", user.user_id)\
        .order("timestamp", desc=True)\
        .execute()
    all_entries = entries_response.data
    print(f"📃 Total entries found: {len(all_entries)}")

    # --- Step 2: Fetch already analyzed timestamps ---
    print("🧠 Fetching already analyzed timestamps...")
    analyzed_response = supabase.table("ai_analysis")\
        .select("timestamp")\
        .eq("user_id", user.user_id)\
        .execute()
    analyzed_timestamps = set(entry["timestamp"] for entry in analyzed_response.data)
    print(f"⏱️ Analyzed timestamps: {analyzed_timestamps}")

    # --- Step 3: Filter unanalyzed entries ---
    unanalysed_entries = [entry for entry in all_entries if entry["timestamp"] not in analyzed_timestamps]
    print(f"🔍 Unanalysed entries count: {len(unanalysed_entries)}")

    if not unanalysed_entries:
        print("✅ No new entries to analyze.")
        return {"status": "No new journal entries to analyze."}

    # --- Step 4: Analyze each entry ---
    for entry in unanalysed_entries:
        timestamp = entry["timestamp"]
        text_entry = entry.get("text_entry")
        audio_path = entry.get("audio_entry")
        print(f"\n📝 Processing entry from {timestamp}")

        text_label, text_scores = "Unknown", []
        audio_label, audio_scores = "Unknown", []

        if text_entry:
            print("✏️ Analyzing text...")
            raw_text_label, text_scores = analyze_text(text_entry)
            text_label = emotion_label_mapping.get(raw_text_label, raw_text_label)

        if audio_path:
            print("🎧 Analyzing audio...")
            raw_audio_label, audio_scores = analyze_audio(audio_path)
            audio_label = emotion_label_mapping.get(raw_audio_label, raw_audio_label)

        # Choose day_label based on priority (text > audio)
        day_label = text_label if text_label != "Unknown" else audio_label
        all_scores = text_scores + audio_scores
        day_score = max(all_scores) if all_scores else 0.0

        if day_label == "Unknown":
            print(f"⚠️ Skipping entry from {timestamp} – no valid emotion label found.")
            continue

        # Prepare insert data
        analysis_data = {
            "user_id": user.user_id,
            "timestamp": timestamp,
            "text_emotion_label": text_label,
            "text_scores": text_scores,
            "audio_emotion_label": audio_label,
            "audio_scores": audio_scores,
            "day_label": day_label,
            "day_score": day_score
        }

        print(f"🧾 Inserting analysis result: {analysis_data}")
        supabase.table("ai_analysis").insert(analysis_data).execute()

    print("✅ [COMPLETE] Emotion analysis finished.")
    return {"status": "Analysis complete", "entries_analyzed": len(unanalysed_entries)}
