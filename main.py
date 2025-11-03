import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Loria persona ---
PERSONALITY_EN = """
You are Loria — an AI trading partner. Analyze charts in a pro but chill way: 
show trend, key levels, RSI/EMA context, possible trade idea (entry, SL, TP), 
and include an NFA disclaimer.
"""

PERSONALITY_MN = """
Та Loria — ухаалаг арилжааны туслах. Графикт: чиг хандлага, гол түвшнүүд, 
RSI/EMA мэдээлэл, боломжит санаа (оролт, SL, TP), төгсгөлд NFA сануулга хавсарга.
"""

def system_prompt(lang: str) -> str:
    return PERSONALITY_EN if lang.lower().startswith("en") else PERSONALITY_MN

class ChatRequest(BaseModel):
    message: str
    lang: str = "en"

# --- SERVE FRONTEND ---
@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"ok": True, "name": "Loria", "mode": "web+vision"}

@app.post("/chat")
def chat(req: ChatRequest):
    """Text chat endpoint."""
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt(req.lang)},
                {"role": "user", "content": req.message}
            ],
            max_tokens=400
        )
        return {"reply": result.choices[0].message.content}
    except Exception as e:
        return {"reply": f"Error: {e}"}

@app.post("/analyze-image")
async def analyze_image(lang: str = "en", file: UploadFile = File(...)):
    """Image analysis endpoint — supports trading charts."""
    try:
        image_bytes = await file.read()
        
        # Convert image to base64 for OpenAI API
        import base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt(lang)},
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze the chart in this image: trend, support/resistance, indicator context, possible trade play."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type or 'image/png'};base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=450
        )
        return {"reply": result.choices[0].message.content}
    except Exception as e:
        return {"reply": f"Error analyzing image: {e}"}
