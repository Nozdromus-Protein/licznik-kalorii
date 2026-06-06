import os
import json
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEYS = [
    k.strip()
    for k in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",")
    if k.strip()
]

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def get_client(index: int):
    if not GEMINI_API_KEYS:
        raise HTTPException(status_code=500, detail="Brak GEMINI_API_KEYS albo GEMINI_API_KEY w Render Environment")
    return genai.Client(api_key=GEMINI_API_KEYS[index % len(GEMINI_API_KEYS)])


def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def to_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", ".")
        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    try:
        return int(round(to_float(value, default)))
    except Exception:
        return default


def normalize_result(data: dict) -> dict:
    micros = data.get("microNutrients") or data.get("micros") or data.get("vitaminsAndMinerals") or {}
    additives = data.get("additives") or data.get("emulsifiers") or data.get("emulgatory") or []

    return {
        "name": str(data.get("name") or data.get("nazwa") or "Posiłek"),
        "category": str(data.get("category") or data.get("kategoria") or "Inne"),
        "calories": to_int(data.get("calories") or data.get("kcal")),
        "protein": to_int(data.get("protein") or data.get("bialko") or data.get("białko")),
        "carbs": to_int(data.get("carbs") or data.get("weglowodany") or data.get("węglowodany")),
        "sugar": to_int(data.get("sugar") or data.get("cukry")),
        "fat": to_int(data.get("fat") or data.get("tluszcz") or data.get("tłuszcz")),
        "saturatedFat": to_float(data.get("saturatedFat") or data.get("nasycone")),
        "fiber": to_float(data.get("fiber") or data.get("blonnik") or data.get("błonnik")),
        "salt": to_float(data.get("salt") or data.get("sol") or data.get("sól")),
        "confidence": to_float(data.get("confidence"), 0.75),
        "note": str(data.get("note") or data.get("uwagi") or ""),
        "microNutrients": micros if isinstance(micros, dict) else {},
        "additives": additives if isinstance(additives, list) else [],
    }


@app.get("/")
async def root():
    return {"status": "ok", "model": MODEL_NAME, "keys": len(GEMINI_API_KEYS)}


@app.post("/analyze-meal")
async def analyze_meal(
    file: UploadFile = File(...),
    description: str = Form("")
):
    image_bytes = await file.read()

    mime_type = "image/jpeg"
    if image_bytes.startswith(b"\x89PNG"):
        mime_type = "image/png"
    elif image_bytes.startswith(b"\xff\xd8"):
        mime_type = "image/jpeg"
    elif image_bytes.startswith(b"RIFF"):
        mime_type = "image/webp"

    prompt = f"""
Przeanalizuj zdjęcie posiłku, produktu albo etykiety.

Dodatkowy opis od użytkownika:
"{description}"

Zwróć WYŁĄCZNIE poprawny JSON, bez markdown.

Format:
{{
  "name": "nazwa posiłku lub produktu",
  "category": "Owoc / Warzywo / Mięso / Nabiał / Pieczywo / Danie gotowe / Napój / Inne",
  "calories": 0,
  "protein": 0,
  "carbs": 0,
  "sugar": 0,
  "fat": 0,
  "saturatedFat": 0,
  "fiber": 0,
  "salt": 0,
  "confidence": 0.75,
  "note": "krótka notatka po polsku",
  "microNutrients": {{
    "Rozpuszczalne w tłuszczach / Witamina A [µg]": 0,
    "Rozpuszczalne w tłuszczach / Witamina D [µg]": 0,
    "Rozpuszczalne w tłuszczach / Witamina E [mg]": 0,
    "Rozpuszczalne w tłuszczach / Witamina K [µg]": 0,
    "Rozpuszczalne w wodzie / Witamina C [mg]": 0,
    "Witaminy z grupy B / B1 Tiamina [mg]": 0,
    "Witaminy z grupy B / B2 Ryboflawina [mg]": 0,
    "Witaminy z grupy B / B3 Niacyna/PP [mg]": 0,
    "Witaminy z grupy B / B5 Kwas pantotenowy [mg]": 0,
    "Witaminy z grupy B / B6 Pirydoksyna [mg]": 0,
    "Witaminy z grupy B / B7 Biotyna [µg]": 0,
    "Witaminy z grupy B / B9 Kwas foliowy [µg]": 0,
    "Witaminy z grupy B / B12 Kobalamina [µg]": 0,
    "Makroelementy / Wapń [mg]": 0,
    "Makroelementy / Fosfor [mg]": 0,
    "Makroelementy / Magnez [mg]": 0,
    "Makroelementy / Potas [mg]": 0,
    "Makroelementy / Sód [mg]": 0,
    "Mikroelementy / Żelazo [mg]": 0,
    "Mikroelementy / Cynk [mg]": 0,
    "Mikroelementy / Jod [µg]": 0,
    "Mikroelementy / Selen [µg]": 0
  }},
  "additives": [
    {{
      "name": "nazwa dodatku",
      "code": "E471",
      "category": "Emulgator",
      "riskLevel": 2,
      "note": "krótki opis"
    }}
  ]
}}

Jeżeli czegoś nie da się realnie ocenić, wpisz 0.
Nie dopisuj tekstu poza JSON.
"""

    last_error = None

    for index in range(len(GEMINI_API_KEYS)):
        try:
            client = get_client(index)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_bytes,
                                }
                            },
                        ],
                    }
                ],
            )

            raw_text = response.text or ""
            json_text = clean_json_text(raw_text)
            data = json.loads(json_text)

            result = normalize_result(data)
            result["usedKeyIndex"] = index + 1
            return result

        except Exception as e:
            error_text = str(e)
            last_error = error_text

            if "429" in error_text or "quota" in error_text.lower() or "rate" in error_text.lower():
                continue

            raise HTTPException(status_code=500, detail=f"Błąd analizy AI: {error_text}")

    raise HTTPException(
        status_code=429,
        detail=f"Wszystkie klucze Gemini przekroczyły limit albo quota. Ostatni błąd: {last_error}"
    )
