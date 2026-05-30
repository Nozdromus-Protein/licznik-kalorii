import json
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class MealAnalysis(BaseModel):
    name: str
    calories: int
    protein: int
    carbs: int
    fat: int
    sugar: int
    confidence: float
    note: Optional[str] = None


@app.get("/")
def home():
    return {"status": "Backend Gemini dziala"}


@app.post("/analyze-meal")
async def analyze_meal(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        mime_type = "image/jpeg"

        if image_bytes.startswith(b"\x89PNG"):
            mime_type = "image/png"
        elif image_bytes.startswith(b"\xff\xd8\xff"):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:20]:
            mime_type = "image/webp"

        prompt = """
Przeanalizuj zdjęcie posiłku i oszacuj wartości odżywcze.

Zwróć WYŁĄCZNIE poprawny JSON, bez markdown, bez ```json, bez komentarzy.

Format:
{
  "name": "nazwa posiłku po polsku",
  "calories": liczba_kcal,
  "protein": liczba_gramow_bialka,
  "carbs": liczba_gramow_weglowodanow,
  "fat": liczba_gramow_tluszczu,
  "sugar": liczba_gramow_cukru,
  "confidence": liczba_od_0_do_1,
  "note": "krótka uwaga po polsku"
}

Jeśli widzisz opakowanie produktu, spróbuj rozpoznać produkt.
Jeśli nie znasz dokładnej porcji, oszacuj rozsądnie na podstawie zdjęcia.
"""

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                image_part,
            ],
        )

        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()
        elif result_text.startswith("```"):
            result_text = result_text.replace("```", "").strip()

        print("ODPOWIEDZ GEMINI:", result_text)

        result_json = json.loads(result_text)

        return {
            "name": result_json.get("name", "Posiłek"),
            "calories": int(result_json.get("calories", 0)),
            "protein": int(result_json.get("protein", 0)),
            "carbs": int(result_json.get("carbs", 0)),
            "fat": int(result_json.get("fat", 0)),
            "sugar": int(result_json.get("sugar", 0)),
            "confidence": float(result_json.get("confidence", 0)),
            "note": result_json.get("note"),
        }

    except Exception as error:
        print("BLAD ANALIZY GEMINI:", error)

        return {
            "error": str(error),
            "name": "Błąd analizy",
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "sugar": 0,
            "confidence": 0,
            "note": "Backend Gemini złapał błąd podczas analizy zdjęcia.",
        }