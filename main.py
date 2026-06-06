import json
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
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

GEMINI_API_KEYS = [
    key.strip()
    for key in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",")
    if key.strip()
]

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class MealAnalysis(BaseModel):
    name: str
    calories: int
    protein: int
    carbs: int
    sugar: int
    fat: int
    saturated_fat: float
    fiber: float
    salt: float
    confidence: float
    note: Optional[str] = None


@app.get("/")
def home():
    return {
        "status": "Backend Gemini dziala - extended nutrition v2",
        "model": GEMINI_MODEL,
        "keys": len(GEMINI_API_KEYS),
    }


def get_client(index: int):
    if not GEMINI_API_KEYS:
        raise RuntimeError("Brak GEMINI_API_KEYS albo GEMINI_API_KEY w Render Environment")
    return genai.Client(api_key=GEMINI_API_KEYS[index % len(GEMINI_API_KEYS)])


def clean_json_text(text: str) -> str:
    result = text.strip()

    if result.startswith("```json"):
        result = result.replace("```json", "").replace("```", "").strip()
    elif result.startswith("```"):
        result = result.replace("```", "").strip()

    start = result.find("{")
    end = result.rfind("}")
    if start != -1 and end != -1 and end > start:
        result = result[start:end + 1]

    return result


def to_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.lower()
            value = value.replace("g", "")
            value = value.replace("kcal", "")
            value = value.replace(",", ".")
            value = value.strip()

        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    return int(round(to_float(value, default)))


def apply_extra_fallbacks(result_json: dict) -> dict:
    calories = to_int(result_json.get("calories"), 0)
    protein = to_int(result_json.get("protein"), 0)
    carbs = to_int(result_json.get("carbs"), 0)
    sugar = to_int(result_json.get("sugar"), 0)
    fat = to_int(result_json.get("fat"), 0)

    saturated_fat = to_float(result_json.get("saturated_fat") or result_json.get("saturatedFat"), 0)
    fiber = to_float(result_json.get("fiber"), 0)
    salt = to_float(result_json.get("salt"), 0)

    name = str(result_json.get("name", "Posiłek")).lower()
    note = str(result_json.get("note", ""))

    if saturated_fat <= 0 and fat > 0:
        if any(word in name for word in ["ser", "pizza", "burger", "mięso", "mieso", "sos", "makaron", "lasagne", "zapiekanka"]):
            saturated_fat = round(fat * 0.30, 1)
        else:
            saturated_fat = round(fat * 0.20, 1)

    if fiber <= 0 and carbs > 15:
        if any(word in name for word in ["makaron", "pieczywo", "chleb", "płatki", "platki", "owsianka", "warzywa", "ryż", "ryz"]):
            fiber = round(carbs * 0.07, 1)
        else:
            fiber = round(carbs * 0.04, 1)

    if salt <= 0:
        if any(word in name for word in ["pizza", "burger", "frytki", "kanapka", "makaron", "sos", "gotowe", "lasagne", "zapiekanka", "ser"]):
            salt = 1.5 if calories > 600 else 1.0
        elif calories > 400:
            salt = 0.8
        elif calories > 150:
            salt = 0.3
        else:
            salt = 0.1

    if note:
        note = note + " Dodatkowe wartości mogły zostać oszacowane, jeśli nie były widoczne na etykiecie."
    else:
        note = "Dodatkowe wartości mogły zostać oszacowane, jeśli nie były widoczne na etykiecie."

    return {
        "name": result_json.get("name", "Posiłek"),
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "sugar": sugar,
        "fat": fat,
        "saturated_fat": float(saturated_fat),
        "fiber": float(fiber),
        "salt": float(salt),
        "confidence": float(to_float(result_json.get("confidence"), 0)),
        "note": note,
    }


@app.post("/analyze-meal")
async def analyze_meal(
    file: UploadFile = File(...),
    description: str = Form("")
):
    image_bytes = await file.read()

    mime_type = "image/jpeg"

    if image_bytes.startswith(b"\x89PNG"):
        mime_type = "image/png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:20]:
        mime_type = "image/webp"

    prompt = f"""
Przeanalizuj zdjęcie posiłku, produktu albo opakowania i oszacuj wartości odżywcze.

Dodatkowy opis od użytkownika:
"{description}"

Zwróć WYŁĄCZNIE poprawny JSON, bez markdown, bez ```json, bez komentarzy.

Format:
{{
  "name": "nazwa posiłku lub produktu po polsku",
  "calories": liczba_kcal,
  "protein": liczba_gramow_bialka,
  "carbs": liczba_gramow_weglowodanow,
  "sugar": liczba_gramow_cukru,
  "fat": liczba_gramow_tluszczu,
  "saturated_fat": liczba_gramow_tluszczow_nasyconych,
  "fiber": liczba_gramow_blonnika,
  "salt": liczba_gramow_soli,
  "confidence": liczba_od_0_do_1,
  "note": "krótka uwaga po polsku"
}}

Bardzo ważne:
- Jeśli widzisz tabelę wartości odżywczych, użyj jej jako głównego źródła.
- Jeśli użytkownik podał ilość porcji, przelicz wartości na zjedzoną ilość.
- Jeśli nie widzisz tabeli, oszacuj wszystkie wartości, także saturated_fat, fiber i salt.
- Nie zwracaj 0 dla saturated_fat, fiber albo salt, jeśli wartość prawdopodobnie nie jest zerowa.
- Sól podawaj jako gramy soli, nie jako sód.
- Cukier oznacza "w tym cukry".
- Tłuszcze nasycone oznaczają "w tym kwasy tłuszczowe nasycone".
"""

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    last_error = None

    for index in range(len(GEMINI_API_KEYS)):
        try:
            current_client = get_client(index)

            response = current_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    prompt,
                    image_part,
                ],
            )

            result_text = clean_json_text(response.text)
            print(f"ODPOWIEDZ GEMINI KLUCZ {index + 1}:", result_text)

            result_json = json.loads(result_text)
            final_result = apply_extra_fallbacks(result_json)

            print("FINALNY WYNIK:", final_result)

            return final_result

        except Exception as error:
            error_text = str(error)
            last_error = error_text
            print(f"BLAD GEMINI KLUCZ {index + 1}:", error_text)

            if "429" in error_text or "quota" in error_text.lower() or "rate" in error_text.lower() or "resource_exhausted" in error_text.lower():
                continue

            return {
                "error": error_text,
                "name": "Błąd analizy",
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "sugar": 0,
                "fat": 0,
                "saturated_fat": 0,
                "fiber": 0,
                "salt": 0,
                "confidence": 0,
                "note": "Backend Gemini złapał błąd podczas analizy zdjęcia.",
            }

    return {
        "error": f"Wszystkie klucze Gemini zwróciły 429/quota. Ostatni błąd: {last_error}",
        "name": "Limit Gemini",
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "sugar": 0,
        "fat": 0,
        "saturated_fat": 0,
        "fiber": 0,
        "salt": 0,
        "confidence": 0,
        "note": "Wszystkie klucze API przekroczyły limit. Spróbuj później albo użyj kluczy z innego projektu Google.",
    }
