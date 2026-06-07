import asyncio
import base64
import json
import os
import re
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_PROVIDER = os.environ.get("AI_PROVIDER", "gemini").strip().lower()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini").strip()

GEMINI_API_KEYS = [
    key.strip()
    for key in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",")
    if key.strip()
]

GEMINI_MODELS = [
    model.strip()
    for model in os.environ.get(
        "GEMINI_MODELS",
        os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    ).split(",")
    if model.strip()
]

if not GEMINI_MODELS:
    GEMINI_MODELS = ["gemini-2.5-flash"]


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
        "status": "Backend AI dziala - OpenAI/Gemini nutrition v5",
        "provider": AI_PROVIDER,
        "openai_model": OPENAI_MODEL,
        "gemini_models": GEMINI_MODELS,
        "gemini_keys": len(GEMINI_API_KEYS),
        "openai_key": bool(OPENAI_API_KEY),
    }


def get_gemini_client(index: int):
    if not GEMINI_API_KEYS:
        raise RuntimeError("Brak GEMINI_API_KEYS albo GEMINI_API_KEY w Render Environment")
    return genai.Client(api_key=GEMINI_API_KEYS[index % len(GEMINI_API_KEYS)])


def get_openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("Brak OPENAI_API_KEY w Render Environment")
    return OpenAI(api_key=OPENAI_API_KEY)


def clean_json_text(text: str) -> str:
    result = (text or "").strip()

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
            text = value.lower().replace(",", ".").strip()
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if not match:
                return default
            return float(match.group(0))

        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    return int(round(to_float(value, default)))


MICRO_KEYS = [
    "Rozpuszczalne w tłuszczach / Witamina A [µg]",
    "Rozpuszczalne w tłuszczach / Witamina D [µg]",
    "Rozpuszczalne w tłuszczach / Witamina E [mg]",
    "Rozpuszczalne w tłuszczach / Witamina K [µg]",
    "Rozpuszczalne w wodzie / Witamina C [mg]",
    "Witaminy z grupy B / B1 Tiamina [mg]",
    "Witaminy z grupy B / B2 Ryboflawina [mg]",
    "Witaminy z grupy B / B3 Niacyna/PP [mg]",
    "Witaminy z grupy B / B5 Kwas pantotenowy [mg]",
    "Witaminy z grupy B / B6 Pirydoksyna [mg]",
    "Witaminy z grupy B / B7 Biotyna [µg]",
    "Witaminy z grupy B / B9 Kwas foliowy [µg]",
    "Witaminy z grupy B / B12 Kobalamina [µg]",
    "Makroelementy / Wapń [mg]",
    "Makroelementy / Fosfor [mg]",
    "Makroelementy / Magnez [mg]",
    "Makroelementy / Potas [mg]",
    "Makroelementy / Sód [mg]",
    "Makroelementy / Chlor [mg]",
    "Makroelementy / Siarka [mg]",
    "Mikroelementy / Żelazo [mg]",
    "Mikroelementy / Cynk [mg]",
    "Mikroelementy / Miedź [mg]",
    "Mikroelementy / Mangan [mg]",
    "Mikroelementy / Jod [µg]",
    "Mikroelementy / Selen [µg]",
    "Mikroelementy / Fluor [mg]",
    "Mikroelementy / Chrom [µg]",
    "Mikroelementy / Molibden [µg]",
]


def empty_micro_map() -> dict:
    return {key: 0 for key in MICRO_KEYS}


def normalize_micro_nutrients(raw_value) -> dict:
    micros = empty_micro_map()

    if isinstance(raw_value, dict):
        for key, value in raw_value.items():
            key_text = str(key).strip()
            number = to_float(value, 0)
            if key_text:
                micros[key_text] = number

    elif isinstance(raw_value, list):
        for item in raw_value:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("label") or item.get("key") or "").strip()
                unit = str(item.get("unit") or "").strip()
                value = to_float(item.get("value") or item.get("amount") or item.get("amount_per_100"), 0)
                if name:
                    final_key = name if not unit else f"{name} [{unit}]"
                    micros[final_key] = value

    return micros


def morele_micro_fallback() -> dict:
    micros = empty_micro_map()
    micros.update({
        "Rozpuszczalne w tłuszczach / Witamina A [µg]": 96,
        "Rozpuszczalne w tłuszczach / Witamina D [µg]": 0,
        "Rozpuszczalne w tłuszczach / Witamina E [mg]": 0.89,
        "Rozpuszczalne w tłuszczach / Witamina K [µg]": 3.3,
        "Rozpuszczalne w wodzie / Witamina C [mg]": 10,
        "Witaminy z grupy B / B1 Tiamina [mg]": 0.03,
        "Witaminy z grupy B / B2 Ryboflawina [mg]": 0.04,
        "Witaminy z grupy B / B3 Niacyna/PP [mg]": 0.6,
        "Witaminy z grupy B / B5 Kwas pantotenowy [mg]": 0.24,
        "Witaminy z grupy B / B6 Pirydoksyna [mg]": 0.05,
        "Witaminy z grupy B / B7 Biotyna [µg]": 0.3,
        "Witaminy z grupy B / B9 Kwas foliowy [µg]": 9,
        "Witaminy z grupy B / B12 Kobalamina [µg]": 0,
        "Makroelementy / Wapń [mg]": 13,
        "Makroelementy / Fosfor [mg]": 23,
        "Makroelementy / Magnez [mg]": 10,
        "Makroelementy / Potas [mg]": 259,
        "Makroelementy / Sód [mg]": 1,
        "Makroelementy / Chlor [mg]": 2,
        "Makroelementy / Siarka [mg]": 6,
        "Mikroelementy / Żelazo [mg]": 0.39,
        "Mikroelementy / Cynk [mg]": 0.2,
        "Mikroelementy / Miedź [mg]": 0.08,
        "Mikroelementy / Mangan [mg]": 0.08,
        "Mikroelementy / Jod [µg]": 1,
        "Mikroelementy / Selen [µg]": 0.1,
        "Mikroelementy / Fluor [mg]": 0.001,
        "Mikroelementy / Chrom [µg]": 1,
        "Mikroelementy / Molibden [µg]": 1,
    })
    return micros


def apply_extra_fallbacks(result_json: dict) -> dict:
    calories = to_int(result_json.get("calories"), 0)
    protein = to_int(result_json.get("protein"), 0)
    carbs = to_int(result_json.get("carbs"), 0)
    sugar = to_int(result_json.get("sugar"), 0)
    fat = to_int(result_json.get("fat"), 0)

    saturated_fat = to_float(result_json.get("saturated_fat") or result_json.get("saturatedFat"), 0)
    fiber = to_float(result_json.get("fiber"), 0)
    salt = to_float(result_json.get("salt"), 0)

    name_raw = str(result_json.get("name", "Posiłek"))
    name = name_raw.lower()
    note = str(result_json.get("note", ""))

    raw_micros = (
        result_json.get("microNutrients")
        or result_json.get("micro_nutrients")
        or result_json.get("micronutrients")
        or result_json.get("vitaminsAndMinerals")
        or result_json.get("vitamins_minerals")
        or {}
    )

    micro_nutrients = normalize_micro_nutrients(raw_micros)

    if ("morele" in name or "morela" in name or "apricot" in name) and all(to_float(v, 0) == 0 for v in micro_nutrients.values()):
        micro_nutrients = morele_micro_fallback()

    additives = (
        result_json.get("additives")
        or result_json.get("emulsifiers")
        or result_json.get("emulgatory")
        or []
    )

    if not isinstance(additives, list):
        additives = []

    if saturated_fat <= 0 and fat > 0:
        if any(word in name for word in ["ser", "pizza", "burger", "mięso", "mieso", "sos", "makaron", "lasagne", "zapiekanka"]):
            saturated_fat = round(fat * 0.30, 1)
        else:
            saturated_fat = round(fat * 0.20, 1)

    if fiber <= 0 and carbs > 15:
        if any(word in name for word in ["makaron", "pieczywo", "chleb", "płatki", "platki", "owsianka", "warzywa", "ryż", "ryz", "morele", "morela"]):
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
        "name": name_raw,
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "sugar": sugar,
        "fat": fat,
        "saturated_fat": float(saturated_fat),
        "fiber": float(fiber),
        "salt": float(salt),
        "confidence": float(to_float(result_json.get("confidence"), 0.8)),
        "note": note,
        "microNutrients": micro_nutrients,
        "additives": additives,
    }


def build_prompt(description: str) -> str:
    return f"""
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
  "note": "krótka uwaga po polsku",
  "microNutrients": {{
    "Rozpuszczalne w tłuszczach / Witamina A [µg]": liczba,
    "Rozpuszczalne w tłuszczach / Witamina D [µg]": liczba,
    "Rozpuszczalne w tłuszczach / Witamina E [mg]": liczba,
    "Rozpuszczalne w tłuszczach / Witamina K [µg]": liczba,
    "Rozpuszczalne w wodzie / Witamina C [mg]": liczba,
    "Witaminy z grupy B / B1 Tiamina [mg]": liczba,
    "Witaminy z grupy B / B2 Ryboflawina [mg]": liczba,
    "Witaminy z grupy B / B3 Niacyna/PP [mg]": liczba,
    "Witaminy z grupy B / B5 Kwas pantotenowy [mg]": liczba,
    "Witaminy z grupy B / B6 Pirydoksyna [mg]": liczba,
    "Witaminy z grupy B / B7 Biotyna [µg]": liczba,
    "Witaminy z grupy B / B9 Kwas foliowy [µg]": liczba,
    "Witaminy z grupy B / B12 Kobalamina [µg]": liczba,
    "Makroelementy / Wapń [mg]": liczba,
    "Makroelementy / Fosfor [mg]": liczba,
    "Makroelementy / Magnez [mg]": liczba,
    "Makroelementy / Potas [mg]": liczba,
    "Makroelementy / Sód [mg]": liczba,
    "Makroelementy / Chlor [mg]": liczba,
    "Makroelementy / Siarka [mg]": liczba,
    "Mikroelementy / Żelazo [mg]": liczba,
    "Mikroelementy / Cynk [mg]": liczba,
    "Mikroelementy / Miedź [mg]": liczba,
    "Mikroelementy / Mangan [mg]": liczba,
    "Mikroelementy / Jod [µg]": liczba,
    "Mikroelementy / Selen [µg]": liczba,
    "Mikroelementy / Fluor [mg]": liczba,
    "Mikroelementy / Chrom [µg]": liczba,
    "Mikroelementy / Molibden [µg]": liczba
  }},
  "additives": [
    {{
      "name": "nazwa dodatku",
      "code": "E471",
      "category": "Emulgator / dodatek",
      "riskLevel": 2,
      "note": "krótki opis"
    }}
  ]
}}

Bardzo ważne:
- Jeśli widzisz tabelę wartości odżywczych, użyj jej jako głównego źródła.
- Jeśli użytkownik podał ilość porcji, przelicz wartości na zjedzoną ilość.
- Jeśli użytkownik pyta o składnik na 100 g, zwróć wartości na 100 g.
- Jeśli nie widzisz tabeli, oszacuj wszystkie wartości, także saturated_fat, fiber, salt oraz microNutrients.
- Nie zwracaj pustego microNutrients dla owoców, warzyw, mięsa, nabiału, pieczywa, pizzy, gotowych dań i produktów.
- Dla Morele / Morela / Apricot koniecznie zwróć witaminy i minerały typowe dla świeżych moreli na 100 g.
- Jeżeli dany produkt realnie nie ma konkretnej witaminy/minerału, wpisz 0.
- Sól podawaj jako gramy soli, nie jako sód.
- Cukier oznacza "w tym cukry".
- Tłuszcze nasycone oznaczają "w tym kwasy tłuszczowe nasycone".
"""


def get_mime_type(image_bytes: bytes) -> str:
    mime_type = "image/jpeg"

    if image_bytes.startswith(b"\x89PNG"):
        mime_type = "image/png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:20]:
        mime_type = "image/webp"

    return mime_type


async def analyze_with_openai(prompt: str, image_bytes: bytes, mime_type: str) -> dict:
    client = get_openai_client()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                    },
                ],
            }
        ],
        max_output_tokens=5000,
    )

    result_text = getattr(response, "output_text", "") or ""

    if not result_text:
        try:
            chunks = []
            for item in response.output:
                for content in item.content:
                    if hasattr(content, "text"):
                        chunks.append(content.text)
            result_text = "\n".join(chunks)
        except Exception:
            result_text = ""

    result_text = clean_json_text(result_text)
    print("ODPOWIEDZ OPENAI:", result_text)

    result_json = json.loads(result_text)
    final_result = apply_extra_fallbacks(result_json)
    final_result["aiProvider"] = "openai"
    final_result["aiModel"] = OPENAI_MODEL

    print("FINALNY WYNIK OPENAI:", final_result)
    return final_result


async def analyze_with_gemini(prompt: str, image_bytes: bytes, mime_type: str) -> dict:
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    last_error = None

    for model_name in GEMINI_MODELS:
        model_unavailable = False

        for index in range(len(GEMINI_API_KEYS)):
            try:
                current_client = get_gemini_client(index)

                response = current_client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        image_part,
                    ],
                )

                result_text = clean_json_text(response.text or "")
                print(f"ODPOWIEDZ GEMINI MODEL {model_name} KLUCZ {index + 1}:", result_text)

                result_json = json.loads(result_text)
                final_result = apply_extra_fallbacks(result_json)
                final_result["aiProvider"] = "gemini"
                final_result["aiModel"] = model_name

                print("FINALNY WYNIK GEMINI:", final_result)

                return final_result

            except Exception as error:
                error_text = str(error)
                error_lower = error_text.lower()
                last_error = error_text

                print(f"BLAD GEMINI MODEL {model_name} KLUCZ {index + 1}:", error_text)

                is_503 = (
                    "503" in error_text
                    or "unavailable" in error_lower
                    or "high demand" in error_lower
                    or "overloaded" in error_lower
                    or "try again later" in error_lower
                )

                is_429 = (
                    "429" in error_text
                    or "quota" in error_lower
                    or "rate" in error_lower
                    or "resource_exhausted" in error_lower
                )

                if is_503:
                    print(f"MODEL {model_name} jest przeciążony. Przełączam na kolejny model.")
                    model_unavailable = True
                    await asyncio.sleep(1.2)
                    break

                if is_429:
                    print(f"KLUCZ {index + 1} ma limit/quota. Próba kolejnego klucza.")
                    continue

                raise RuntimeError(error_text)

        if model_unavailable:
            continue

    raise RuntimeError(f"Wszystkie modele/klucze Gemini zwróciły błąd. Ostatni błąd: {last_error}")


def error_result(name: str, error_text: str, note: str) -> dict:
    return {
        "error": error_text,
        "name": name,
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "sugar": 0,
        "fat": 0,
        "saturated_fat": 0,
        "fiber": 0,
        "salt": 0,
        "confidence": 0,
        "note": note,
        "microNutrients": empty_micro_map(),
        "additives": [],
    }


@app.post("/analyze-meal")
async def analyze_meal(
    file: UploadFile = File(...),
    description: str = Form("")
):
    image_bytes = await file.read()
    mime_type = get_mime_type(image_bytes)
    prompt = build_prompt(description)

    try:
        if AI_PROVIDER == "openai":
            return await analyze_with_openai(prompt, image_bytes, mime_type)

        if AI_PROVIDER == "gemini":
            return await analyze_with_gemini(prompt, image_bytes, mime_type)

        if AI_PROVIDER == "openai_then_gemini":
            try:
                return await analyze_with_openai(prompt, image_bytes, mime_type)
            except Exception as openai_error:
                print("OPENAI PADLO, PROBUJE GEMINI:", openai_error)
                return await analyze_with_gemini(prompt, image_bytes, mime_type)

        if AI_PROVIDER == "gemini_then_openai":
            try:
                return await analyze_with_gemini(prompt, image_bytes, mime_type)
            except Exception as gemini_error:
                print("GEMINI PADLO, PROBUJE OPENAI:", gemini_error)
                return await analyze_with_openai(prompt, image_bytes, mime_type)

        return error_result(
            "Błąd konfiguracji",
            f"Nieznany AI_PROVIDER: {AI_PROVIDER}",
            "Ustaw w Render AI_PROVIDER=openai albo AI_PROVIDER=gemini.",
        )

    except Exception as error:
        error_text = str(error)
        print("BLAD ANALIZY AI:", error_text)

        return error_result(
            "Błąd analizy",
            error_text,
            "Backend złapał błąd podczas analizy zdjęcia. Sprawdź Logs w Render albo popraw konfigurację API.",
        )
