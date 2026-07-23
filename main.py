import asyncio
import base64
import json
import os
import re
from typing import Optional

import requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
        "status": "Backend AI dziala - OpenAI/Gemini nutrition v6 provider per request",
        "default_provider": AI_PROVIDER,
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

    result = {
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

    # Dodatkowe pola z odpowiedzi AI (pantry_category, brand, barcode,
    # search_queries, portion_grams...) przechodza do klienta - prosza o nie
    # prompty FoodRadar (kategoria produktu, import przez obraz). Pola
    # wyliczone wyzej maja pierwszenstwo.
    for key, value in result_json.items():
        if key not in result:
            result[key] = value

    return result


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
- Analizuj tylko posiłek, produkt albo opakowanie jako całość.
- Nie zwracaj listy ingredients.
- Nie rozbijaj posiłku na składniki, chyba że użytkownik wyraźnie opisze porcję i trzeba oszacować całość.
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
    description: str = Form(""),
    ai_provider: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
):
    image_bytes = await file.read()
    mime_type = get_mime_type(image_bytes)
    prompt = build_prompt(description)

    selected_provider = (ai_provider or provider or AI_PROVIDER).strip().lower()

    try:
        if selected_provider == "openai":
            return await analyze_with_openai(prompt, image_bytes, mime_type)

        if selected_provider == "gpt":
            return await analyze_with_openai(prompt, image_bytes, mime_type)

        if selected_provider == "gemini":
            return await analyze_with_gemini(prompt, image_bytes, mime_type)

        if selected_provider == "openai_then_gemini":
            try:
                return await analyze_with_openai(prompt, image_bytes, mime_type)
            except Exception as openai_error:
                print("OPENAI PADLO, PROBUJE GEMINI:", openai_error)
                return await analyze_with_gemini(prompt, image_bytes, mime_type)

        if selected_provider == "gemini_then_openai":
            try:
                return await analyze_with_gemini(prompt, image_bytes, mime_type)
            except Exception as gemini_error:
                print("GEMINI PADLO, PROBUJE OPENAI:", gemini_error)
                return await analyze_with_openai(prompt, image_bytes, mime_type)

        return error_result(
            "Błąd konfiguracji",
            f"Nieznany provider AI: {selected_provider}",
            "Ustaw provider jako openai, gpt albo gemini.",
        )

    except Exception as error:
        error_text = str(error)
        print("BLAD ANALIZY AI:", error_text)

        return error_result(
            "Błąd analizy",
            error_text,
            "Backend złapał błąd podczas analizy zdjęcia. Sprawdź Logs w Render albo popraw konfigurację API.",
        )


# ============================================================================
# Wspolne API ekosystemu (FoodRadar + Kalorie + Trainer): /v1/...
# Klucze API zyja WYLACZNIE tutaj (zmienne srodowiskowe Render), nigdy
# w aplikacjach. Kazdy endpoint zwraca czysty JSON o stalym kontrakcie.
# ============================================================================


def generate_text_json(prompt: str) -> dict:
    """Tekstowe zapytanie do Gemini (bez obrazu) z ta sama rotacja modeli
    i kluczy co analiza posilkow. Zwraca zdekodowany JSON."""
    last_error = None
    for model_name in GEMINI_MODELS:
        for index in range(len(GEMINI_API_KEYS)):
            try:
                current_client = get_gemini_client(index)
                response = current_client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                )
                return json.loads(clean_json_text(response.text or ""))
            except Exception as error:
                last_error = str(error)
                print(f"BLAD TEXT-JSON MODEL {model_name} KLUCZ {index + 1}:", last_error)
                continue
    raise RuntimeError(f"Wszystkie modele/klucze Gemini zwrocily blad: {last_error}")


def generate_openai_text_json(prompt: str) -> dict:
    """Tekstowy JSON przez OpenAI, bez sztucznego obrazka-placeholdera."""
    client = get_openai_client()
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        max_output_tokens=8000,
    )
    result_text = getattr(response, "output_text", "") or ""
    if not result_text:
        chunks = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if hasattr(content, "text"):
                    chunks.append(content.text)
        result_text = "\n".join(chunks)
    return json.loads(clean_json_text(result_text))


class TranslateRequest(BaseModel):
    text: str
    target: str
    source: str = ""


# Prosta pamiec tlumaczen w procesie - ogranicza zapytania do Google.
_translation_cache: dict = {}


@app.post("/v1/translate")
def translate(body: TranslateRequest):
    api_key = os.environ.get("GOOGLE_TRANSLATOR_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Tlumaczenia niedostepne: brak GOOGLE_TRANSLATOR_API_KEY.",
        )
    text = body.text.strip()
    target = body.target.strip().lower()
    if not text or not target:
        raise HTTPException(status_code=400, detail="Podaj text i target.")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Tekst za dlugi (max 2000).")

    cache_key = f"{target}|{body.source}|{text.lower()}"
    cached = _translation_cache.get(cache_key)
    if cached is not None:
        return {"translation": cached, "cached": True}

    payload = {"q": text, "target": target, "format": "text"}
    if body.source.strip():
        payload["source"] = body.source.strip().lower()
    response = requests.post(
        "https://translation.googleapis.com/language/translate/v2",
        params={"key": api_key},
        json=payload,
        timeout=12,
    )
    if response.status_code != 200:
        print("BLAD GOOGLE TRANSLATE:", response.status_code, response.text[:300])
        raise HTTPException(
            status_code=502,
            detail=f"Google Translate zwrocil {response.status_code}.",
        )
    data = response.json()
    try:
        translation = data["data"]["translations"][0]["translatedText"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Niepoprawna odpowiedz Google.")

    if len(_translation_cache) > 5000:
        _translation_cache.clear()
    _translation_cache[cache_key] = translation
    return {"translation": translation, "cached": False}


@app.get("/v1/prices/barcode/{gtin}")
def prices_by_barcode(gtin: str):
    """Realne ceny produktu po kodzie z Open Prices (otwarte API spolecznosci
    Open Food Facts) - bez klucza. Zwracamy tylko to, co zrodlo potwierdza."""
    code = "".join(ch for ch in gtin if ch.isdigit())
    if len(code) < 8:
        raise HTTPException(status_code=400, detail="Za krotki kod GTIN.")
    response = requests.get(
        "https://prices.openfoodfacts.org/api/v1/prices",
        params={"product_code": code, "size": 20, "order_by": "-date"},
        headers={"User-Agent": "FoodRadar-ecosystem-backend/1.0"},
        timeout=12,
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Open Prices zwrocil {response.status_code}.",
        )
    items = response.json().get("items", [])
    offers = []
    for item in items:
        price = item.get("price")
        if price is None:
            continue
        location = item.get("location") or {}
        offers.append({
            "store": location.get("osm_name") or "Sklep?",
            "address": ", ".join(
                part for part in [
                    location.get("osm_address_city"),
                    location.get("osm_address_country"),
                ] if part
            ),
            "latitude": location.get("osm_lat") or 0,
            "longitude": location.get("osm_lon") or 0,
            "price": float(price),
            "currency": item.get("currency") or "",
            "date": item.get("date") or "",
            "source": "Open Prices",
        })
    return {"gtin": code, "offers": offers, "source": "Open Prices"}


class PriceEstimateRequest(BaseModel):
    name: str
    country: str = "Polska"


@app.post("/v1/prices/estimate")
def estimate_price(body: PriceEstimateRequest):
    """SZACUNEK ceny skladnika/produktu przez AI - zawsze jawnie oznaczony
    estimate=true. Aplikacja pokazuje go jako orientacyjny, nigdy jako
    potwierdzona cene sklepowa."""
    name = body.name.strip()
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Podaj nazwe produktu.")
    prompt = f"""
Oszacuj typowa cene detaliczna produktu spozywczego w kraju: {body.country}.
Produkt: {name}
Zwroc WYLACZNIE poprawny JSON bez markdown:
{{
  "price_low": typowa_najnizsza_cena_liczba,
  "price_high": typowa_najwyzsza_cena_liczba,
  "currency": "waluta np. zl / EUR",
  "package": "typowe opakowanie np. 1 kg / 500 g / sztuka",
  "note": "krotka uwaga po polsku"
}}
Jesli nie potrafisz sensownie oszacowac, zwroc price_low i price_high = 0.
"""
    try:
        data = generate_text_json(prompt)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error))
    return {
        "name": name,
        "estimate": True,
        "price_low": to_float(data.get("price_low"), 0),
        "price_high": to_float(data.get("price_high"), 0),
        "currency": str(data.get("currency", "zl")),
        "package": str(data.get("package", "")),
        "note": str(data.get("note", "")),
        "source": "szacunek AI",
    }


class RecipeGenerateRequest(BaseModel):
    description: str
    servings: int = 2
    meal_type: str = ""


def build_recipe_prompt(
    description: str,
    servings: int,
    meal_type: str = "",
    *,
    from_image: bool = False,
) -> str:
    """Kontrakt AI przepisu z audytowalnym rozliczeniem skladnikow."""
    meal_type_line = (
        f"Rodzaj posilku: {meal_type.strip()}" if meal_type.strip() else ""
    )
    image_line = (
        "Zdjecie moze przedstawic gotowe danie, skladniki, etykiete albo "
        "zrzut przepisu. Uzyj go razem z opisem."
        if from_image
        else ""
    )
    return f"""
Przygotuj kompletny przepis kulinarny po polsku na podstawie danych uzytkownika.
Opis uzytkownika: {description}
Liczba porcji: {servings}
{meal_type_line}
{image_line}

Zwroc WYLACZNIE poprawny JSON bez markdown:
{{
  "name": "nazwa dania",
  "description": "2-3 zdania opisu",
  "category": "jedna z: Domowe, Sniadanie, Obiad, Kolacja, Zupa, Sos, Deser, Przekaska, Napoj",
  "time_of_day": "jedna z: Dowolnie, Sniadanie, Obiad, Kolacja, Przekaska",
  "tags": ["krotkie tagi"],
  "ingredients": ["czytelna linia skladnika z iloscia i stanem produktu"],
  "ingredient_details": [
    {{
      "line": "500 g makaronu suchego (masa przed gotowaniem)",
      "name": "makaron pszenny",
      "amount_grams": 500,
      "state": "dry",
      "included": true,
      "calories_per_100g": 350,
      "protein_per_100g": 12,
      "carbs_per_100g": 72,
      "sugar_per_100g": 3,
      "fat_per_100g": 1.5,
      "saturated_fat_per_100g": 0.3,
      "fiber_per_100g": 3,
      "salt_per_100g": 0.02
    }}
  ],
  "steps": ["krok 1", "krok 2"],
  "servings": {servings},
  "prep_minutes": 30,
  "difficulty": "latwy/sredni/trudny",
  "equipment": ["potrzebny sprzet"],
  "calories": kcal_calego_dania,
  "protein": g_bialka_calego_dania,
  "carbs": g_weglowodanow_calego_dania,
  "sugar": g_cukru_calego_dania,
  "fat": g_tluszczu_calego_dania,
  "saturated_fat": g_tluszczow_nasyconych_calego_dania,
  "fiber": g_blonnika_calego_dania,
  "salt": g_soli_calego_dania,
  "fluid_ml": ml_plynow_w_calym_daniu,
  "ingredients_weight_grams": masa_skladnikow_przed_przygotowaniem,
  "total_weight_grams": szacowana_masa_gotowego_dania,
  "confidence": liczba_od_0_do_1,
  "allergens": ["alergeny"],
  "substitutes": ["SKLADNIK -> ZAMIENNIK (proporcja)"],
  "storage": "1-2 zdania o przechowywaniu"
}}

ZASADY OBLICZEN:
- Podane przez uzytkownika ilosci sa nadrzedne. Nie zmieniaj 500 g na 300 g.
- Dla makaronu, ryzu, kaszy i platkow podanych przed gotowaniem uzyj masy
  SUCHEJ i wartosci odzywczych suchego produktu. Woda wchlonieta podczas
  gotowania zwieksza tylko total_weight_grams, nigdy kalorie ani makro.
- Dla miesa i warzyw uzyj masy surowej, jezeli opis nie mowi inaczej.
- Dla konserw podanych jako odsadzone uzyj masy po odsaczeniu. Tunczyk w
  sosie wlasnym/solance nie zawiera oleju z puszki.
- Nie dodawaj oleju, masla, sera ani sosu, jesli uzytkownik ich nie podal.
  Sugestie opcjonalne oznacz included=false i nie wliczaj ich do sumy.
- W ingredient_details podaj jadalna mase uzyta do obliczen i wartosci na
  100 g dokladnie dla stanu dry/raw/drained/cooked/as_sold.
- Wody do gotowania, ktora jest odlewana, nie umieszczaj w ingredient_details.
  Wode pozostajaca w daniu mozna dodac z zerowymi wartosciami odzywczymi.
- Pola calories/protein/carbs/sugar/fat/fiber/salt maja oznaczac SUME CALEGO
  przepisu. Serwer i tak przeliczy je ponownie z ingredient_details.
- ingredients_weight_grams oznacza sume mas uzytych do obliczen przed
  przygotowaniem. total_weight_grams oznacza mase po ugotowaniu/upieczeniu.
- Kazdy skladnik kaloryczny musi miec osobny wpis ingredient_details.
- Gdy ilosc jest niewidoczna, oszacuj ja ostroznie i obniz confidence.
"""


def normalize_recipe_result(data: dict, servings: int) -> dict:
    """Sumuje makro z mas i wartosci /100 g zamiast ufac jednej sumie AI."""
    result = dict(data)
    details = result.get("ingredient_details")
    normalized_details = []
    ingredient_lines = []
    totals = {
        "calories": 0.0,
        "protein": 0.0,
        "carbs": 0.0,
        "sugar": 0.0,
        "fat": 0.0,
        "saturated_fat": 0.0,
        "fiber": 0.0,
        "salt": 0.0,
    }
    ingredients_weight = 0.0
    calculated_count = 0

    if isinstance(details, list):
        for raw_item in details:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            included_raw = item.get("included", True)
            included = not (
                included_raw is False
                or str(included_raw).strip().lower() in ("false", "0", "no", "nie")
            )
            amount = max(
                0.0,
                to_float(
                    item.get("amount_grams")
                    or item.get("grams")
                    or item.get("edible_grams"),
                    0,
                ),
            )
            name = str(item.get("name") or "skladnik").strip()
            state = str(item.get("state") or "as_sold").strip().lower()
            line = str(item.get("line") or "").strip()
            if not line:
                amount_text = f"{amount:g} g " if amount > 0 else ""
                line = f"{amount_text}{name} ({state})".strip()
            if not included and "opcjonal" not in line.lower():
                line += " (opcjonalnie, poza wyliczeniem)"
            ingredient_lines.append(line)

            per100 = {
                key: max(
                    0.0,
                    to_float(
                        item.get(f"{key}_per_100g")
                        or item.get(f"{key}_per100")
                        or item.get(f"{key}_100g"),
                        0,
                    ),
                )
                for key in totals
            }
            if per100["calories"] <= 0 and any(
                per100[key] > 0 for key in ("protein", "carbs", "fat")
            ):
                per100["calories"] = (
                    per100["protein"] * 4
                    + per100["carbs"] * 4
                    + per100["fat"] * 9
                )

            item.update(
                {
                    "line": line,
                    "name": name,
                    "state": state,
                    "amount_grams": round(amount, 2),
                    "included": included,
                    **{f"{key}_per_100g": round(value, 4) for key, value in per100.items()},
                }
            )
            normalized_details.append(item)

            if not included or amount <= 0:
                continue
            ingredients_weight += amount
            if any(value > 0 for value in per100.values()):
                calculated_count += 1
            factor = amount / 100
            for key, value in per100.items():
                totals[key] += value * factor

    if ingredient_lines:
        result["ingredients"] = ingredient_lines
    result["ingredient_details"] = normalized_details
    result["servings"] = max(1, min(999, int(servings)))

    if calculated_count > 0:
        # Python stosuje round-half-to-even, a aplikacja Dart zaokragla .5
        # od zera. Dla dodatnich wartosci odzywczych zachowujemy semantyke UI.
        whole = lambda value: int(value + 0.5)
        result["calories"] = whole(totals["calories"])
        result["protein"] = whole(totals["protein"])
        result["carbs"] = whole(totals["carbs"])
        result["sugar"] = whole(totals["sugar"])
        result["fat"] = whole(totals["fat"])
        result["saturated_fat"] = round(totals["saturated_fat"], 1)
        result["fiber"] = round(totals["fiber"], 1)
        result["salt"] = round(totals["salt"], 2)
        result["ingredients_weight_grams"] = round(ingredients_weight, 1)
        result["nutrition_calculation_note"] = (
            "Makro zsumowano ze skladnikow wedlug masy suchej, surowej lub "
            "odsaczonej. Zmiana masy podczas gotowania wplywa tylko na mase "
            "gotowego dania i wartosci na 100 g."
        )
        result["nutrition_calculation_method"] = "ingredient_sum_v1"

    total_weight = max(
        0.0,
        to_float(
            result.get("total_weight_grams")
            or result.get("totalWeightGrams"),
            0,
        ),
    )
    if total_weight <= 0 and ingredients_weight > 0:
        total_weight = ingredients_weight
    result["total_weight_grams"] = round(total_weight, 1)
    return result


@app.post("/v1/recipes/generate")
def generate_recipe(body: RecipeGenerateRequest):
    """Generowanie przepisu przez AI - osobny kontrakt JSON (nie schemat
    posilku z /analyze-meal). Zrodlo w aplikacji oznaczane jako AI."""
    description = body.description.strip()
    if len(description) < 3:
        raise HTTPException(status_code=400, detail="Opisz danie.")
    servings = max(1, min(999, body.servings))
    prompt = build_recipe_prompt(description, servings, body.meal_type)
    try:
        data = normalize_recipe_result(generate_text_json(prompt), servings)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error))
    print("PRZEPIS AI:", str(data)[:400])
    if not data.get("name") or not data.get("ingredients") or not data.get("steps"):
        raise HTTPException(
            status_code=502,
            detail="AI zwrocilo niepelny przepis - sprobuj ponownie.",
        )
    data["source"] = "AI"
    return data


@app.post("/v1/recipes/analyze")
async def analyze_recipe(
    file: Optional[UploadFile] = File(None),
    description: str = Form(""),
    servings: int = Form(1),
    meal_type: str = Form(""),
    ai_provider: Optional[str] = Form(None),
):
    """Uzupelnia edytor przepisu z opisu i/lub zdjecia bez zmiany /analyze-meal."""
    image_bytes = await file.read() if file is not None else b""
    if len(image_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Zdjecie jest za duze (max 12 MB).")
    if not image_bytes and len(description.strip()) < 3:
        raise HTTPException(status_code=400, detail="Dodaj zdjecie albo opis przepisu.")

    safe_servings = max(1, min(999, int(servings)))
    prompt = build_recipe_prompt(
        description.strip(),
        safe_servings,
        meal_type,
        from_image=True,
    )
    selected_provider = (ai_provider or AI_PROVIDER).strip().lower()
    mime_type = get_mime_type(image_bytes) if image_bytes else ""

    async def analyze_with(provider_name: str) -> dict:
        if provider_name in ("openai", "gpt"):
            if image_bytes:
                return await analyze_with_openai(prompt, image_bytes, mime_type)
            return generate_openai_text_json(prompt)
        if image_bytes:
            return await analyze_with_gemini(prompt, image_bytes, mime_type)
        return generate_text_json(prompt)

    try:
        if selected_provider in ("openai", "gpt"):
            data = await analyze_with(selected_provider)
        elif selected_provider == "gemini":
            data = await analyze_with("gemini")
        elif selected_provider == "openai_then_gemini":
            try:
                data = await analyze_with("openai")
            except Exception:
                data = await analyze_with("gemini")
        elif selected_provider == "gemini_then_openai":
            try:
                data = await analyze_with("gemini")
            except Exception:
                data = await analyze_with("openai")
        else:
            raise HTTPException(status_code=400, detail="Nieznany dostawca AI.")
        data = normalize_recipe_result(data, safe_servings)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error))

    if not data.get("name") or not data.get("ingredients"):
        raise HTTPException(
            status_code=502,
            detail="AI zwrocilo niepelne dane przepisu - doprecyzuj opis.",
        )
    data["source"] = "AI"
    return data


class ImportUrlRequest(BaseModel):
    url: str


@app.post("/v1/recipes/import-url")
def import_recipe_url(body: ImportUrlRequest):
    """Pobiera strone przepisu PO STRONIE BACKENDU (przegladarki blokuja
    aplikacje po User-Agent) i zwraca HTML do parsowania JSON-LD Recipe
    w aplikacji. Limit rozmiaru chroni przed ogromnymi stronami."""
    url = body.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Podaj pelny link http(s).")
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "pl,nl;q=0.9,en;q=0.8",
            },
            timeout=15,
            allow_redirects=True,
        )
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Nie pobrano strony: {error}")
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Strona zwrocila HTTP {response.status_code}.",
        )
    html = response.text
    if len(html) > 1_500_000:
        html = html[:1_500_000]
    return {"url": url, "html": html}
