# Aktualizacja bezpieczeństwa Render — 25.10.2026

Zmiana dotyczy certyfikatów używanych przez klienty HTTPS i procesu budowania.
Nie zmienia kontraktów endpointów ani modeli AI. Nie jest dowodem naprawienia
produkcji: wersje i zachowanie działającej usługi trzeba sprawdzić osobno.

## Co zawiera zmiana

- `requirements.txt`: jawne minimum `certifi>=2026.10.19`.
- `requirements.lock.txt`: przypięte 37 zależności wykonawczych rozwiązywanych
  na Linux/Python 3.12.14. To snapshot wersji, bez hashy; nie jest uniwersalnym
  lockfile dla wszystkich wersji Pythona i systemów.
- `scripts/render_build.sh`: aktualizuje pip do 26.2.2 z artefaktu wskazanego
  przez Python Security, sprawdza SHA256, instaluje zależności z ograniczeniami
  snapshotu, wykonuje `pip check` i kontrolę wersji/ustawień CA.
- `scripts/check_tls.py`: diagnostyka wersji i opcjonalne testy odrzucenia
  dwóch hostów wskazanych przez Render, przez Requests i oba SDK AI.
- Testy regresji kontraktów HTTP oraz rozróżnienia błędu certyfikatu i awarii sieci.

## 1. Zapisz stan przed zmianą

W Render sprawdź repozytorium, gałąź, Root Directory, SHA wdrożenia, runtime
(native Python czy Docker), Python, Build Command i Start Command. SHA źródłowe
tej propozycji: `7ebdd4ec89ba13c685e8d625ffa6743341226778`.
Zapisz ustawienia i identyfikator ostatniego sprawnego wdrożenia. Nie kopiuj
wartości sekretów do raportu ani GitHuba.

W Shell działającej usługi, przed aktualizacją:

```bash
python --version
python -m pip --version
python -m pip freeze > /tmp/kalorie-before-update.txt
python -m pip check
```

Zachowaj snapshot poza nietrwałą instancją. Sprawdź, czy ustawiono
`REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `SSL_CERT_FILE`, `SSL_CERT_DIR` i proxy.
Nie usuwaj potrzebnych firmowych CA bez analizy. Aktualizacja certifi nie
naprawia automatycznie osobnego, starego pliku PEM. Skrypt budowania zatrzyma
się przy nadpisaniu CA; wtedy zweryfikuj bundle i świadomie dostosuj kontrolę.

## 2. Zbuduj i sprawdź środowisko testowe

Użyj tej samej wspieranej wersji Pythona co na produkcji. Nie zmieniaj wersji
Pythona jednocześnie bez osobnych testów. Snapshot sprawdzono na 3.12.14;
jeżeli Render używa innej wersji, najpierw sprawdź instalację i testy właśnie
na niej. Nowe SDK mogą różnić się od starych wersji produkcyjnych.

Build Command w testowej usłudze native Python:

```bash
bash scripts/render_build.sh
```

Zachowaj dotychczasowy Start Command; typowy przykład, jeżeli usługa go używa:

```bash
uvicorn main:app --host 0.0.0.0 --port "$PORT"
```

W oddzielnym środowisku testowym po instalacji zależności:

```bash
python -m pip install pytest
python -m pytest -q
```

Oczekiwany wynik: 11 testów. Wywołania dostawców w testach API są mockowane.
Testy nie potwierdzają działania prawdziwych kluczy, limitów ani modeli AI.

Sprawdzenie znanych podatności, w oddzielnym venv z narzędziem pip-audit:

```bash
python -m pip_audit -r requirements.lock.txt --no-deps --disable-pip
```

Skan obejmuje wszystkie 37 przypiętych paczek, ale nie bazowy system/CPython
ani nieznane podatności. Przed następną aktualizacją odtwórz snapshot w czystym
venv z samymi zależnościami aplikacji i powtórz testy oraz skan.

## 3. Przebuduj usługę

Dla native Python: po zatwierdzeniu i scaleniu poprawki ustaw Build Command
jak wyżej. Wybierz Manual Deploy → Clear build cache & deploy. Nowy build
Render odświeża systemowy magazyn CA, a skrypt odświeża certifi i pip.
Samo ponowne uruchomienie starego procesu nie wystarcza.

Dla Docker: niniejsze repo nie ma Dockerfile. Jeżeli Render korzysta z obrazu
z innego źródła, zaktualizuj tam wspierany obraz bazowy i pakiet
`ca-certificates`, wykonaj bezpieczny bootstrap pip i instalację z constraints,
zbuduj obraz bez cache z odświeżeniem bazy i wdroż nowy digest. Aktualizacja
samego repo bez przebudowania używanego obrazu nie wystarczy.

Node.js/PHP: w przejrzanym repo brak kodu i manifestów tych runtime'ów.
Sprawdź również Build Command/obraz. Samo zainstalowanie Node w środowisku
Render nie oznacza, że aplikacja go używa. Jeśli jest używany do budowania,
zastosuj również odpowiednią aktualizację z komunikatu Render.

## 4. Sprawdź TLS z działającej instancji

Uruchom w Render Shell dla nowego wdrożenia, tym samym interpreterem i z tymi
samymi zmiennymi środowiskowymi co API:

```bash
python scripts/check_tls.py --network
python -m pip check
```

Oczekiwane: certifi ≥2026.10.19, pip ≥26.2.2, brak niezbadanych nadpisań CA
i sześć wyników `PASS_CERTIFICATE_REJECTED`. Skrypt używa fikcyjnych kluczy
w SDK i nie wysyła kluczy produkcyjnych do hostów testowych. SDK zachowują
domyślny transport stosowany w main.py, zmieniona jest tylko baza URL testu.

`FAIL_TLS_ACCEPTED` (także HTTP 401/403/404/500) oznacza, że TLS zaakceptowano.
`INCONCLUSIVE` oznacza konieczność dalszego sprawdzenia DNS, sieci lub proxy.
Odrzucenie może też pochodzić od proxy lub innego błędu certyfikatu; interpretuj
wynik łącznie z wersjami i poprawnym działaniem rzeczywistych dostawców.
Nie zastępuj testu samym curl — może używać innego magazynu CA.

## 5. Sprawdź rzeczywiste API i aplikację

Ustaw rzeczywisty adres, bez końcowego ukośnika:

```bash
API_BASE='https://TWOJA-USLUGA.onrender.com'
curl --fail-with-body --max-time 120 "$API_BASE/"
curl --fail-with-body --max-time 120 "$API_BASE/openapi.json" > /tmp/kalorie-openapi.json
curl --fail-with-body --max-time 120 -X POST "$API_BASE/v1/translate" -H 'Content-Type: application/json' --data '{"text":"jabłko","target":"en"}'
curl --fail-with-body --max-time 120 "$API_BASE/v1/prices/barcode/3017620422003"
```

Tłumaczenie wymaga skonfigurowanego klucza Google; 503 z powodu braku klucza
nie jest błędem TLS. Pusta lista cen może być prawidłową odpowiedzią.

Z aplikacji wykonaj analizę zwykłego zdjęcia osobno przez OpenAI i Gemini,
czat, analizę przepisu z tekstu oraz tłumaczenie. Sprawdź odpowiedź JSON,
nie tylko HTTP 200: część istniejących endpointów zwraca pole `error` mimo 200.
Potwierdź pola wartości odżywczych, `reply`, `aiProvider`, `aiModel` i brak
niezamierzonego fallbacku. Te wywołania mogą zużywać płatne limity.
Sprawdź logi pod kątem błędów TLS, 5xx, uwierzytelnienia i limitów; nie publikuj
treści posiłków, czatu ani sekretów. Monitoruj działanie po wdrożeniu.

## 6. Gdy wystąpi regresja

Przywróć poprzedni kod aplikacji, zachowując poprawione certifi, pip i CA,
a następnie zbuduj i sprawdź go ponownie. Jeżeli problemem jest SDK, użyj
poprzedniej kompatybilnej wersji po skanie bezpieczeństwa. Nie przywracaj
w ciemno starego obrazu/artefaktu zawierającego skompromitowane zaufanie CA.

## Źródła

- https://render.com/docs/advisory-emudhra
- https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576
- https://requests.readthedocs.io/en/latest/user/advanced/
- https://www.python-httpx.org/advanced/ssl/
