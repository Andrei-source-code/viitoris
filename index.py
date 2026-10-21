import os
import re
import json
import math
import time
import unicodedata

import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------- configurare
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Poți fixa modelul din Vercel (variabila GEMINI_MODEL); altfel se încearcă pe rând
MODELE_GEMINI = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-flash-lite-latest",
                             "gemini-flash-latest", "gemini-3.5-flash"] if m]

# Google Maps (Places API New) - sursa principală pentru competitori
GOOGLE_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
# GOOGLE_RATINGS=1 adaugă rating + nr. recenzii (SKU mai scump la Google)
GOOGLE_RATINGS = os.environ.get("GOOGLE_RATINGS") == "1"

# Groq rămâne ca rezervă, doar dacă ai setat GROQ_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    client = None

# Toate fisierele sunt in acelasi folder cu index.py (radacina repo-ului)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
CSV_PATH = os.path.join(DATA_DIR, "baza_cariere.csv")
JSON_PATH = os.path.join(DATA_DIR, "surse_piata.json")

CATEGORII = ['REALIST', 'INVESTIGATIV', 'ARTISTIC',
             'SOCIAL', 'INTREPRINZATOR', 'CONVENTIONAL']


# ------------------------------------------------------------- încărcare date
LITERE = {"REALIST": "R", "INVESTIGATIV": "I", "ARTISTIC": "A",
          "SOCIAL": "S", "INTREPRINZATOR": "E", "CONVENTIONAL": "C"}
NUME_LITERA = {v: k for k, v in LITERE.items()}
ORDINE = "RIASEC"
NUME_AFISAT = {"R": "Realist", "I": "Investigativ", "A": "Artistic",
               "S": "Social", "E": "Întreprinzător", "C": "Convențional"}
DESCRIERI = {
    "R": "Îți place munca practică: unelte, mașini, natură, construit și reparat lucruri concrete.",
    "I": "Te atrage înțelegerea lucrurilor: cercetare, analiză, știință, probleme logice complexe.",
    "A": "Ai nevoie de creativitate și libertate de exprimare: artă, scris, design, muzică, idei originale.",
    "S": "Te motivează oamenii: să ajuți, să înveți, să îngrijești și să lucrezi în echipă.",
    "E": "Îți place să conduci, să convingi și să iei inițiativa: afaceri, vânzări, proiecte, influență.",
    "C": "Preferi ordinea și precizia: organizare, cifre, proceduri clare, lucrul bine structurat.",
}


def curata_cariere(df):
    """Unește rândurile de continuare (pașii 2, 3, 4...) cu cariera lor și normalizează datele."""
    if "id_cariera" not in df.columns or "cod_psihologic" not in df.columns:
        return df
    df = df.copy()
    df["_g"] = df["id_cariera"].notna().cumsum()
    df = df[df["_g"] > 0]
    fixe = [c for c in ("id_cariera", "cod_psihologic", "salariu_mediu", "cod_holland")
            if c in df.columns]
    text = [c for c in df.columns if c not in fixe and c != "_g"]

    def lipeste(serie):
        return "\n".join(str(x).strip() for x in serie.dropna() if str(x).strip())

    agg = {c: "first" for c in fixe}
    agg.update({c: lipeste for c in text})
    out = df.groupby("_g", sort=True).agg(agg).reset_index(drop=True)

    out["id_cariera"] = out["id_cariera"].astype(float).astype(int)
    out["cod_psihologic"] = out["cod_psihologic"].astype(str).str.strip().str.upper()
    out["salariu_mediu"] = pd.to_numeric(out.get("salariu_mediu"), errors="coerce")

    def cod3(rand):
        prim = LITERE.get(rand["cod_psihologic"], "")
        brut = rand.get("cod_holland")
        brut = re.sub(r"[^RIASEC]", "", str(brut).upper()) if isinstance(brut, str) else ""
        brut = "".join(dict.fromkeys(brut))[:3]
        return brut or prim
    out["cod_holland"] = out.apply(cod3, axis=1)
    return out


def incarca_csv():
    for enc in ("utf-8-sig", "cp1250", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc, sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            # numele reale din CSV conțin explicații ("facultate_ro / Școală"),
            # le aducem la numele scurte folosite în cod
            redenumiri = {}
            for c in df.columns:
                baza = c.split("/")[0].split("(")[0].strip()
                if baza != c:
                    redenumiri[c] = baza
            df = df.rename(columns=redenumiri)
            df = curata_cariere(df)
            print(f"baza_cariere.csv incarcat (encoding: {enc}), {len(df)} cariere.")
            return df
        except FileNotFoundError:
            print(f"Fisierul {CSV_PATH} nu exista.")
            return None
        except Exception as e:
            print(f"Incercare {enc} esuata: {e}")
    return None


def incarca_surse():
    try:
        with open(JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"surse_piata.json nu a putut fi incarcat: {e}")
        return {}


df_cariere = incarca_csv()
SURSE = incarca_surse()


# ------------------------------------------------------------------- utilitare
def fara_diacritice(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


def citeste_date():
    """Acceptă JSON (din JavaScript) sau form-encoded (webhook Forminator)."""
    js = request.get_json(silent=True)
    if isinstance(js, dict):
        return js
    return {k: (v if len(v) > 1 else v[0]) for k, v in request.form.lists()}


# ------------------------------------------------------------- partea CARIERĂ
def calculeaza_punctaje(data):
    if any(f'punctaj_{c.lower()}' in data for c in CATEGORII):
        out = {}
        for c in CATEGORII:
            try:
                out[c] = int(float(data.get(f'punctaj_{c.lower()}', 0)))
            except (TypeError, ValueError):
                out[c] = 0
        return out

    out = {c: 0 for c in CATEGORII}
    for valoare in data.values():
        for v in (valoare if isinstance(valoare, list) else [valoare]):
            cod = fara_diacritice(str(v)).strip().upper()
            if cod in out:
                out[cod] += 1
    return out


# --------------------------------------------------------- partea ANTREPRENOR
UA = "viitoris-business-analyzer/2.0 (proiect educational Viitoris)"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
CACHE_COMPETITORI = {}      # (oras, tip) -> (moment, rezultat); ține câteva ore
TTL_CACHE = 6 * 3600
LIMITA_ELEMENTE = 2000
# Timpul total al unei analize, în secunde. Trebuie să fie MAI MIC decât maxDuration din vercel.json
# (60 acolo -> 55 aici). Poți schimba din Vercel prin variabila TIMP_TOTAL_SECUNDE.
TIMP_TOTAL = int(os.environ.get("TIMP_TOTAL_SECUNDE", "55"))
REZERVA_GROQ = 10   # secunde păstrate pentru Groq, doar dacă ai GROQ_API_KEY

DISCLAIMER = ("Analiza este orientativă și NU garantează succesul afacerii. "
              "Rezultatul real depinde de locația exactă, de execuție și management, "
              "de calitatea produselor sau serviciilor și de condițiile pieței. "
              "Numărul și lista competitorilor provin din harta folosită ca sursă și pot fi incomplete.")

# cheie -> (cuvinte cheie, filtre OpenStreetMap)
# ORDINEA CONTEAZĂ: "salon" înaintea lui "bar" (ca "barber" să nu fie bar)
TIPURI = {
    "salon": (["salon", "barber", "frizer", "coafor", "infrumuset", "beauty",
               "manichiur", "cosmetic"],
              ['["shop"~"^(hairdresser|beauty)$"]']),
    "patiserie": (["patiser", "brutar", "covrig", "panific", "plachin"],
                  ['["shop"="bakery"]']),
    "cafenea": (["cafenea", "cafe", "coffee", "ceainar"],
                ['["amenity"="cafe"]']),
    "restaurant": (["restaurant", "bistro", "pizz", "fast", "mancare", "grill"],
                   ['["amenity"~"^(restaurant|fast_food)$"]']),
    "fitness": (["fitness", "gym", "sala de sport", "sport"],
                ['["leisure"="fitness_centre"]']),
    "bar": (["bar", "pub", "club"],
            ['["amenity"~"^(bar|pub|nightclub)$"]']),
    "stomatologie": (["stomatolog", "dentar", "dentist"], ['["amenity"="dentist"]']),
    "farmacie": (["farmacie"], ['["amenity"="pharmacy"]']),
    "service_auto": (["service auto", "auto", "vulcaniz", "service"],
                     ['["shop"="car_repair"]']),
}

ALOCARE_IMPLICITA = [
    ("Amenajare și adaptare spațiu", 22, "Renovări, mobilier, instalații, firmă și semnalistică."),
    ("Echipamente și dotări", 28, "Echipamentele principale necesare activității zilnice."),
    ("Stoc inițial și materiale", 10, "Prima aprovizionare și consumabile."),
    ("Autorizări, avize și juridic", 5, "Înființare firmă, avize sanitare/ISU, contabil."),
    ("Marketing și lansare", 10, "Rețele sociale, materiale, promoții de deschidere."),
    ("Personal și cheltuieli curente (primele luni)", 15, "Chirie, utilități, salarii până la stabilizarea încasărilor."),
    ("Fond de rezervă", 10, "Pentru costuri neprevăzute și lunile slabe de început."),
]


def detecteaza_tip(tip_afacere):
    t = fara_diacritice(tip_afacere).lower()
    for cheie, (cuvinte, _) in TIPURI.items():
        if any(c in t for c in cuvinte):
            return cheie
    return None


def buget_minim_pentru(cheie):
    """Returnează (buget_minim, nume_afacere, sursa) sau (None, None, None)."""
    info = SURSE.get(cheie) if cheie else None
    if not info:
        return None, None, None
    loc = next(iter(info.get("locatii", {}).values()), {})
    return loc.get("buget_minim_necesar"), info.get("nume"), loc.get("sursa_informatii")


# ------------------------------------------------------- competitori (hartă)
def geocodeaza_oras(oras):
    """Centru, bounding box și populație (dacă există) din Nominatim."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": oras, "format": "json", "limit": 1,
                    "countrycodes": "ro", "extratags": 1},
            headers={"User-Agent": UA, "Accept-Language": "ro"},
            timeout=8,
        )
        r.raise_for_status()
        rez = r.json()
        if not rez:
            return None
        x = rez[0]
        lat, lon = float(x["lat"]), float(x["lon"])
        s, n, w, e = [float(v) for v in x["boundingbox"]]
        # dacă zona e uriașă (ex. județ), o restrângem în jurul centrului
        if (n - s) > 0.5 or (e - w) > 0.7:
            s, n, w, e = lat - 0.08, lat + 0.08, lon - 0.12, lon + 0.12
        pop = None
        p = (x.get("extratags") or {}).get("population")
        if p:
            primul = re.split(r"[;\s]", str(p).strip())[0]
            cifre = re.sub(r"\D", "", primul)
            pop = int(cifre) if cifre else None
        return {"lat": lat, "lon": lon, "bbox": (s, w, n, e), "populatie": pop}
    except Exception as e:
        print(f"Avertisment Nominatim: {e}")
        return None


def ruleaza_overpass(query, deadline):
    """Returnează (elemente | None, lista de erori pe server)."""
    erori = []
    for url in OVERPASS_URLS:
        gazda = url.split("/")[2]
        ramas = deadline - time.time()
        if ramas < 3:
            erori.append(f"{gazda}: timp insuficient")
            break
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": UA}, timeout=min(8, ramas))
            if r.status_code != 200:
                erori.append(f"{gazda}: HTTP {r.status_code}")
                continue
            js = r.json()
            elemente = js.get("elements", [])
            remarca = str(js.get("remark", ""))
            if not elemente and ("error" in remarca.lower() or "timed out" in remarca.lower()):
                erori.append(f"{gazda}: {remarca[:70]}")
                continue
            return elemente, erori
        except Exception as e:
            erori.append(f"{gazda}: {str(e)[:60]}")
    return None, erori


def distanta_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def proceseaza_elemente(elements, centru):
    vazute, lista, total, fara_nume = set(), [], 0, 0
    for el in elements:
        tags = el.get("tags") or {}
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lon = el.get("lon", (el.get("center") or {}).get("lon"))
        nume = (tags.get("name") or tags.get("brand") or tags.get("operator") or "").strip()
        if lat is not None and lon is not None:
            k = (nume.lower(), round(lat, 3), round(lon, 3))
            if k in vazute:
                continue
            vazute.add(k)
        total += 1
        if not nume:
            fara_nume += 1
            continue
        strada = (tags.get("addr:street") or "").strip()
        nr = (tags.get("addr:housenumber") or "").strip()
        adresa = (f"{strada} {nr}".strip() or tags.get("addr:full") or "").strip()
        item = {"nume": nume, "adresa": adresa, "distanta_km": None, "link": None,
                "lat": lat, "lon": lon}
        if lat is not None and lon is not None:
            item["link"] = (f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}"
                            f"#map=18/{lat:.6f}/{lon:.6f}")
            if centru:
                item["distanta_km"] = round(distanta_km(centru[0], centru[1], lat, lon), 1)
        lista.append(item)
    if centru:
        lista.sort(key=lambda x: (x["distanta_km"] is None, x["distanta_km"] or 0))
    else:
        lista.sort(key=lambda x: x["nume"].lower())
    return {"total": total, "fara_nume": fara_nume, "lista": lista[:40],
            "plafonat": len(elements) >= LIMITA_ELEMENTE, "sursa": "OpenStreetMap"}


def gaseste_competitori(oras, cheie, geo, deadline):
    """Returnează dict cu total + listă, sau None dacă harta nu a răspuns."""
    if not cheie:
        return None, ["tip necunoscut"]
    uniune = "\n".join(f"nwr{f}{{FILTRU_ZONA}};" for f in TIPURI[cheie][1])
    if geo:
        s, w, n, e = geo["bbox"]
        zona = f"({s:.5f},{w:.5f},{n:.5f},{e:.5f})"
        query = (f"[out:json][timeout:25];\n(\n{uniune.replace('{FILTRU_ZONA}', zona)}\n);\n"
                 f"out center tags {LIMITA_ELEMENTE};")
        centru = (geo["lat"], geo["lon"])
    else:
        oras_safe = oras.replace('"', '').replace('\\', '')
        query = (f"[out:json][timeout:25];\n"
                 f"area[\"name\"=\"{oras_safe}\"][\"boundary\"=\"administrative\"]->.a;\n(\n"
                 f"{uniune.replace('{FILTRU_ZONA}', '(area.a)')}\n);\n"
                 f"out center tags {LIMITA_ELEMENTE};")
        centru = None
    elemente, erori = ruleaza_overpass(query, deadline)
    if elemente is None:
        return None, erori
    return proceseaza_elemente(elemente, centru), erori


# ---------------------------------------------------------- Google Maps (Places)
TERMENI_GOOGLE = {
    "cafenea": ("cafenea", "cafe"),
    "restaurant": ("restaurant", "restaurant"),
    "salon": ("salon de înfrumusețare frizerie coafor", None),
    "patiserie": ("patiserie brutărie", "bakery"),
    "fitness": ("sală de fitness", "gym"),
    "bar": ("bar pub", "bar"),
    "stomatologie": ("cabinet stomatologic", "dentist"),
    "farmacie": ("farmacie", "pharmacy"),
    "service_auto": ("service auto", "car_repair"),
}
RAZA_MAXIMA_KM = 15
PAGINI_GOOGLE = 3  # 3 x 20 = maximum 60 de rezultate


def gaseste_competitori_google(oras, tip_afacere, cheie, geo, deadline):
    """Returnează (dict | None, eroare). Folosește Places API (New) - Text Search."""
    if not GOOGLE_API_KEY:
        return None, "GOOGLE_MAPS_API_KEY nu este setată"
    termen, tip_inclus = TERMENI_GOOGLE.get(cheie, (tip_afacere, None))
    corp = {"textQuery": f"{termen} în {oras}, România",
            "languageCode": "ro", "regionCode": "RO", "pageSize": 20}
    if tip_inclus:
        corp["includedType"] = tip_inclus
    if geo:
        corp["locationBias"] = {"circle": {
            "center": {"latitude": geo["lat"], "longitude": geo["lon"]},
            "radius": 10000.0}}
    campuri = ("nextPageToken,places.id,places.displayName,places.formattedAddress,"
               "places.location,places.googleMapsUri")
    if GOOGLE_RATINGS:
        campuri += ",places.rating,places.userRatingCount"

    locuri, token, eroare = [], None, ""
    for _ in range(PAGINI_GOOGLE):
        ramas = deadline - time.time()
        if ramas < 4:
            break
        if token:
            corp["pageToken"] = token
        try:
            r = requests.post(
                "https://places.googleapis.com/v1/places:searchText",
                headers={"X-Goog-Api-Key": GOOGLE_API_KEY,
                         "X-Goog-FieldMask": campuri,
                         "Content-Type": "application/json"},
                json=corp, timeout=min(10, ramas))
        except Exception as e:
            eroare = str(e)[:140]
            break
        if r.status_code != 200:
            eroare = f"HTTP {r.status_code} {r.text[:200]}".replace("\n", " ")
            break
        js = r.json()
        locuri += js.get("places", [])
        token = js.get("nextPageToken")
        if not token:
            break
    if eroare and not locuri:
        return None, eroare

    centru = (geo["lat"], geo["lon"]) if geo else None
    vazute, lista = set(), []
    for pl in locuri:
        pid = pl.get("id") or id(pl)
        if pid in vazute:
            continue
        vazute.add(pid)
        loc = pl.get("location") or {}
        lat, lon = loc.get("latitude"), loc.get("longitude")
        dist = None
        if centru and lat is not None and lon is not None:
            dist = round(distanta_km(centru[0], centru[1], lat, lon), 1)
            if dist > RAZA_MAXIMA_KM:
                continue  # în afara orașului
        lista.append({
            "nume": ((pl.get("displayName") or {}).get("text") or "Fără nume").strip(),
            "adresa": (pl.get("formattedAddress") or "").replace(", România", "").strip(),
            "distanta_km": dist,
            "link": pl.get("googleMapsUri"),
            "lat": lat, "lon": lon,
            "rating": pl.get("rating"),
            "nr_recenzii": pl.get("userRatingCount"),
        })
    if centru:
        lista.sort(key=lambda x: (x["distanta_km"] is None, x["distanta_km"] or 0))
    return {"total": len(lista), "fara_nume": 0, "lista": lista[:40],
            "plafonat": len(locuri) >= 20 * PAGINI_GOOGLE and bool(token),
            "sursa": "Google Maps"}, ""


# ----------------------------------------- rezervă: căutare de locuri în Nominatim
TERMENI_NOMINATIM = {
    "cafenea": "cafe", "restaurant": "restaurant", "salon": "hairdresser",
    "patiserie": "bakery", "fitness": "fitness centre", "bar": "bar",
    "stomatologie": "dentist", "farmacie": "pharmacy", "service_auto": "car repair",
}


def gaseste_competitori_nominatim(tip_afacere, cheie, geo, deadline):
    """Rezervă când Overpass nu răspunde. Returnează (dict | None, eroare)."""
    if not geo:
        return None, "zona orașului nu a putut fi determinată"
    time.sleep(1.1)  # politica Nominatim: maximum 1 cerere pe secundă
    ramas = deadline - time.time()
    if ramas < 3:
        return None, "timp insuficient"
    s_, w, n, e = geo["bbox"][0], geo["bbox"][1], geo["bbox"][2], geo["bbox"][3]
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": TERMENI_NOMINATIM.get(cheie, tip_afacere), "format": "jsonv2",
                    "limit": 40, "viewbox": f"{w},{n},{e},{s_}", "bounded": 1,
                    "addressdetails": 1, "accept-language": "ro"},
            headers={"User-Agent": UA}, timeout=min(10, ramas))
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        rezultate = r.json()
    except Exception as ex:
        return None, str(ex)[:80]
    elemente = []
    for x in rezultate:
        try:
            adr = x.get("address") or {}
            elemente.append({"lat": float(x["lat"]), "lon": float(x["lon"]), "tags": {
                "name": x.get("name") or "",
                "addr:street": adr.get("road", ""),
                "addr:housenumber": adr.get("house_number", "")}})
        except (KeyError, TypeError, ValueError):
            continue
    comp = proceseaza_elemente(elemente, (geo["lat"], geo["lon"]))
    comp["plafonat"] = len(rezultate) >= 40
    return comp, ""


def gaseste_competitori_automat(oras, tip_afacere, cheie, geo, deadline):
    """Cache -> Google Maps -> OpenStreetMap -> cache vechi. Returnează (comp, diagnostic)."""
    cheie_cache = (oras.lower().strip(), cheie or tip_afacere.lower().strip())
    in_cache = CACHE_COMPETITORI.get(cheie_cache)
    if in_cache and time.time() - in_cache[0] < TTL_CACHE:
        return in_cache[1], "din memorie (rezultat recent)"

    diag = []
    comp, eroare = gaseste_competitori_google(oras, tip_afacere, cheie, geo, deadline)
    if comp is None:
        diag.append(f"google: {eroare}")
        comp, erori_osm = gaseste_competitori(oras, cheie, geo, min(deadline, time.time() + 16))
        if comp is None:
            diag.append("overpass: " + "; ".join(erori_osm))
            comp, eroare_n = gaseste_competitori_nominatim(tip_afacere, cheie, geo, deadline)
            if comp is None:
                diag.append("nominatim: " + eroare_n)
            else:
                diag.append("lista obținută prin Nominatim")
    if comp is not None:
        CACHE_COMPETITORI[cheie_cache] = (time.time(), comp)
        return comp, " | ".join(diag)
    if in_cache:  # sursele nu răspund, dar avem un rezultat mai vechi
        diag.append("folosit rezultat mai vechi din memorie")
        return in_cache[1], " | ".join(diag)
    return None, " | ".join(diag)


# ---------------------------------------------------------------- evaluare
def evalueaza(buget, buget_min, competitori, populatie, sursa="OpenStreetMap"):
    """Nivel de risc calculat de cod (determinist). AI-ul îl explică."""
    scor = 0
    raport_buget = None
    if buget_min:
        raport_buget = buget / buget_min
        if raport_buget >= 1.5:
            scor += 2
        elif raport_buget >= 1:
            scor += 1
        elif raport_buget >= 0.7:
            scor -= 1
        else:
            scor -= 3

    densitate = None
    if competitori is not None:
        if populatie and populatie > 0:
            densitate = competitori / (populatie / 10000.0)
            if densitate <= 3:
                sc = 2
            elif densitate <= 6:
                sc = 1
            elif densitate <= 10:
                sc = -1
            else:
                sc = -2
        else:
            if competitori <= 10:
                sc = 2
            elif competitori <= 30:
                sc = 1
            elif competitori <= 80:
                sc = -1
            else:
                sc = -2
        # OpenStreetMap e adesea incomplet: puțini competitori nu înseamnă piață liberă
        if sursa != "Google Maps":
            sc = min(sc, 1)
        scor += sc

    if scor >= 3:
        nivel, titlu = "bun", "FEZABIL – ȘANSE BUNE"
    elif scor >= 1:
        nivel, titlu = "moderat", "FEZABIL, DAR CU RISCURI MODERATE"
    elif scor >= -1:
        nivel, titlu = "ridicat", "RISC RIDICAT"
    else:
        nivel, titlu = "foarte_ridicat", "RISC FOARTE RIDICAT"
    return {"scor": scor, "nivel": nivel, "titlu": titlu,
            "densitate": densitate, "raport_buget": raport_buget}


def calculeaza_alocare(alocare_ai, buget):
    """Normalizează procentele la 100% și calculează sumele în EUR."""
    randuri = []
    if isinstance(alocare_ai, list):
        for x in alocare_ai:
            try:
                randuri.append((str(x["categorie"]), float(x["procent"]),
                                str(x.get("detalii", ""))))
            except (KeyError, TypeError, ValueError):
                randuri = []
                break
    if not randuri or sum(r[1] for r in randuri) <= 0:
        randuri = [(c, float(p), d) for c, p, d in ALOCARE_IMPLICITA]
    total = sum(r[1] for r in randuri)
    return [{"categorie": c, "procent": round(p * 100 / total, 1),
             "suma": int(round(buget * p / total)), "detalii": d}
            for c, p, d in randuri]


# ------------------------------------------------------------------- raport AI
def lista_text(items):
    return [str(i).strip() for i in items if str(i).strip()] if isinstance(items, list) else []


def valideaza_raport(r):
    if not isinstance(r, dict):
        return None
    out = {
        "rezumat": str(r.get("rezumat", "")).strip(),
        "concurenta": str(r.get("concurenta", "")).strip(),
        "analiza_buget": str(r.get("analiza_buget", "")).strip(),
        "alocare_buget": r.get("alocare_buget"),
        "riscuri": lista_text(r.get("riscuri")),
        "oportunitati": lista_text(r.get("oportunitati")),
        "plan_actiune": lista_text(r.get("plan_actiune")),
        "concluzie": str(r.get("concluzie", "")).strip(),
    }
    if not out["rezumat"] or not out["concurenta"] or not out["plan_actiune"]:
        return None
    return out


def parseaza_json(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


SCHEMA_JSON = """{
  "rezumat": "2-3 propoziții: concluzia generală, nuanțată, fără promisiuni",
  "concurenta": "1-2 paragrafe: cât de aglomerată e piața, ce înseamnă pentru un nou venit, cum te-ai putea diferenția; menționează pe nume 3-5 competitori din listă, DOAR ce știi din date (nume, adresă, distanță), fără a inventa detalii despre ei",
  "analiza_buget": "1-2 paragrafe: dacă bugetul acoperă investiția inițială ȘI primele luni de funcționare (chirie, utilități, salarii, stoc), ce sacrificii ar trebui făcute",
  "alocare_buget": [{"categorie": "...", "procent": 0, "detalii": "..."}],
  "riscuri": ["4-6 riscuri concrete pentru acest tip de afacere în acest oraș"],
  "oportunitati": ["3-5 oportunități sau idei de diferențiere"],
  "plan_actiune": ["6-8 pași concreți, în ordine cronologică, de la validare până la lansare și primele 6 luni"],
  "concluzie": "un paragraf echilibrat care spune clar că succesul nu poate fi garantat și de ce anume depinde"
}"""


def construieste_prompt(oras, tip, buget, buget_min, sursa, comp, ev):
    if comp is None:
        linie_comp = "Numărul de competitori NU a putut fi determinat (nu inventa o cifră)."
        exemple = "(listă indisponibilă)"
    else:
        linie_comp = (f"Pe {comp['sursa']} au fost găsite {comp['total']} locații similare"
                      f"{' (limită atinsă, numărul real poate fi mai mare)' if comp['plafonat'] else ''}"
                      f"{', dintre care ' + str(comp['fara_nume']) + ' fără nume înregistrat' if comp['fara_nume'] else ''}.")
        if comp["sursa"] != "Google Maps":
            linie_comp += (" Datele OpenStreetMap sunt adesea incomplete în orașele mici, "
                           "deci numărul real e probabil mai mare.")
        if ev["densitate"] is not None:
            linie_comp += f" Densitate: aprox. {ev['densitate']:.1f} la 10.000 locuitori."
        rand = []
        for c in comp["lista"][:12]:
            parti = [c["nume"]]
            if c["adresa"]:
                parti.append(c["adresa"])
            if c["distanta_km"] is not None:
                parti.append(f"~{c['distanta_km']} km de centru")
            if c.get("rating"):
                parti.append(f"rating {c['rating']} ({c.get('nr_recenzii') or 0} recenzii)")
            rand.append("- " + ", ".join(parti))
        exemple = "\n".join(rand) or "(niciun competitor cu nume)"
    linie_min = (f"Buget minim orientativ pentru acest tip de afacere: {buget_min} EUR "
                 f"(sursa: {sursa}). Raport buget/minim: {ev['raport_buget']:.2f}."
                 if buget_min else
                 "Nu există un buget minim de referință pentru acest tip; estimează prudent.")
    return f"""Ești un consultant de afaceri din România, onest și realist. Evaluezi un proiect de antreprenoriat.

DATE:
- Oraș: {oras}
- Tip afacere: {tip}
- Buget disponibil: {buget:.0f} EUR
- {linie_min}
- Concurență: {linie_comp}
- Competitori găsiți (cei mai apropiați de centru):
{exemple}
- Nivel de risc calculat de sistem: {ev['titlu']} (nu îl schimba).

REGULI:
- Scrie în română, ton profesionist, concret și realist. Nu promite succes și nu spune „garantat".
- Un buget suficient pentru deschidere NU garantează succesul; explică ce mai contează (locație, calitate, marketing, fluxul de numerar, experiența proprietarului).
- Nu inventa statistici exacte, prețuri sau date despre competitori; folosește formulări ca „estimativ" sau „de obicei".
- alocare_buget: 6-8 categorii, procentele însumează 100, adaptate tipului de afacere și bugetului.
- Răspunde STRICT cu un obiect JSON valid, fără text în afara lui, cu exact această structură:
{SCHEMA_JSON}"""


def apel_gemini(model, prompt, timeout):
    """Returnează (text, eroare)."""
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0.5, "maxOutputTokens": 8000,
                                       "responseMimeType": "application/json"}},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None, f"HTTP {r.status_code} {r.text[:140]}".replace("\n", " ")
        try:
            parti = r.json()["candidates"][0]["content"]["parts"]
        except Exception:
            return None, "răspuns gol sau blocat"
        text = "".join(p.get("text", "") for p in parti if not p.get("thought")).strip()
        return (text, None) if text else (None, "text gol")
    except Exception as e:
        return None, str(e)[:140]


def modele_descoperite():
    """Modele flash disponibile pentru cheia ta (dacă numele fixe au fost retrase)."""
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         headers={"x-goog-api-key": GEMINI_API_KEY},
                         params={"pageSize": 200}, timeout=8)
        r.raise_for_status()
        out = []
        for m in r.json().get("models", []):
            nume = m.get("name", "").replace("models/", "")
            if ("generateContent" in m.get("supportedGenerationMethods", [])
                    and "flash" in nume
                    and not any(x in nume for x in ("image", "tts", "live", "audio",
                                                     "embedding", "robotics", "computer"))):
                out.append(nume)
        return sorted(out, reverse=True)
    except Exception as e:
        print(f"Descoperire modele: {e}")
        return []


def genereaza_raport(prompt, deadline):
    """Returnează (raport_validat | None, diagnostic)."""
    erori = []
    if GEMINI_API_KEY:
        incercate = []
        candidati = list(MODELE_GEMINI)
        descoperite_folosite = False
        while candidati:
            model = candidati.pop(0)
            if model in incercate:
                continue
            incercate.append(model)
            rezerva = REZERVA_GROQ if client is not None else 0   # doar cât să apuce Groq
            ramas = deadline - time.time() - rezerva
            if ramas < (6 if not incercate[:-1] else 12):
                erori.append("timp insuficient pentru Gemini")
                break
            text, eroare = apel_gemini(model, prompt, ramas)   # tot timpul rămas
            if text:
                raport = valideaza_raport(parseaza_json(text))
                if raport:
                    return raport, ""
                eroare = "JSON invalid"
            erori.append(f"{model}: {eroare}")
            if eroare and ("HTTP 403" in eroare or "HTTP 401" in eroare or "API_KEY_INVALID" in eroare):
                erori.append("cheia/proiectul Gemini nu are acces; celelalte modele nu se mai încearcă")
                break
            if not candidati and not descoperite_folosite:
                descoperite_folosite = True
                candidati = [m for m in modele_descoperite() if m not in incercate][:2]
    if client is not None and deadline - time.time() > 4:
        try:
            c = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.5, max_tokens=3000,
                response_format={"type": "json_object"},
                timeout=max(5, min(30, deadline - time.time())))
            raport = valideaza_raport(parseaza_json(c.choices[0].message.content))
            if raport:
                return raport, ""
            erori.append("groq: JSON invalid")
        except Exception as e:
            erori.append(f"groq: {str(e)[:100]}")
    return None, " | ".join(erori) or "nicio cheie AI configurată"


def raport_rezerva(ev, oras, tip, buget, buget_min, comp):
    """Raport de bază, folosit doar dacă AI-ul nu răspunde."""
    if comp is None:
        conc = ("Numărul de competitori nu a putut fi determinat automat acum. "
                f"Verificați manual pe Google Maps câte locații de tip „{tip}” există în {oras}, "
                "vizitați-le și notați prețurile, programul și gradul de aglomerare.")
    else:
        conc = f"Pe harta ({comp['sursa']}) apar {comp['total']} locații similare în {oras}."
        if ev["densitate"] is not None:
            conc += f" Asta înseamnă aproximativ {ev['densitate']:.1f} la 10.000 de locuitori."
        nume = [c["nume"] for c in comp["lista"][:5]]
        if nume:
            conc += " Printre cei mai apropiați de centru: " + ", ".join(nume) + "."
    if buget_min:
        if buget >= buget_min:
            ab = (f"Bugetul de {buget:.0f} EUR depășește minimul orientativ de {buget_min} EUR, "
                  "deci poate acoperi investiția inițială. Păstrați însă o rezervă pentru "
                  "primele luni, când încasările sunt de obicei sub cheltuieli.")
        else:
            ab = (f"Bugetul de {buget:.0f} EUR este sub minimul orientativ de {buget_min} EUR. "
                  "Ar trebui un spațiu mai mic, echipamente second-hand sau lansare în etape.")
    else:
        ab = ("Nu avem un buget de referință pentru acest tip de afacere. "
              "Cereți oferte reale pentru spațiu, echipamente și autorizări.")
    return {
        "rezumat": (f"Pentru {tip} în {oras} cu {buget:.0f} EUR, nivelul estimat este: "
                    f"{ev['titlu'].lower()}. Rezultatul este orientativ, nu o garanție."),
        "concurenta": conc,
        "analiza_buget": ab,
        "alocare_buget": None,
        "riscuri": ["Încasări sub așteptări în primele luni",
                    "Chirie și utilități mari față de venituri",
                    "Concurență deja stabilită în zonă",
                    "Dependența de o singură locație și de proprietar"],
        "oportunitati": ["Diferențiere prin calitate, servicii sau specializare",
                         "Comunitate online locală și recenzii bune",
                         "Parteneriate cu afaceri din zonă"],
        "plan_actiune": ["Vizitați și analizați competitorii din zonă",
                         "Alegeți locația după trafic pedestru și chirie",
                         "Cereți oferte reale pentru echipamente și amenajare",
                         "Faceți un plan financiar cu 3 scenarii (optimist, realist, pesimist)",
                         "Obțineți avizele și înființați firma",
                         "Pregătiți lansarea (rețele sociale, promoții)",
                         "Urmăriți lunar încasările și ajustați oferta"],
        "concluzie": ("Un buget suficient permite deschiderea, dar nu garantează succesul. "
                      "Depinde mult de locație, execuție, constanță și de felul în care vă "
                      "diferențiați de ceilalți."),
    }


# ---------------------------------------------------------------------- rute
def cod_utilizator(punctaje):
    """Primele 3 categorii ale utilizatorului, ca literă (ex. 'SAI')."""
    ordonat = sorted(ORDINE, key=lambda l: (-punctaje[NUME_LITERA[l]], ORDINE.index(l)))
    return "".join(l for l in ordonat[:3] if punctaje[NUME_LITERA[l]] > 0)


def vector_cariera(cod):
    v = [0, 0, 0, 0, 0, 0]
    for poz, l in enumerate(cod[:3]):
        v[ORDINE.index(l)] = 3 - poz  # prima literă contează cel mai mult
    return v


def corelatie(a, b):
    """Corelația Pearson dintre profilul utilizatorului și cel al meseriei (-1..1)."""
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0


def clasament_cariere(df, punctaje, limita=8):
    u = [punctaje[NUME_LITERA[l]] for l in ORDINE]
    cod_u = cod_utilizator(punctaje)
    rez = []
    for _, rand in df.iterrows():
        cod = rand["cod_holland"]
        potrivire = max(0.0, corelatie(u, vector_cariera(cod))) * 100
        comun = [NUME_AFISAT[l] for l in cod if l in cod_u]
        forta = sum((3 - poz) * punctaje[NUME_LITERA[l]] for poz, l in enumerate(cod[:3]))
        rez.append((potrivire, forta, rand, comun))
    rez.sort(key=lambda x: (-round(x[0], 6), -x[1], int(x[2]["id_cariera"])))
    rez = [(p, r, c) for p, _, r, c in rez]
    return rez[:limita]


def text_curat(rand, camp):
    v = rand.get(camp, "")
    return "" if pd.isna(v) else str(v)


def cariera_json(potrivire, rand, comun):
    sal = rand.get("salariu_mediu")
    return {
        'id_cariera': int(rand['id_cariera']),
        'titlu_job': text_curat(rand, 'titlu_job'),
        'cod_holland': rand['cod_holland'],
        'potrivire': int(round(potrivire)),
        'potrivit_pe': comun,
        'facultate_ro': text_curat(rand, 'facultate_ro'),
        'admitere_ro': text_curat(rand, 'admitere_ro'),
        'facultate_strainatate': text_curat(rand, 'facultate_strainatate'),
        'admitere_strainatate': text_curat(rand, 'admitere_strainatate'),
        'pas_practic': text_curat(rand, 'pas_practic'),
        'salariu_mediu': None if pd.isna(sal) else int(sal),
    }


@app.route('/predict-cariera', methods=['POST'])
def predict_cariera():
    if df_cariere is None:
        return jsonify({'error': 'Baza de date cu cariere nu este disponibilă.'}), 500

    punctaje = calculeaza_punctaje(citeste_date())
    scor_maxim = max(punctaje.values())
    if scor_maxim == 0:
        return jsonify({'error': 'Nu s-au primit răspunsuri valide din test.'}), 400

    if len(set(punctaje.values())) == 1:
        return jsonify({'error': 'Ai dat același scor tuturor categoriilor, deci profilul nu se poate '
                                 'stabili. Reia testul și alege mai selectiv.'}), 400

    try:
        cod_u = cod_utilizator(punctaje)
        clasament = clasament_cariere(df_cariere, punctaje)
        top = [cariera_json(p, r, c) for p, r, c in clasament]
    except Exception as e:
        print(f"Eroare clasament cariere: {e}")
        return jsonify({'error': f'Eroare la calcularea potrivirilor: {e}'}), 500

    profil = max(punctaje, key=punctaje.get)
    total = sum(punctaje.values()) or 1
    litere_profil = [NUME_AFISAT[l] for l in cod_u]
    descriere = (f"Profilul tău Holland este {cod_u} ({', '.join(litere_profil)}). "
                 f"{DESCRIERI[cod_u[0]]}")
    if len(cod_u) > 1:
        descriere += (f" Pe locul doi, {litere_profil[1].lower()}: "
                      f"{DESCRIERI[cod_u[1]][0].lower() + DESCRIERI[cod_u[1]][1:]}")

    rezultat = {
        'status': 'success',
        'cod_holland': cod_u,
        'profil_detectat': profil,
        'profil_complet': [NUME_LITERA[l] for l in cod_u],
        'punctaje': punctaje,
        'punctaje_procente': {k: round(v * 100 / total) for k, v in punctaje.items()},
        'descriere_profil': descriere,
        'punctaj_obtinut': scor_maxim,
        'cariere_in_baza': int(len(df_cariere)),
        'top_cariere': top,
        'avertisment': ("Testul orientează, nu hotărăște: folosește rezultatele ca punct de plecare "
                        "și verifică ofertele concrete ale facultăților și angajatorilor."),
    }
    # câmpurile vechi (cariera cea mai potrivită), pentru compatibilitate cu pagina existentă
    rezultat.update(top[0])
    rezultat['cod_holland_cariera'] = rezultat['cod_holland']
    rezultat['cod_holland'] = cod_u
    return jsonify(rezultat)


@app.route('/')
def health():
    return jsonify({
        'status': 'ok',
        'cariere_incarcate': df_cariere is not None,
        'numar_cariere': int(len(df_cariere)) if df_cariere is not None else 0,
        'gemini_configurat': GEMINI_API_KEY is not None,
        'google_maps_configurat': GOOGLE_API_KEY is not None,
        'groq_configurat': client is not None,
    })


@app.route('/test-ai')
def test_ai():
    """Deschide /test-ai în browser ca să vezi de ce nu răspunde Gemini."""
    if not GEMINI_API_KEY:
        return jsonify({'error': 'GEMINI_API_KEY nu este setată.'}), 400
    rez = {}
    for model in MODELE_GEMINI:
        text, eroare = apel_gemini(model, 'Răspunde cu JSON: {"ok": true}', 15)
        rez[model] = "OK" if text else eroare
    rez["_modele_disponibile_pentru_cheia_ta"] = modele_descoperite()
    # ultimele 4 caractere, ca să poți verifica în AI Studio că rulează cheia nouă
    rez["_cheia_gemini_se_termina_in"] = GEMINI_API_KEY[-4:]
    rez["_groq_configurat"] = client is not None
    return jsonify(rez)


@app.route('/test-harta')
def test_harta():
    """Ex: /test-harta?oras=Bacău&tip=cafenea"""
    oras = request.args.get('oras', 'Bacău')
    tip = request.args.get('tip', 'cafenea')
    cheie = detecteaza_tip(tip)
    geo = geocodeaza_oras(oras)
    comp, diag = gaseste_competitori_automat(oras, tip, cheie, geo, time.time() + 25)
    return jsonify({'tip_detectat': cheie, 'geocodare': geo, 'diagnostic': diag,
                    'competitori': comp and {'sursa': comp['sursa'], 'total': comp['total'],
                                             'plafonat': comp['plafonat'],
                                             'primii_5': comp['lista'][:5]}})


@app.route('/predict', methods=['POST'])
def analizeaza_business_live():
    start = time.time()
    data = citeste_date()
    try:
        buget = float(data.get('buget', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Bugetul trebuie să fie un număr.'}), 400
    if buget <= 0:
        return jsonify({'error': 'Bugetul trebuie să fie mai mare ca zero.'}), 400

    oras = str(data.get('oras', 'Bacău')).strip() or 'Bacău'
    tip_afacere = str(data.get('tip_afacere', 'Cafenea')).strip() or 'Cafenea'

    cheie = detecteaza_tip(tip_afacere)
    buget_min, nume_oficial, sursa = buget_minim_pentru(cheie)

    geo = geocodeaza_oras(oras)
    comp, diag_harta = gaseste_competitori_automat(oras, tip_afacere, cheie, geo, start + 28)
    total = comp["total"] if comp else None
    populatie = geo["populatie"] if geo else None
    ev = evalueaza(buget, buget_min, total, populatie,
                   comp["sursa"] if comp else "OpenStreetMap")

    raport, diag = None, ""
    if GEMINI_API_KEY or client is not None:
        prompt = construieste_prompt(oras, tip_afacere, buget, buget_min, sursa, comp, ev)
        raport, diag = genereaza_raport(prompt, start + TIMP_TOTAL)
    ai_ok = raport is not None
    if not ai_ok:
        raport = raport_rezerva(ev, oras, tip_afacere, buget, buget_min, comp)

    alocare = calculeaza_alocare(raport.get("alocare_buget"), buget)
    raport.pop("alocare_buget", None)

    return jsonify({
        'rezultat': ev['titlu'],
        'nivel': ev['nivel'],
        'scor': ev['scor'],
        'oras': oras,
        'tip_detectat': nume_oficial or tip_afacere,
        'buget': buget,
        'buget_minim': buget_min,
        'raport_buget': round(ev['raport_buget'], 2) if ev['raport_buget'] else None,
        'centru': {'lat': geo['lat'], 'lon': geo['lon']} if geo else None,
        'sursa_competitori': comp['sursa'] if comp else None,
        'harta_diagnostic': diag_harta,
        'competitori_gasiti': total,
        'competitori_lista': comp["lista"] if comp else [],
        'competitori_fara_nume': comp["fara_nume"] if comp else 0,
        'competitori_plafonat': bool(comp and comp["plafonat"]),
        'densitate_la_10k': round(ev['densitate'], 1) if ev['densitate'] is not None else None,
        'populatie': populatie,
        'raport': raport,
        'alocare_buget': alocare,
        'raport_ai': ai_ok,
        'ai_diagnostic': "" if ai_ok else diag,
        'disclaimer': DISCLAIMER,
    })


# Doar pentru rulare locală
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
