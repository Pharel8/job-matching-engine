import os
import requests
import PyPDF2
import streamlit as st

st.set_page_config(page_title="Job Matching Engine", page_icon="🎯")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/match")
MAX_CV_CHARS = 20000  # keep in sync with the limit in src/api/main.py
SUPPORTED = ["de", "en", "fr"]

T = {
    "de": {
        "title": "🎯 Job Matching Engine",
        "intro": "Lade deinen Lebenslauf hoch oder füge den Text ein, um passende Stellenanzeigen zu finden.",
        "tab_pdf": "📄 PDF hochladen",
        "tab_text": "✏️ Text einfügen",
        "uploader": "Lebenslauf als PDF",
        "extracted": "Extrahierter Text (zur Kontrolle)",
        "textarea": "Lebenslauf (Text)",
        "placeholder": "Füge hier deinen Lebenslauf ein...",
        "slider": "Anzahl der Ergebnisse",
        "checkbox": "Erklärungen generieren (dauert länger)",
        "button": "Passende Stellen finden",
        "empty": "Bitte lade einen Lebenslauf hoch oder füge den Text ein.",
        "too_long": "Der Lebenslauf ist zu lang (maximal {max} Zeichen).",
        "spinner": "Suche läuft... (beim ersten Aufruf kann der Start einige Minuten dauern)",
        "found": "{n} passende Stellen gefunden",
        "why": "💡 Warum passt das?",
        "no_expl": "Erklärung momentan nicht verfügbar.",
        "score": "Score",
        "conn_error": "Kann die API nicht erreichen.",
        "error": "Fehler: {e}",
    },
    "en": {
        "title": "🎯 Job Matching Engine",
        "intro": "Upload your CV or paste its text to find matching job postings.",
        "tab_pdf": "📄 Upload PDF",
        "tab_text": "✏️ Paste text",
        "uploader": "CV as PDF",
        "extracted": "Extracted text (for checking)",
        "textarea": "CV (text)",
        "placeholder": "Paste your CV here...",
        "slider": "Number of results",
        "checkbox": "Generate explanations (slower)",
        "button": "Find matching jobs",
        "empty": "Please upload a CV or paste its text.",
        "too_long": "The CV is too long (maximum {max} characters).",
        "spinner": "Searching... (the first request may take a few minutes to start up)",
        "found": "{n} matching jobs found",
        "why": "💡 Why is this a match?",
        "no_expl": "Explanation currently unavailable.",
        "score": "Score",
        "conn_error": "Cannot reach the API.",
        "error": "Error: {e}",
    },
    "fr": {
        "title": "🎯 Job Matching Engine",
        "intro": "Téléverse ton CV ou colle son texte pour trouver des offres adaptées.",
        "tab_pdf": "📄 Téléverser un PDF",
        "tab_text": "✏️ Coller le texte",
        "uploader": "CV au format PDF",
        "extracted": "Texte extrait (pour vérification)",
        "textarea": "CV (texte)",
        "placeholder": "Colle ton CV ici...",
        "slider": "Nombre de résultats",
        "checkbox": "Générer des explications (plus lent)",
        "button": "Trouver des offres",
        "empty": "Merci de téléverser un CV ou de coller son texte.",
        "too_long": "Le CV est trop long (maximum {max} caractères).",
        "spinner": "Recherche en cours... (la première requête peut prendre quelques minutes)",
        "found": "{n} offres trouvées",
        "why": "💡 Pourquoi cette offre ?",
        "no_expl": "Explication momentanément indisponible.",
        "score": "Score",
        "conn_error": "Impossible de joindre l'API.",
        "error": "Erreur : {e}",
    },
}


def detect_language():
    # Browser language, e.g. "fr-FR" -> "fr"; fall back to English if unsupported/unavailable
    locale = getattr(st.context, "locale", None) or ""
    code = locale.split("-")[0].lower()
    return code if code in SUPPORTED else "en"


def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# Language: browser default on first load, then the user's own choice
if "lang" not in st.session_state:
    st.session_state.lang = detect_language()

title_col, lang_col = st.columns([4, 2])
with lang_col:
    st.radio("Language", SUPPORTED, key="lang", horizontal=True,
             label_visibility="collapsed", format_func=str.upper)

t = T[st.session_state.lang]

with title_col:
    st.title(t["title"])
st.write(t["intro"])

tab1, tab2 = st.tabs([t["tab_pdf"], t["tab_text"]])
cv_text = ""

with tab1:
    uploaded_file = st.file_uploader(t["uploader"], type=["pdf"])
    if uploaded_file is not None:
        cv_text = extract_text_from_pdf(uploaded_file)
        with st.expander(t["extracted"]):
            st.text(cv_text[:1000] + "..." if len(cv_text) > 1000 else cv_text)

with tab2:
    pasted_text = st.text_area(t["textarea"], height=250, placeholder=t["placeholder"])
    if pasted_text.strip():
        cv_text = pasted_text

top_n = st.slider(t["slider"], min_value=5, max_value=20, value=10)
with_explanation = st.checkbox(t["checkbox"], value=False)

# Results survive Streamlit reruns (expander clicks, slider moves) via session_state
if "results" not in st.session_state:
    st.session_state.results = None
if "expl_requested" not in st.session_state:
    st.session_state.expl_requested = False

if st.button(t["button"], type="primary"):
    if not cv_text.strip():
        st.warning(t["empty"])
    elif len(cv_text) > MAX_CV_CHARS:
        st.warning(t["too_long"].format(max=MAX_CV_CHARS))
    else:
        with st.spinner(t["spinner"]):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "cv_text": cv_text,
                        "top_n": top_n,
                        "with_explanation": with_explanation,
                    },
                    headers={"X-API-Key": os.getenv("APP_API_KEY", "3kKSEztArgKo9c5mARKga53NMqks1ZFKocGbefxsWUs")},
                    timeout=600,
                )
                response.raise_for_status()
                st.session_state.results = response.json()
                st.session_state.expl_requested = with_explanation
            except requests.exceptions.ConnectionError:
                st.error(t["conn_error"])
            except Exception as e:
                st.error(t["error"].format(e=e))

if st.session_state.results:
    results = st.session_state.results
    st.success(t["found"].format(n=len(results)))
    for r in results:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(r["job_title"])
                st.write(f"**{r['company']}** — {r['job_location']}")
            with col2:
                st.metric(t["score"], f"{r['score']:.2f}")
            if r.get("explanation"):
                with st.expander(t["why"]):
                    st.markdown(r["explanation"])
            elif st.session_state.expl_requested:
                st.caption(t["no_expl"])