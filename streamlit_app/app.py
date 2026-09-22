import streamlit as st
import requests
import PyPDF2
import os

st.set_page_config(page_title="Job Matching Engine", page_icon="🎯")

st.title("🎯 Job Matching Engine")
st.write("Lade deinen Lebenslauf hoch oder füge den Text ein, um passende Stellenanzeigen zu finden.")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/match")


def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# Two input methods: file upload or paste text
tab1, tab2 = st.tabs(["📄 PDF hochladen", "✏️ Text einfügen"])

cv_text = ""

with tab1:
    uploaded_file = st.file_uploader("Lebenslauf als PDF", type=["pdf"])
    if uploaded_file is not None:
        cv_text = extract_text_from_pdf(uploaded_file)
        with st.expander("Extrahierter Text (zur Kontrolle)"):
            st.text(cv_text[:1000] + "..." if len(cv_text) > 1000 else cv_text)

with tab2:
    pasted_text = st.text_area("Lebenslauf (Text)", height=250, placeholder="Füge hier deinen Lebenslauf ein...")
    if pasted_text.strip():
        cv_text = pasted_text

top_n = st.slider("Anzahl der Ergebnisse", min_value=5, max_value=20, value=10)
with_explanation = st.checkbox("Erklärungen generieren (dauert länger)", value=False)

if st.button("Passende Stellen finden", type="primary"):
    if not cv_text.strip():
        st.warning("Bitte lade einen Lebenslauf hoch oder füge den Text ein.")
    else:
        with st.spinner("Suche läuft..."):
            try:
                response = requests.post(API_URL, json={
                    "cv_text": cv_text,
                    "top_n": top_n,
                    "with_explanation": with_explanation
                })
                response.raise_for_status()
                results = response.json()

                st.success(f"{len(results)} passende Stellen gefunden")

                for r in results:
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(r["job_title"])
                            st.write(f"**{r['company']}** — {r['job_location']}")
                        with col2:
                            st.metric("Score", f"{r['score']:.2f}")

                        if r.get("explanation"):
                            with st.expander("💡 Warum passt das?"):
                                st.markdown(r["explanation"])

            except requests.exceptions.ConnectionError:
                st.error("Kann die API nicht erreichen. Läuft der FastAPI-Server?")
            except Exception as e:
                st.error(f"Fehler: {e}")

