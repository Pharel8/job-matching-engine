import logging
import os
import secrets

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

logger = logging.getLogger("job-matching-api")

app = FastAPI(title="Job Matching Engine API")

# Large data files live on the Hugging Face Hub and are downloaded at startup if missing
HF_BASE_URL = "https://huggingface.co/datasets/pharelMl/job-matching-engine-data/resolve/main"


def download_if_missing(local_path, filename):
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        response = requests.get(f"{HF_BASE_URL}/{filename}", timeout=300)
        response.raise_for_status()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")


download_if_missing("data/combined_with_scores.pkl", "combined_with_scores.pkl")
download_if_missing("data/job_embeddings.npy", "job_embeddings.npy")

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

combined = pd.read_pickle("data/combined_with_scores.pkl")
job_embeddings = np.load("data/job_embeddings.npy")


def check_key(x_api_key: str = Header(default="3kKSEztArgKo9c5mARKga53NMqks1ZFKocGbefxsWUs")):
    """Optional shared secret: only enforced when APP_API_KEY is set in the environment."""
    expected = os.getenv("APP_API_KEY")
    if expected and not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


class MatchRequest(BaseModel):
    cv_text: str = Field(max_length=20000)
    top_n: int = Field(default=10, ge=1, le=20)
    with_explanation: bool = False  # LLM explanations are slower and cost money: opt-in


class MatchResult(BaseModel):
    job_title: str
    company: str
    job_location: str
    score: float
    explanation: str | None = None


def generate_match_explanation(cv_text, job_title, company, job_summary):
    prompt = f"""You are an assistant that explains why a CV matches a job posting.

IMPORTANT:
- Only mention skills and experience that are LITERALLY present in the CV text below. Do not invent anything.
- Always respond in the SAME LANGUAGE as the CV text below, regardless of the language of the job posting.

CV:
{cv_text}

JOB POSTING (may be in a different language than the CV):
Title: {job_title}
Company: {company}
Description: {job_summary}

Respond in the CV's language, in exactly this format:
MATCHING SKILLS: [list 2-4 concrete overlaps between CV and posting]
MISSING SKILLS: [list 1-3 requirements from the posting NOT found in the CV]
BRIEF ASSESSMENT: [1-2 sentences on whether this match makes sense]
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content


@app.post("/match", response_model=list[MatchResult], dependencies=[Depends(check_key)])
def match_cv(request: MatchRequest):
    cv_embedding = model.encode([request.cv_text])
    scores = cosine_similarity(cv_embedding, job_embeddings).flatten()

    # Indices of the best matches, highest score first (no DataFrame copy needed)
    top_idx = np.argsort(scores)[::-1][: request.top_n]

    results = []
    for idx in top_idx:
        row = combined.iloc[idx]
        explanation = None
        if request.with_explanation:
            try:
                explanation = generate_match_explanation(
                    request.cv_text, row["job_title"], row["company"], row["job_summary_clean"]
                )
            except Exception as e:
                # e.g. OpenAI credits exhausted: still return the scores
                logger.warning("Explanation failed: %s", e)
        results.append(MatchResult(
            job_title=row["job_title"],
            company=row["company"],
            job_location=row["job_location"],
            score=round(float(scores[idx]), 3),
            explanation=explanation,
        ))

    return results


@app.get("/")
def root():
    return {"status": "Job Matching Engine API is running"}