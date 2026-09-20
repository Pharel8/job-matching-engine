from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Job Matching Engine API")

# Load model, data, and pre-computed embeddings once at startup
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
# TODO: will be used for RAG explanations (see notebook section 13) — not yet wired into /match
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

combined = pd.read_pickle('data/combined_with_scores.pkl')
job_embeddings = np.load('data/job_embeddings.npy')  # pre-computed, instant load


class MatchRequest(BaseModel):
    cv_text: str
    top_n: int = 10


class MatchResult(BaseModel):
    job_title: str
    company: str
    job_location: str
    score: float


@app.post("/match", response_model=list[MatchResult])
def match_cv(request: MatchRequest):
    cv_embedding = model.encode([request.cv_text])
    scores = cosine_similarity(cv_embedding, job_embeddings).flatten()

    results_df = combined.copy()
    results_df['score'] = scores
    top_matches = results_df.sort_values('score', ascending=False).head(request.top_n)

    return [
        MatchResult(
            job_title=row['job_title'],
            company=row['company'],
            job_location=row['job_location'],
            score=round(row['score'], 3)
        )
        for _, row in top_matches.iterrows()
    ]


@app.get("/")
def root():
    return {"status": "Job Matching Engine API is running"}