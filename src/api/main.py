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

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

combined = pd.read_pickle('data/combined_with_scores.pkl')
job_embeddings = np.load('data/job_embeddings.npy')


class MatchRequest(BaseModel):
    cv_text: str
    top_n: int = 10
    with_explanation: bool = False  # RAG explanations are slower/cost money, opt-in


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
        max_tokens=300
    )
    return response.choices[0].message.content


@app.post("/match", response_model=list[MatchResult])
def match_cv(request: MatchRequest):
    cv_embedding = model.encode([request.cv_text])
    scores = cosine_similarity(cv_embedding, job_embeddings).flatten()

    results_df = combined.copy()
    results_df['score'] = scores
    top_matches = results_df.sort_values('score', ascending=False).head(request.top_n)

    results = []
    for _, row in top_matches.iterrows():
        explanation = None
        if request.with_explanation:
            explanation = generate_match_explanation(
                request.cv_text, row['job_title'], row['company'], row['job_summary_clean']
            )
        results.append(MatchResult(
            job_title=row['job_title'],
            company=row['company'],
            job_location=row['job_location'],
            score=round(row['score'], 3),
            explanation=explanation
        ))

    return results


@app.get("/")
def root():
    return {"status": "Job Matching Engine API is running"}