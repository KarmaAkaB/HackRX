# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from .parser import parse_query
from .retrieval import SemanticRetriever
from .matching import ClauseMatcher
from .evaluation import DecisionEngine

app = FastAPI(title="Intelligent Query - Retrieval API", version="1.0")

class RunRequest(BaseModel):
    documents: str
    questions: list[str]

class RunResponse(BaseModel):
    answers: list[str]

retriever = SemanticRetriever()
matcher = ClauseMatcher()
engine = DecisionEngine()

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run(req: RunRequest):
    answers = []
    for q in req.questions:
        parsed = parse_query(q)
        clauses = retriever.retrieve(parsed)
        matched = matcher.match(parsed, clauses)
        human_answer = engine.evaluate(parsed, matched)
        answers.append(human_answer)
    return RunResponse(answers=answers)
