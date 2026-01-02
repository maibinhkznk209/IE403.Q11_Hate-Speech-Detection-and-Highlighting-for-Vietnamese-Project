from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any

LabelName = Literal["HATE", "OFFENSIVE", "NEUTRAL"]

class AnalyzeOptions(BaseModel):
    return_spans: bool = True
    threshold: float = 0.5
    highlight_policy: Literal["lexicon", "model+lexicon"] = "lexicon"

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    lang: str = "vi"
    options: AnalyzeOptions = AnalyzeOptions()

class HighlightSpan(BaseModel):
    start: int
    end: int
    text: str
    type: Literal["negative", "positive", "neutral"] = "negative"
    source: Literal["lexicon", "model_rationale"] = "lexicon"
    confidence: float = 0.99

class LabelScore(BaseModel):
    name: LabelName
    score: float

class AnalyzeResponse(BaseModel):
    request_id: str
    label: LabelName
    score: float
    labels: List[LabelScore]
    highlights: List[HighlightSpan] = []
    meta: Dict[str, Any] = {}

class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(min_length=1, max_length=200)
    lang: str = "vi"
    options: AnalyzeOptions = AnalyzeOptions()

class BatchAnalyzeResponse(BaseModel):
    request_id: str
    results: List[AnalyzeResponse]
    meta: Dict[str, Any] = {}
