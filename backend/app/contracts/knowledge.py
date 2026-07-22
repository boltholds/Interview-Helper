from pydantic import BaseModel, Field


class KnowledgeSearchItem(BaseModel):
    chunk_id: str
    content: str
    source_path: str
    title: str
    ordinal: int
    metadata: dict[str, str]
    bm25_score: float


class KnowledgeSearchResponse(BaseModel):
    query: str
    total: int = Field(ge=0)
    items: list[KnowledgeSearchItem]
