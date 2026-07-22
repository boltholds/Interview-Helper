from pydantic import BaseModel, Field


class KnowledgeSearchItem(BaseModel):
    chunk_id: str
    content: str
    source_path: str
    title: str
    ordinal: int
    metadata: dict[str, str]
    bm25_score: float
    semantic_score: float
    bm25_normalized: float = Field(ge=0, le=1)
    semantic_normalized: float = Field(ge=0, le=1)
    metadata_score: float = Field(ge=0, le=1)
    final_score: float = Field(ge=0, le=1)


class KnowledgeSearchResponse(BaseModel):
    query: str
    mode: str = "hybrid"
    total: int = Field(ge=0)
    items: list[KnowledgeSearchItem]
