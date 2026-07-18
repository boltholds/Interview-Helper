from pydantic import BaseModel, Field


class CandidateProject(BaseModel):
    name: str
    facts: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    target_role: str
    skills: list[str] = Field(default_factory=list)
    projects: list[CandidateProject] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(
        default_factory=lambda: ["Do not invent experience absent from the profile"]
    )


class KnowledgeSource(BaseModel):
    title: str
    url: str | None = None
    fragment_id: str | None = None


class AnswerSuggestion(BaseModel):
    question: str
    key_points: list[str]
    suggested_answer: str
    missing_personal_fact: str | None = None
    sources: list[KnowledgeSource] = Field(default_factory=list)
