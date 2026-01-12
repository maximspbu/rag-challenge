from typing import List, Literal, Union, Any, Dict
from pydantic import BaseModel, Field


class Question(BaseModel):
    """
    Schema for an input question from the dataset.
    """

    question: str = Field(..., description="Answer on this question")
    kind: Literal["number", "name", "boolean", "names"] = Field(..., description="Answer should be this format")


class SourceReference(BaseModel):
    """
    Schema for citing a specific page in a PDF file.
    """

    pdf_sha1: str = Field(..., description="PDF filename (SHA1 hash)")
    page_index: int = Field(..., description="Zero-based physical page number in the PDF file")


class Answer(BaseModel):
    """
    Schema for the LLM's structured output.
    """

    value: Union[float, str, bool, List[str], Literal["N/A"]] = Field(..., description="Answer to the question")
    references: List[SourceReference] = Field([], description="List of exact filenames and pages used")


class AnswerSubmission(BaseModel):
    """
    Schema for the final JSON submission file.
    """

    team_email: str
    submission_name: str
    answers: List[Dict[str, Any]]


class ReformulatedQuery(BaseModel):
    """
    Schema for the output of the query reformulation step.
    """

    reformulated_query: str = Field(..., description="Reformulated query")


class SearchQuery(BaseModel):
    """
    Schema for the output of the query analysis step (company extraction).
    """

    extracted_company: str = Field(description="Company name mentioned in the question. Return empty string if none.")
    search_query: str = Field(description="Refined search query for semantic search")
