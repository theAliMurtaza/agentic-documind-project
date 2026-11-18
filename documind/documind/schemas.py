"""Pydantic models for the agent's final answer, plus a citation parser."""
import re                                          # Regular expressions, used to find citation tags.

from pydantic import BaseModel, Field              # BaseModel validates data; Field adds descriptions.

# Matches citations such as [notes.pdf p.12] inside tool results and answers.
CITATION_PATTERN = re.compile(r"\[([^\[\]]+?) p\.(\d+)\]")   # Group 1 = filename, group 2 = page number.


class Citation(BaseModel):
    """One place in the documents that supports the answer."""

    source: str = Field(description="PDF filename exactly as shown in the tool result")
    page: int = Field(description="Page number exactly as shown in the tool result")


class Answer(BaseModel):
    """The validated final answer returned to the UI."""

    text: str = Field(description="The final answer in plain language. Be concise.")
    citations: list[Citation] = Field(
        default_factory=list,                      # Defaults to an empty list, not a shared mutable object.
        description="Only [filename p.page] tags that appear in the tool results",
    )
    found: bool = Field(
        description=(
            "True if the question was answered from the documents or by a "
            "calculation. False if the documents do not cover the question."
        )
    )


def parse_citations(text: str) -> set[tuple[str, int]]:
    """Return every (source, page) pair that appears as [source p.page] in text."""
    return {(m.group(1), int(m.group(2))) for m in CITATION_PATTERN.finditer(text)}   # A set removes duplicates.