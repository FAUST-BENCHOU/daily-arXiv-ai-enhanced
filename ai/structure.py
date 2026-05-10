from pydantic import BaseModel, Field, field_validator
import re

class Structure(BaseModel):
    tldr: str = Field(description="generate a too long; didn't read summary")
    motivation: str = Field(description="describe the motivation in this paper")
    method: str = Field(description="method of this paper")
    result: str = Field(description="result of this paper")
    conclusion: str = Field(description="conclusion of this paper")


class StructureWithDigestFocus(Structure):
    digest_theme_relevant: bool = Field(
        description="True if this paper's topic clearly aligns with the user-provided digest focus (same broad subfield or directly applicable techniques); False if unrelated or only tangential."
    )