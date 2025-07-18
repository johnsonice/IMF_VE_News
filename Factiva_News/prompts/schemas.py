
from typing import List, Optional
from pydantic import BaseModel, Field

class CountryIdentificationResponse(BaseModel):
    main_country: str = Field(
        default="",
        description="The single primary country identified in the article; Follow IMF country name conventions. Use English only.",
        max_length=50
    )
    other_countries: List[str] = Field(
        default_factory=list,
        description="A de‑duplicated list of other countries mentioned in the article; Follow IMF country name conventions. Must contain fewer than 6 countries.",
        max_length=100
    )