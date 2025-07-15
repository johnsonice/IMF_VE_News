
from typing import List, Optional
from pydantic import BaseModel, Field
class CountryIdentificationResponse(BaseModel):
    main_country: str = Field(..., 
                             description="The single primary country identified in the article; Follow IMF country name conventions. Use english onliy",
                             max_length=50)
    other_countries: List[str] = Field(
        default_factory=list,
        description="A de‑duplicated list of any other countries mentioned; Follow IMF country name conventions."
    )
    brief_reason: str = Field(..., 
                            description="Brief reasoning for country tagging, explaining why the main country was chosen and any context for other countries mentioned.",
                            max_length=500)