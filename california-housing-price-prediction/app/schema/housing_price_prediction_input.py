# schemas.py
from pydantic import BaseModel, Field
from typing import Literal


class HousePricePredictionInput(BaseModel):
    longitude: float = Field(..., ge=-125, le=-113)
    latitude: float = Field(..., ge=32, le=43)
    housing_median_age: float = Field(..., ge=1)
    total_rooms: float = Field(..., gt=0)
    total_bedrooms: float = Field(..., gt=0)
    population: float = Field(..., gt=0)
    households: float = Field(..., gt=0)
    median_income: float = Field(..., gt=0)
    ocean_proximity: Literal[
        '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
    ] = Field(..., description="Proximity to the ocean")
