from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str
    probabilite: float

class Token(BaseModel):
    access_token: str
    token_type: str
