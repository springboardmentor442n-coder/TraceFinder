from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str



class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class PredictionBase(BaseModel):
    filename: str
    model_used: str
    prediction_result: str
    confidence: float

class PredictionCreate(PredictionBase):
    pass

class Prediction(PredictionBase):
    id: int
    timestamp: datetime
    user_id: int

    class Config:
        from_attributes = True

class User(UserBase):
    id: int
    predictions: List[Prediction] = []
    class Config:
        from_attributes = True
