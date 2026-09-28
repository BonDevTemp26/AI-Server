from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ModelBase(BaseModel):
    name: str
    type: str

class ModelCreate(ModelBase):
    pass

class ModelResponse(ModelBase):
    id: int
    version: str
    file_path: Optional[str]
    status: str
    is_default: bool
    created_at: datetime

    model_config = {"from_attributes": True}

class ModelTypeBase(BaseModel):
    name: str
    display_name: str

class ModelTypeCreate(ModelTypeBase):
    pass

class ModelTypeResponse(ModelTypeBase):
    id: int

    model_config = {"from_attributes": True}
