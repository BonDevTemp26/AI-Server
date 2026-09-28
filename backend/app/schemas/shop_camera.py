from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class CameraBase(BaseModel):
    name: str
    rtsp_url: str
    is_active: Optional[bool] = True

class CameraCreate(CameraBase):
    pass

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    is_active: Optional[bool] = None

class CameraResponse(CameraBase):
    id: int
    shop_id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class ShopBase(BaseModel):
    name: str

class ShopCreate(ShopBase):
    pass

class ShopUpdate(ShopBase):
    pass

class ShopResponse(ShopBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class ShopWithCamerasResponse(ShopResponse):
    cameras: List[CameraResponse] = []
