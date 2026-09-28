from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from app.database import Base

class ModelType(Base):
    __tablename__ = "model_types"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(100), nullable=False)

class Shop(Base):
    __tablename__ = "shops"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    cameras = relationship("Camera", back_populates="shop", cascade="all, delete-orphan")

class Camera(Base):
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    shop_id = Column(Integer, ForeignKey("shops.id", ondelete="CASCADE"))
    name = Column(String(255), nullable=False)
    rtsp_url = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    motion_active = Column(Boolean, default=False)
    theft_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (UniqueConstraint('shop_id', 'name', name='_shop_camera_uc'),)
    
    shop = relationship("Shop", back_populates="cameras")
    assignments = relationship("CameraModelAssignment", back_populates="camera", cascade="all, delete-orphan")

class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    type = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    file_path = Column(String)
    status = Column(String(50), default="pending")
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CameraModelAssignment(Base):
    __tablename__ = "camera_model_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"))
    model_id = Column(Integer, ForeignKey("models.id", ondelete="CASCADE"))
    is_running = Column(Boolean, default=False)
    started_at = Column(DateTime(timezone=True))
    
    __table_args__ = (UniqueConstraint('camera_id', 'model_id', name='_camera_model_uc'),)
    
    camera = relationship("Camera", back_populates="assignments")
    model = relationship("Model")

class DetectionEvent(Base):
    __tablename__ = "detection_events"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"))
    model_id = Column(Integer, ForeignKey("models.id"))
    shop_id = Column(Integer, ForeignKey("shops.id"))
    shop_name = Column(String(255))
    camera_name = Column(String(255))
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True))
    detection_type = Column(String(100))
    confidence = Column(Float)
    metadata_ = Column("metadata", JSONB)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id", ondelete="CASCADE"))
    version_tag = Column(String(50))
    deployed_at = Column(DateTime(timezone=True), server_default=func.now())
    is_current = Column(Boolean, default=True)
