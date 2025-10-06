from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from database import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)

class Emotion(Base):
    __tablename__ = 'emotions'
    
    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String, index=True)