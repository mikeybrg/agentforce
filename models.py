"""SQLAlchemy database models for AgentForge."""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float,
    DateTime, ForeignKey, JSON, UniqueConstraint
)
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    """User model for authentication and agent creation."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_creator = Column(Boolean, default=False)

    # Relationships
    agents = relationship("Agent", back_populates="creator")
    agent_uses = relationship("AgentUse", back_populates="user")
    agent_likes = relationship("AgentLike", back_populates="user")


class Agent(Base):
    """AI Agent model."""
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=False)
    category = Column(String(100), index=True)
    system_prompt = Column(Text, nullable=False)
    config = Column(JSON, default={})
    price = Column(Float, default=0.0)
    is_free = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    uses_count = Column(Integer, default=0)
    likes_count = Column(Integer, default=0)
    icon = Column(String(50), default="🤖")

    # Relationships
    creator = relationship("User", back_populates="agents")
    uses = relationship("AgentUse", back_populates="agent")
    likes = relationship("AgentLike", back_populates="agent")


class AgentUse(Base):
    """Track agent usage."""
    __tablename__ = "agent_uses"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(100))

    # Relationships
    agent = relationship("Agent", back_populates="uses")
    user = relationship("User", back_populates="agent_uses")


class AgentLike(Base):
    """Track agent likes."""
    __tablename__ = "agent_likes"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Unique constraint: one like per user per agent
    __table_args__ = (
        UniqueConstraint('agent_id', 'user_id', name='unique_agent_like'),
    )

    # Relationships
    agent = relationship("Agent", back_populates="likes")
    user = relationship("User", back_populates="agent_likes")
