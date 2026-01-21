"""AgentForge - AI Agent Marketplace Backend."""
import os
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, Request, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_
from pydantic import BaseModel, EmailStr

from database import engine, get_db, Base
from models import User, Agent, AgentUse, AgentLike
from auth import hash_password, verify_password, create_session_token, get_user_from_token

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AgentForge", description="AI Agent Marketplace")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Pydantic models for API
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class AgentCreate(BaseModel):
    name: str
    description: str
    category: str
    system_prompt: str
    icon: str = "🤖"
    is_free: bool = True
    price: float = 0.0


class AgentChat(BaseModel):
    message: str
    conversation_history: List[dict] = []


# Helper to get current user from cookie
def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    token = request.cookies.get("session_token")
    if not token:
        return None
    user_id = get_user_from_token(token)
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


# ============== PAGE ROUTES ==============

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request, db: Session = Depends(get_db)):
    """Render the homepage with featured agents."""
    user = get_current_user(request, db)

    # Get featured agents (top by likes)
    featured_agents = db.query(Agent).order_by(Agent.likes_count.desc()).limit(6).all()

    # Get categories with counts
    categories = db.query(Agent.category).distinct().all()
    categories = [c[0] for c in categories if c[0]]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "featured_agents": featured_agents,
        "categories": categories
    })


@app.get("/browse", response_class=HTMLResponse)
async def browse_agents(
    request: Request,
    category: Optional[str] = None,
    search: Optional[str] = None,
    sort: str = "popular",
    db: Session = Depends(get_db)
):
    """Browse all agents with filters."""
    user = get_current_user(request, db)

    query = db.query(Agent)

    # Apply filters
    if category:
        query = query.filter(Agent.category == category)

    if search:
        query = query.filter(
            or_(
                Agent.name.ilike(f"%{search}%"),
                Agent.description.ilike(f"%{search}%")
            )
        )

    # Apply sorting
    if sort == "popular":
        query = query.order_by(Agent.uses_count.desc())
    elif sort == "newest":
        query = query.order_by(Agent.created_at.desc())
    elif sort == "likes":
        query = query.order_by(Agent.likes_count.desc())

    agents = query.all()

    # Get all categories for filter
    categories = db.query(Agent.category).distinct().all()
    categories = [c[0] for c in categories if c[0]]

    return templates.TemplateResponse("browse.html", {
        "request": request,
        "user": user,
        "agents": agents,
        "categories": categories,
        "current_category": category,
        "current_search": search,
        "current_sort": sort
    })


@app.get("/agent/{agent_id}", response_class=HTMLResponse)
async def agent_detail(request: Request, agent_id: int, db: Session = Depends(get_db)):
    """View agent details and demo."""
    user = get_current_user(request, db)

    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check if user has liked this agent
    user_liked = False
    if user:
        like = db.query(AgentLike).filter(
            AgentLike.agent_id == agent_id,
            AgentLike.user_id == user.id
        ).first()
        user_liked = like is not None

    return templates.TemplateResponse("agent_detail.html", {
        "request": request,
        "user": user,
        "agent": agent,
        "user_liked": user_liked
    })


@app.get("/create", response_class=HTMLResponse)
async def create_agent_page(request: Request, db: Session = Depends(get_db)):
    """Agent creation page."""
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login?next=/create", status_code=302)

    categories = [
        "Writing", "Coding", "Analysis", "Creative", "Education",
        "Business", "Research", "Entertainment", "Productivity", "Other"
    ]

    return templates.TemplateResponse("create_agent.html", {
        "request": request,
        "user": user,
        "categories": categories
    })


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/", db: Session = Depends(get_db)):
    """Login page."""
    user = get_current_user(request, db)
    if user:
        return RedirectResponse(url="/", status_code=302)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "user": None,
        "next": next
    })


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request, db: Session = Depends(get_db)):
    """Signup page."""
    user = get_current_user(request, db)
    if user:
        return RedirectResponse(url="/", status_code=302)

    return templates.TemplateResponse("signup.html", {
        "request": request,
        "user": None
    })


# ============== API ROUTES ==============

@app.post("/api/signup")
async def api_signup(
    request: Request,
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Create a new user account."""
    error = None

    # Validate input
    if len(password) < 6:
        error = "Password must be at least 6 characters"
    elif len(username) < 3:
        error = "Username must be at least 3 characters"
    elif db.query(User).filter(User.email == email).first():
        error = "Email already registered. Try logging in instead."
    elif db.query(User).filter(User.username == username).first():
        error = "Username already taken. Please choose another."

    if error:
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "user": None,
            "error": error,
            "email": email,
            "username": username
        })

    # Create user
    try:
        user = User(
            email=email,
            username=username,
            password_hash=hash_password(password),
            is_creator=False
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # Create session
        token = create_session_token(user.id)

        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="session_token", value=token, httponly=True, max_age=604800)
        return response
    except Exception as e:
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "user": None,
            "error": f"Database error: {str(e)}",
            "email": email,
            "username": username
        })


@app.post("/api/login")
async def api_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
    db: Session = Depends(get_db)
):
    """Log in user."""
    # Check if user exists
    user = db.query(User).filter(User.email == email).first()

    if not user:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "user": None,
            "error": "No account found with this email. Please sign up first.",
            "next": next,
            "email": email
        })

    if not verify_password(password, user.password_hash):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "user": None,
            "error": "Incorrect password. Please try again.",
            "next": next,
            "email": email
        })

    token = create_session_token(user.id)

    response = RedirectResponse(url=next, status_code=302)
    response.set_cookie(key="session_token", value=token, httponly=True, max_age=604800)
    return response


@app.get("/api/logout")
async def api_logout(request: Request):
    """Log out user."""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key="session_token")
    return response


@app.post("/api/agents")
async def api_create_agent(
    request: Request,
    name: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    system_prompt: str = Form(...),
    icon: str = Form("🤖"),
    is_free: str = Form("true"),
    price: float = Form(0.0),
    tools: List[str] = Form(default=[]),
    workflow: str = Form("custom"),
    db: Session = Depends(get_db)
):
    """Create a new agent."""
    user = get_current_user(request, db)

    # Get categories for error responses
    categories = [
        "Writing", "Coding", "Analysis", "Creative", "Education",
        "Business", "Research", "Entertainment", "Productivity", "Other"
    ]

    if not user:
        return templates.TemplateResponse("create_agent.html", {
            "request": request,
            "user": None,
            "categories": categories,
            "error": "You must be logged in to create an agent."
        })

    # Convert is_free string to boolean
    is_free_bool = is_free.lower() == "true"

    # Valid tools and workflows
    valid_tools = ["web_search", "run_python", "calculator", "analyze_text", "http_request", "multi_step"]
    valid_workflows = ["custom", "research", "code_assistant", "data_analyst", "task_executor"]
    enabled_tools = [t for t in tools if t in valid_tools]
    selected_workflow = workflow if workflow in valid_workflows else "custom"

    # Validation
    error = None
    if not name or len(name.strip()) < 3:
        error = "Agent name must be at least 3 characters."
    elif not description or len(description.strip()) < 10:
        error = "Description must be at least 10 characters."
    elif not category:
        error = "Please select a category."
    elif not system_prompt or len(system_prompt.strip()) < 20:
        error = "System prompt must be at least 20 characters."

    if error:
        return templates.TemplateResponse("create_agent.html", {
            "request": request,
            "user": user,
            "categories": categories,
            "error": error,
            "form_data": {
                "name": name,
                "description": description,
                "category": category,
                "system_prompt": system_prompt,
                "icon": icon,
                "is_free": is_free,
                "price": price,
                "tools": tools
            }
        })

    try:
        # Store tools and workflow in config
        config = {
            "tools": enabled_tools,
            "workflow": selected_workflow,
            "use_langchain": True  # Enable LangChain ReAct agent
        }

        agent = Agent(
            creator_id=user.id,
            name=name.strip(),
            description=description.strip(),
            category=category,
            system_prompt=system_prompt.strip(),
            icon=icon or "🤖",
            is_free=is_free_bool,
            price=price if not is_free_bool else 0.0,
            config=config
        )
        db.add(agent)

        # Mark user as creator
        user.is_creator = True

        db.commit()
        db.refresh(agent)

        return RedirectResponse(url=f"/agent/{agent.id}", status_code=302)

    except Exception as e:
        return templates.TemplateResponse("create_agent.html", {
            "request": request,
            "user": user,
            "categories": categories,
            "error": f"Failed to create agent: {str(e)}",
            "form_data": {
                "name": name,
                "description": description,
                "category": category,
                "system_prompt": system_prompt,
                "icon": icon,
                "is_free": is_free,
                "price": price,
                "tools": tools,
                "workflow": workflow
            }
        })


@app.post("/api/agents/{agent_id}/like")
async def api_like_agent(agent_id: int, request: Request, db: Session = Depends(get_db)):
    """Like or unlike an agent."""
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check if already liked
    existing_like = db.query(AgentLike).filter(
        AgentLike.agent_id == agent_id,
        AgentLike.user_id == user.id
    ).first()

    if existing_like:
        # Unlike
        db.delete(existing_like)
        agent.likes_count = max(0, agent.likes_count - 1)
        liked = False
    else:
        # Like
        like = AgentLike(agent_id=agent_id, user_id=user.id)
        db.add(like)
        agent.likes_count += 1
        liked = True

    db.commit()

    return {"liked": liked, "likes_count": agent.likes_count}


@app.post("/api/agents/{agent_id}/chat")
async def api_chat_with_agent(
    agent_id: int,
    request: Request,
    chat: AgentChat,
    db: Session = Depends(get_db)
):
    """Chat with an agent using LangChain ReAct framework."""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Record usage
    user = get_current_user(request, db)
    use = AgentUse(
        agent_id=agent_id,
        user_id=user.id if user else None
    )
    db.add(use)
    agent.uses_count += 1
    db.commit()

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "response": "⚠️ OpenAI API key not configured. This is a demo response.\n\n"
                       f"I am **{agent.name}** - {agent.description}\n\n"
                       f"You said: *{chat.message}*\n\n"
                       "To enable real AI responses, set the OPENAI_API_KEY environment variable.",
            "demo_mode": True,
            "actions": []
        }

    # Get agent configuration
    agent_config = agent.config or {}
    enabled_tools = agent_config.get("tools", [])
    workflow = agent_config.get("workflow", "custom")
    use_langchain = agent_config.get("use_langchain", True)

    # Create session ID for memory
    session_id = f"agent_{agent_id}_user_{user.id if user else 'anonymous'}"

    try:
        # Try to use LangChain agent framework
        if use_langchain:
            try:
                from agent_framework import run_agent, LANGCHAIN_AVAILABLE

                if LANGCHAIN_AVAILABLE:
                    result = run_agent(
                        agent_name=agent.name,
                        system_prompt=agent.system_prompt,
                        user_message=chat.message,
                        enabled_tools=enabled_tools,
                        workflow=workflow,
                        session_id=session_id,
                        conversation_history=chat.conversation_history
                    )

                    return {
                        "response": result.get("response", "No response"),
                        "demo_mode": False,
                        "actions": result.get("actions", []),
                        "workflow": workflow,
                        "framework": "langchain"
                    }
            except ImportError:
                pass  # Fall back to simple agent

        # Fallback: Use simple OpenAI function calling
        from agent_framework import run_simple_agent

        result = run_simple_agent(
            agent_name=agent.name,
            system_prompt=agent.system_prompt,
            user_message=chat.message,
            enabled_tools=enabled_tools,
            conversation_history=chat.conversation_history
        )

        return {
            "response": result.get("response", "No response"),
            "demo_mode": False,
            "actions": result.get("actions", []),
            "workflow": workflow,
            "framework": "openai"
        }

    except Exception as e:
        import traceback
        return {
            "response": f"Error: {str(e)}",
            "demo_mode": True,
            "actions": [],
            "error_details": traceback.format_exc()
        }


@app.get("/api/agents")
async def api_list_agents(
    category: Optional[str] = None,
    search: Optional[str] = None,
    sort: str = "popular",
    limit: int = Query(default=20, le=100),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List agents with filters (JSON API)."""
    query = db.query(Agent)

    if category:
        query = query.filter(Agent.category == category)

    if search:
        query = query.filter(
            or_(
                Agent.name.ilike(f"%{search}%"),
                Agent.description.ilike(f"%{search}%")
            )
        )

    if sort == "popular":
        query = query.order_by(Agent.uses_count.desc())
    elif sort == "newest":
        query = query.order_by(Agent.created_at.desc())
    elif sort == "likes":
        query = query.order_by(Agent.likes_count.desc())

    total = query.count()
    agents = query.offset(offset).limit(limit).all()

    return {
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "category": a.category,
                "icon": a.icon,
                "is_free": a.is_free,
                "price": a.price,
                "uses_count": a.uses_count,
                "likes_count": a.likes_count,
                "creator": a.creator.username if a.creator else "Unknown"
            }
            for a in agents
        ],
        "total": total
    }


# ============== SEED DATA ==============

def seed_database(db: Session):
    """Add initial dummy data to the database."""
    # Check if already seeded
    if db.query(User).first():
        return

    # Create demo user
    demo_user = User(
        email="demo@agentforge.com",
        username="AgentForge",
        password_hash=hash_password("demo123"),
        is_creator=True
    )
    db.add(demo_user)
    db.commit()
    db.refresh(demo_user)

    # Create sample agents
    sample_agents = [
        {
            "name": "Code Reviewer",
            "description": "Expert code reviewer that analyzes your code for bugs, security issues, and best practices. Supports multiple languages.",
            "category": "Coding",
            "system_prompt": "You are an expert code reviewer. Analyze code for bugs, security vulnerabilities, performance issues, and adherence to best practices. Provide constructive feedback with specific suggestions for improvement.",
            "icon": "🔍",
            "uses_count": 1542,
            "likes_count": 234
        },
        {
            "name": "Story Writer",
            "description": "Creative writing assistant that helps you craft engaging stories, develop characters, and overcome writer's block.",
            "category": "Creative",
            "system_prompt": "You are a creative writing assistant. Help users craft engaging stories with vivid descriptions, compelling characters, and interesting plot twists. Adapt your style to match the genre they're working in.",
            "icon": "📚",
            "uses_count": 2103,
            "likes_count": 456
        },
        {
            "name": "SQL Helper",
            "description": "Database expert that helps write, optimize, and debug SQL queries. Explains complex queries in simple terms.",
            "category": "Coding",
            "system_prompt": "You are a SQL expert. Help users write efficient SQL queries, explain complex queries, optimize performance, and debug issues. Support multiple database systems including PostgreSQL, MySQL, and SQLite.",
            "icon": "🗃️",
            "uses_count": 987,
            "likes_count": 167
        },
        {
            "name": "Email Composer",
            "description": "Professional email writer that helps craft clear, effective emails for any situation - from cold outreach to follow-ups.",
            "category": "Business",
            "system_prompt": "You are a professional email writing assistant. Help users compose clear, effective, and appropriately toned emails for various business situations. Consider the audience, purpose, and desired outcome.",
            "icon": "✉️",
            "uses_count": 3421,
            "likes_count": 521
        },
        {
            "name": "Math Tutor",
            "description": "Patient math tutor that explains concepts step-by-step, from algebra to calculus. Perfect for students of all levels.",
            "category": "Education",
            "system_prompt": "You are a patient and encouraging math tutor. Explain mathematical concepts step-by-step, use multiple approaches when needed, and provide practice problems. Adapt your explanations to the student's level.",
            "icon": "🧮",
            "uses_count": 1876,
            "likes_count": 312
        },
        {
            "name": "Resume Builder",
            "description": "Career expert that helps optimize your resume with impactful bullet points and ATS-friendly formatting.",
            "category": "Business",
            "system_prompt": "You are a career expert and resume specialist. Help users create impactful resumes with strong action verbs, quantified achievements, and ATS-friendly formatting. Tailor advice to their industry and experience level.",
            "icon": "📄",
            "uses_count": 2654,
            "likes_count": 398
        },
        {
            "name": "Debate Partner",
            "description": "Skilled debater that helps you explore arguments from multiple perspectives and strengthen your reasoning.",
            "category": "Education",
            "system_prompt": "You are a skilled debate partner. Help users explore topics from multiple perspectives, strengthen their arguments, identify logical fallacies, and develop critical thinking skills. Play devil's advocate when helpful.",
            "icon": "⚖️",
            "uses_count": 743,
            "likes_count": 89
        },
        {
            "name": "Recipe Creator",
            "description": "Creative chef that generates recipes based on ingredients you have, dietary restrictions, and cuisine preferences.",
            "category": "Creative",
            "system_prompt": "You are a creative chef and recipe developer. Create delicious recipes based on available ingredients, accommodate dietary restrictions, and explain cooking techniques. Provide substitution suggestions when needed.",
            "icon": "👨‍🍳",
            "uses_count": 1234,
            "likes_count": 267
        },
        {
            "name": "Data Analyst",
            "description": "Data analysis expert that helps interpret data, suggest visualizations, and derive insights from datasets.",
            "category": "Analysis",
            "system_prompt": "You are a data analysis expert. Help users interpret data, suggest appropriate visualizations, identify patterns and trends, and derive actionable insights. Explain statistical concepts clearly.",
            "icon": "📊",
            "uses_count": 892,
            "likes_count": 145
        },
        {
            "name": "Fitness Coach",
            "description": "Personal trainer that creates workout plans, explains exercises, and provides motivation for your fitness journey.",
            "category": "Productivity",
            "system_prompt": "You are a knowledgeable fitness coach. Create personalized workout plans, explain proper exercise form, provide nutrition guidance, and offer motivation. Adapt recommendations to the user's fitness level and goals.",
            "icon": "💪",
            "uses_count": 1567,
            "likes_count": 234
        },
        {
            "name": "Legal Advisor",
            "description": "Legal information assistant that explains legal concepts, helps understand documents, and provides general guidance.",
            "category": "Business",
            "system_prompt": "You are a legal information assistant. Help users understand legal concepts, explain documents in plain language, and provide general guidance. Always clarify you're not providing legal advice and recommend consulting a lawyer for specific situations.",
            "icon": "⚖️",
            "uses_count": 654,
            "likes_count": 98
        },
        {
            "name": "Travel Planner",
            "description": "Travel expert that helps plan trips, suggests destinations, creates itineraries, and provides local tips.",
            "category": "Entertainment",
            "system_prompt": "You are an experienced travel planner. Help users plan trips, suggest destinations based on preferences, create detailed itineraries, and share local tips and cultural insights. Consider budget, time, and travel style.",
            "icon": "✈️",
            "uses_count": 2341,
            "likes_count": 412
        }
    ]

    for agent_data in sample_agents:
        agent = Agent(
            creator_id=demo_user.id,
            **agent_data
        )
        db.add(agent)

    db.commit()
    print("Database seeded with sample data!")


@app.on_event("startup")
async def startup_event():
    """Initialize database with seed data on startup."""
    db = next(get_db())
    seed_database(db)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
