# AgentForge - AI Agent Marketplace

A marketplace for creating, sharing, and using AI agents with real capabilities.

## Features

- **Create AI Agents** - Build custom agents with specific personalities and capabilities
- **Real Agent Powers** - Agents can search the web, run code, perform calculations, and more
- **Browse & Discover** - Find agents created by the community
- **Like & Share** - Support your favorite agents

## Agent Capabilities

Agents can be equipped with real tools:
- **Web Search** - Find current news, prices, and facts
- **Code Execution** - Write and run Python code
- **Calculator** - Perform complex math calculations
- **Text Analysis** - Extract information from documents
- **API Access** - Fetch live data from external APIs
- **Multi-step Reasoning** - Break down complex problems

## Local Development

### Prerequisites
- Python 3.10+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentforge.git
cd agentforge
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key-here
# On Windows: set OPENAI_API_KEY=your-api-key-here
```

5. Run the app:
```bash
python main.py
```

6. Open http://localhost:8001 in your browser

### Demo Account
- Email: `demo@agentforge.com`
- Password: `demo123`

## Deploy to Railway

### One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

### Manual Deploy

1. Push this repo to GitHub

2. Create a new project on [Railway](https://railway.app)

3. Connect your GitHub repository

4. Add environment variable:
   - `OPENAI_API_KEY` = your OpenAI API key

5. Deploy! Railway will automatically:
   - Detect the Python app
   - Install dependencies from `requirements.txt`
   - Start the app using the `Procfile`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key for AI features |
| `PORT` | No | Port to run on (auto-set by Railway) |

## Tech Stack

- **Backend**: FastAPI + Python
- **Database**: SQLite (file-based)
- **AI**: OpenAI GPT-4 with function calling
- **Frontend**: Jinja2 templates + vanilla JS
- **Styling**: Custom CSS (dark theme)

## Project Structure

```
agentforge/
├── main.py              # FastAPI app & routes
├── database.py          # SQLAlchemy setup
├── models.py            # Database models
├── auth.py              # Authentication
├── tools.py             # Agent tool implementations
├── agent_framework.py   # LangChain agent framework
├── templates/           # Jinja2 HTML templates
├── static/
│   ├── css/style.css    # Styles
│   └── js/app.js        # Frontend JS
├── requirements.txt     # Python dependencies
├── Procfile            # Railway/Heroku start command
└── railway.json        # Railway config
```

## API Endpoints

### Pages
- `GET /` - Homepage
- `GET /browse` - Browse agents
- `GET /agent/{id}` - Agent detail page
- `GET /create` - Create agent form
- `GET /login` - Login page
- `GET /signup` - Signup page

### API
- `POST /api/signup` - Create account
- `POST /api/login` - Login
- `GET /api/logout` - Logout
- `POST /api/agents` - Create agent
- `GET /api/agents` - List agents (JSON)
- `POST /api/agents/{id}/chat` - Chat with agent
- `POST /api/agents/{id}/like` - Like/unlike agent

## License

MIT
