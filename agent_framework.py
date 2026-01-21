"""
AgentForge LangChain Framework
Provides ReAct agents, memory, tools, and workflow execution.
"""
import os
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# LangChain imports - all LangChain dependent code is guarded
LANGCHAIN_AVAILABLE = False
ChatOpenAI = None
AgentExecutor = None
create_react_agent = None
PromptTemplate = None
Tool = None
ConversationBufferWindowMemory = None

try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate
    from langchain.tools import Tool
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish
    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseCallbackHandler = object  # Fallback to basic object

# Conversation memory storage (in production, use Redis or database)
conversation_memories: Dict[str, Any] = {}


# ActionTracker - only functional when LangChain is available
class ActionTracker(BaseCallbackHandler):
    """Tracks agent actions for display in the UI."""

    def __init__(self):
        self.actions: List[Dict] = []
        self.thoughts: List[str] = []
        self.current_tool: Optional[str] = None

    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        self.actions.append({
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.current_tool = action.tool

    def on_tool_end(self, output: str, **kwargs):
        """Called when a tool finishes."""
        if self.actions and self.current_tool:
            self.actions[-1]["output"] = output[:1000]  # Limit output size
            self.actions[-1]["tool_name"] = TOOL_DISPLAY_NAMES.get(
                self.current_tool, self.current_tool
            )
            self.actions[-1]["tool_icon"] = TOOL_ICONS.get(
                self.current_tool, "🔧"
            )

    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes."""
        pass


# Tool display names and icons
TOOL_DISPLAY_NAMES = {
    "web_search": "Web Search",
    "calculator": "Calculator",
    "python_repl": "Python Code",
    "analyze_text": "Text Analysis",
    "http_request": "API Request",
    "get_current_time": "Get Time",
    "wikipedia": "Wikipedia",
    "create_document": "Create Document",
    "read_file": "Read File",
    "reasoning": "Reasoning Step",
}

TOOL_ICONS = {
    "web_search": "🔍",
    "calculator": "🔢",
    "python_repl": "💻",
    "analyze_text": "📄",
    "http_request": "🌐",
    "get_current_time": "🕐",
    "wikipedia": "📚",
    "create_document": "📝",
    "read_file": "📂",
    "reasoning": "🧠",
}


# ==================== TOOL IMPLEMENTATIONS ====================

def web_search(query: str) -> str:
    """Search the web for current information."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No results found for: {query}"

        output = f"Search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. **{r.get('title', 'No title')}**\n"
            output += f"   {r.get('body', 'No description')[:200]}\n"
            output += f"   Source: {r.get('href', 'Unknown')}\n\n"

        return output
    except Exception as e:
        return f"Search error: {str(e)}"


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    safe_dict = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'pow': pow, 'len': len,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
        'exp': math.exp, 'pi': math.pi, 'e': math.e,
        'floor': math.floor, 'ceil': math.ceil,
        'factorial': math.factorial,
    }
    try:
        # Clean expression
        expression = expression.strip()
        expression = expression.replace('^', '**')

        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def python_repl(code: str) -> str:
    """Execute Python code and return the output."""
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    # Restricted builtins
    safe_builtins = {
        'abs': abs, 'all': all, 'any': any, 'bool': bool,
        'dict': dict, 'enumerate': enumerate, 'filter': filter,
        'float': float, 'int': int, 'len': len, 'list': list,
        'map': map, 'max': max, 'min': min, 'print': print,
        'range': range, 'round': round, 'set': set, 'sorted': sorted,
        'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
        'True': True, 'False': False, 'None': None,
    }

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    exec_globals = {
        '__builtins__': safe_builtins,
        'math': math,
        'json': json,
        're': re,
    }
    exec_locals = {}

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals, exec_locals)

        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        output = ""
        if stdout:
            output += f"Output:\n{stdout}\n"
        if stderr:
            output += f"Errors:\n{stderr}\n"
        if not output:
            # Return last assigned variable
            for var in reversed(list(exec_locals.keys())):
                if not var.startswith('_'):
                    output = f"Result: {exec_locals[var]}"
                    break

        return output or "Code executed successfully (no output)"

    except Exception as e:
        return f"Execution error: {type(e).__name__}: {str(e)}"


def analyze_text(text: str, analysis_type: str = "summary") -> str:
    """Analyze text content."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)

    if analysis_type == "summary" or analysis_type == "stats":
        return f"""Text Analysis:
- Characters: {len(text)}
- Words: {len(words)}
- Sentences: {len([s for s in sentences if s.strip()])}
- Avg word length: {sum(len(w) for w in words) / len(words):.1f} chars
- Avg sentence length: {len(words) / max(len(sentences), 1):.1f} words"""

    elif analysis_type == "emails":
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        return f"Found {len(emails)} emails: {', '.join(emails) if emails else 'None'}"

    elif analysis_type == "urls":
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
        return f"Found {len(urls)} URLs: {', '.join(urls) if urls else 'None'}"

    elif analysis_type == "numbers":
        numbers = re.findall(r'-?\d+\.?\d*', text)
        nums = [float(n) if '.' in n else int(n) for n in numbers]
        if nums:
            return f"Found {len(nums)} numbers. Sum: {sum(nums)}, Avg: {sum(nums)/len(nums):.2f}"
        return "No numbers found"

    return f"Unknown analysis type: {analysis_type}"


def http_request(url: str, method: str = "GET") -> str:
    """Make an HTTP request."""
    import urllib.request
    import urllib.error

    # Security: limit to safe domains
    allowed = ['api.github.com', 'jsonplaceholder.typicode.com', 'httpbin.org',
               'api.publicapis.org', 'catfact.ninja', 'dog.ceo', 'api.coindesk.com']

    if not any(d in url for d in allowed):
        return f"URL not allowed. Allowed domains: {', '.join(allowed)}"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AgentForge/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')[:2000]
            try:
                data = json.loads(content)
                return f"Response (JSON):\n{json.dumps(data, indent=2)[:1500]}"
            except:
                return f"Response:\n{content[:1000]}"
    except Exception as e:
        return f"Request error: {str(e)}"


def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time."""
    now = datetime.utcnow()
    return f"Current time (UTC): {now.strftime('%A, %B %d, %Y at %H:%M:%S')}"


def reasoning_step(thought: str) -> str:
    """Document a reasoning step. Use this to think through complex problems."""
    return f"Recorded thought: {thought}"


def create_document(title: str, content: str) -> str:
    """Create a document/note with the given content."""
    return f"Document created:\n\nTitle: {title}\n\n{content}"


# ==================== TOOL DEFINITIONS ====================

def get_langchain_tools(enabled_tools: List[str]) -> List[Tool]:
    """Get LangChain tool objects for enabled tools."""
    if not LANGCHAIN_AVAILABLE or Tool is None:
        return []

    all_tools = {
        "web_search": Tool(
            name="web_search",
            func=web_search,
            description="Search the web for current information, news, or facts. Input should be a search query string."
        ),
        "calculator": Tool(
            name="calculator",
            func=calculator,
            description="Evaluate mathematical expressions. Input should be a valid math expression like '2 + 2' or 'sqrt(16) * 3'."
        ),
        "run_python": Tool(
            name="python_repl",
            func=python_repl,
            description="Execute Python code. Input should be valid Python code. Use print() to output results."
        ),
        "analyze_text": Tool(
            name="analyze_text",
            func=lambda x: analyze_text(x, "summary"),
            description="Analyze text for statistics like word count, sentence count, etc. Input should be the text to analyze."
        ),
        "http_request": Tool(
            name="http_request",
            func=http_request,
            description="Make HTTP GET requests to APIs. Input should be a URL. Limited to safe public APIs."
        ),
        "multi_step": Tool(
            name="think_step",
            func=reasoning_step,
            description="Use this to reason through complex problems step-by-step. Input should describe your current thought and what you plan to do next. Use this BEFORE taking actions on complex tasks."
        ),
    }

    # Always include these utility tools
    tools = [
        Tool(
            name="get_current_time",
            func=get_current_time,
            description="Get the current date and time. No input required."
        ),
    ]

    # Add reasoning tool if multi_step is enabled
    if "multi_step" in enabled_tools:
        tools.append(Tool(
            name="reasoning",
            func=reasoning_step,
            description="Document your reasoning process. Use this to think through complex problems step by step. Input should be your thought or reasoning."
        ))

    for tool_name in enabled_tools:
        if tool_name in all_tools:
            tools.append(all_tools[tool_name])

    return tools


# ==================== REACT AGENT PROMPT ====================

REACT_PROMPT = """You are {agent_name}, an AI agent with the following capabilities and personality:

{system_prompt}

You have access to the following tools:

{tools}

IMPORTANT: Use the ReAct (Reasoning + Acting) pattern:
1. Think about what you need to do
2. Decide which tool to use (if any)
3. Use the tool and observe the result
4. Repeat until you have enough information
5. Provide your final answer

Use this EXACT format:

Question: the input question you must answer
Thought: think about what to do, break down the problem
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to ALWAYS use tools when you need current information or to perform calculations.

Question: {input}
{agent_scratchpad}"""


# ==================== WORKFLOW DEFINITIONS ====================

WORKFLOW_TEMPLATES = {
    "research": {
        "name": "Research Assistant",
        "description": "Searches multiple sources, synthesizes information, and provides cited answers",
        "steps": [
            {"action": "search", "description": "Search for relevant information"},
            {"action": "analyze", "description": "Analyze and extract key points"},
            {"action": "synthesize", "description": "Combine findings into coherent answer"},
            {"action": "cite", "description": "Add sources and citations"}
        ],
        "required_tools": ["web_search"],
        "prompt_addition": """
When researching:
1. Search for multiple perspectives on the topic
2. Extract key facts and findings
3. Synthesize information into a clear answer
4. Always cite your sources with URLs when available"""
    },
    "code_assistant": {
        "name": "Code Assistant",
        "description": "Writes, tests, and explains code with execution capabilities",
        "steps": [
            {"action": "understand", "description": "Understand the coding requirement"},
            {"action": "plan", "description": "Plan the solution approach"},
            {"action": "implement", "description": "Write the code"},
            {"action": "test", "description": "Test the code by running it"},
            {"action": "explain", "description": "Explain the solution"}
        ],
        "required_tools": ["run_python", "calculator"],
        "prompt_addition": """
When coding:
1. First understand what the user needs
2. Plan your approach before writing code
3. Write clean, well-commented code
4. ALWAYS test your code by running it
5. Explain what the code does and how it works"""
    },
    "data_analyst": {
        "name": "Data Analyst",
        "description": "Analyzes data, performs calculations, and creates insights",
        "steps": [
            {"action": "collect", "description": "Gather relevant data"},
            {"action": "clean", "description": "Validate and clean data"},
            {"action": "analyze", "description": "Perform analysis and calculations"},
            {"action": "visualize", "description": "Describe findings clearly"},
            {"action": "recommend", "description": "Provide actionable insights"}
        ],
        "required_tools": ["calculator", "run_python", "analyze_text"],
        "prompt_addition": """
When analyzing data:
1. Clearly state your methodology
2. Show your calculations step by step
3. Validate results for reasonableness
4. Present findings clearly with numbers
5. Provide actionable recommendations"""
    },
    "task_executor": {
        "name": "Task Executor",
        "description": "Breaks down complex tasks and executes them step by step",
        "steps": [
            {"action": "decompose", "description": "Break task into subtasks"},
            {"action": "prioritize", "description": "Order subtasks by dependency"},
            {"action": "execute", "description": "Execute each subtask"},
            {"action": "verify", "description": "Verify completion"},
            {"action": "report", "description": "Report results"}
        ],
        "required_tools": [],
        "prompt_addition": """
When executing tasks:
1. Break complex tasks into smaller steps
2. Identify dependencies between steps
3. Execute steps in the correct order
4. Verify each step completed successfully
5. Provide a summary of what was accomplished"""
    },
    "custom": {
        "name": "Custom Workflow",
        "description": "Define your own workflow",
        "steps": [],
        "required_tools": [],
        "prompt_addition": ""
    }
}


# ==================== MEMORY MANAGEMENT ====================

def get_memory(session_id: str, window_size: int = 10) -> ConversationBufferWindowMemory:
    """Get or create conversation memory for a session."""
    if not LANGCHAIN_AVAILABLE:
        return None

    if session_id not in conversation_memories:
        conversation_memories[session_id] = ConversationBufferWindowMemory(
            k=window_size,
            memory_key="chat_history",
            return_messages=True
        )
    return conversation_memories[session_id]


def clear_memory(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in conversation_memories:
        del conversation_memories[session_id]


# ==================== AGENT EXECUTION ====================

def run_agent(
    agent_name: str,
    system_prompt: str,
    user_message: str,
    enabled_tools: List[str],
    workflow: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_history: List[Dict] = None
) -> Dict[str, Any]:
    """
    Run a LangChain ReAct agent with the given configuration.

    Returns:
        Dict with 'response', 'actions', 'thoughts', and 'error' keys
    """
    if not LANGCHAIN_AVAILABLE:
        return {
            "response": "LangChain is not installed. Please run: pip install langchain langchain-openai",
            "actions": [],
            "error": "langchain_not_available"
        }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "response": "OpenAI API key not configured. Set OPENAI_API_KEY environment variable.",
            "actions": [],
            "error": "no_api_key"
        }

    try:
        # Get tools
        tools = get_langchain_tools(enabled_tools)

        # Apply workflow template if specified
        enhanced_prompt = system_prompt
        if workflow and workflow in WORKFLOW_TEMPLATES:
            wf = WORKFLOW_TEMPLATES[workflow]
            enhanced_prompt += "\n" + wf.get("prompt_addition", "")

            # Add required tools from workflow
            for req_tool in wf.get("required_tools", []):
                if req_tool not in enabled_tools:
                    enabled_tools.append(req_tool)
            tools = get_langchain_tools(enabled_tools)

        # Create the prompt
        prompt = PromptTemplate.from_template(REACT_PROMPT)

        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=api_key
        )

        # Create the ReAct agent
        agent = create_react_agent(llm, tools, prompt)

        # Create action tracker
        tracker = ActionTracker()

        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            callbacks=[tracker]
        )

        # Build context from conversation history
        context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = "User" if msg.get("role") == "user" else "Assistant"
                context += f"{role}: {msg.get('content', '')}\n"

        if context:
            full_input = f"Previous conversation:\n{context}\n\nCurrent question: {user_message}"
        else:
            full_input = user_message

        # Run the agent
        result = agent_executor.invoke({
            "input": full_input,
            "agent_name": agent_name,
            "system_prompt": enhanced_prompt,
        })

        # Format actions for display
        formatted_actions = []
        for action in tracker.actions:
            formatted_actions.append({
                "tool": action.get("tool", "unknown"),
                "tool_name": action.get("tool_name", action.get("tool", "Unknown")),
                "tool_icon": action.get("tool_icon", "🔧"),
                "input": action.get("tool_input", ""),
                "output": action.get("output", ""),
                "thought": action.get("log", "").split("Action:")[0].replace("Thought:", "").strip()
            })

        return {
            "response": result.get("output", "No response generated"),
            "actions": formatted_actions,
            "error": None
        }

    except Exception as e:
        import traceback
        return {
            "response": f"Agent error: {str(e)}",
            "actions": [],
            "error": traceback.format_exc()
        }


# ==================== SIMPLE FALLBACK (NO LANGCHAIN) ====================

def run_simple_agent(
    agent_name: str,
    system_prompt: str,
    user_message: str,
    enabled_tools: List[str],
    conversation_history: List[Dict] = None
) -> Dict[str, Any]:
    """
    Fallback agent using direct OpenAI API when LangChain isn't available.
    """
    from tools import get_tools_for_agent, execute_tool, TOOL_INFO
    import json

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "response": "OpenAI API key not configured.",
            "actions": [],
            "error": "no_api_key"
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        tools = get_tools_for_agent(enabled_tools)

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        actions_taken = []

        for _ in range(5):
            if tools:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=2048,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            else:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=2048,
                    messages=messages
                )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                })

                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    result = execute_tool(tc.function.name, args)

                    tool_info = TOOL_INFO.get(tc.function.name, {})
                    actions_taken.append({
                        "tool": tc.function.name,
                        "tool_name": tool_info.get("name", tc.function.name),
                        "tool_icon": tool_info.get("icon", "🔧"),
                        "input": args,
                        "output": result
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result)
                    })
            else:
                return {
                    "response": msg.content or "",
                    "actions": actions_taken,
                    "error": None
                }

        return {
            "response": msg.content or "Max iterations reached",
            "actions": actions_taken,
            "error": None
        }

    except Exception as e:
        import traceback
        return {
            "response": f"Error: {str(e)}",
            "actions": [],
            "error": traceback.format_exc()
        }
