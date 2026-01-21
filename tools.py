"""AI Agent Tools - Real capabilities for agents."""
import json
import re
import math
import subprocess
import tempfile
import os
from typing import Any
from datetime import datetime

# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = {
    "multi_step": {
        "type": "function",
        "function": {
            "name": "think_step",
            "description": "Use this to think through a problem step by step. Document your reasoning before taking action. Always use this for complex problems.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_number": {
                        "type": "integer",
                        "description": "The current step number in your reasoning"
                    },
                    "thought": {
                        "type": "string",
                        "description": "Your thought or reasoning for this step"
                    },
                    "next_action": {
                        "type": "string",
                        "description": "What you plan to do next based on this reasoning"
                    }
                },
                "required": ["step_number", "thought"]
            }
        }
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this when you need up-to-date information, news, or facts you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    "run_python": {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return the output. Use this for calculations, data processing, or any task that benefits from code execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations. Supports basic operations, trigonometry, logarithms, and more.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sin(3.14)', 'log(100)')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    "analyze_text": {
        "type": "function",
        "function": {
            "name": "analyze_text",
            "description": "Analyze text content - count words, find patterns, extract information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["word_count", "char_count", "find_emails", "find_urls", "find_numbers", "summarize_stats"],
                        "description": "Type of analysis to perform"
                    }
                },
                "required": ["text", "analysis_type"]
            }
        }
    },
    "http_request": {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make HTTP requests to external APIs. Use this to fetch data from web APIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to request"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "HTTP method"
                    }
                },
                "required": ["url"]
            }
        }
    },
    "get_current_time": {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (e.g., 'UTC', 'US/Eastern'). Defaults to UTC."
                    }
                },
                "required": []
            }
        }
    }
}

# Map tool names to their display info
TOOL_INFO = {
    "think_step": {
        "name": "Reasoning Step",
        "description": "Thinking through the problem",
        "icon": "🧠"
    },
    "web_search": {
        "name": "Web Search",
        "description": "Searching the web for current information",
        "icon": "🔍"
    },
    "run_python": {
        "name": "Code Execution",
        "description": "Running Python code",
        "icon": "💻"
    },
    "calculator": {
        "name": "Calculator",
        "description": "Performing calculations",
        "icon": "🔢"
    },
    "analyze_text": {
        "name": "Text Analysis",
        "description": "Analyzing text content",
        "icon": "📄"
    },
    "http_request": {
        "name": "API Request",
        "description": "Fetching data from API",
        "icon": "🌐"
    }
}


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool and return the result."""
    try:
        if tool_name == "think_step":
            return execute_think_step(
                arguments.get("step_number", 1),
                arguments.get("thought", ""),
                arguments.get("next_action", "")
            )
        elif tool_name == "web_search":
            return execute_web_search(arguments.get("query", ""))
        elif tool_name == "run_python":
            return execute_python(arguments.get("code", ""))
        elif tool_name == "calculator":
            return execute_calculator(arguments.get("expression", ""))
        elif tool_name == "analyze_text":
            return execute_analyze_text(
                arguments.get("text", ""),
                arguments.get("analysis_type", "word_count")
            )
        elif tool_name == "http_request":
            return execute_http_request(
                arguments.get("url", ""),
                arguments.get("method", "GET")
            )
        elif tool_name == "get_current_time":
            return execute_get_time(arguments.get("timezone", "UTC"))
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        return {"error": str(e)}


def execute_think_step(step_number: int, thought: str, next_action: str = "") -> dict:
    """Record a reasoning step - this helps the agent think through problems."""
    result = {
        "step": step_number,
        "recorded_thought": thought,
        "status": "Reasoning recorded. Continue with your analysis."
    }
    if next_action:
        result["planned_action"] = next_action
    return result


def execute_web_search(query: str) -> dict:
    """Execute a web search using DuckDuckGo."""
    try:
        import urllib.request
        import urllib.parse

        # Use DuckDuckGo instant answer API
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"

        req = urllib.request.Request(url, headers={'User-Agent': 'AgentForge/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        results = []

        # Get abstract if available
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Result"),
                "snippet": data.get("Abstract"),
                "source": data.get("AbstractSource", "DuckDuckGo"),
                "url": data.get("AbstractURL", "")
            })

        # Get related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:50],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", "")
                })

        if not results:
            # Fallback: return that we searched but found limited results
            return {
                "query": query,
                "results": [{
                    "title": "Search completed",
                    "snippet": f"Searched for '{query}'. For comprehensive results, the user may want to check a search engine directly.",
                    "source": "DuckDuckGo"
                }],
                "note": "Limited results from instant answer API"
            }

        return {
            "query": query,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "query": query,
            "error": f"Search failed: {str(e)}",
            "fallback": "Unable to perform web search. Please try rephrasing the query."
        }


def execute_python(code: str) -> dict:
    """Execute Python code in a sandboxed environment."""
    # Create a restricted globals dict with safe builtins
    safe_builtins = {
        'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
        'bin': bin, 'bool': bool, 'bytes': bytes,
        'chr': chr, 'dict': dict, 'divmod': divmod,
        'enumerate': enumerate, 'filter': filter, 'float': float,
        'format': format, 'frozenset': frozenset, 'hex': hex,
        'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
        'iter': iter, 'len': len, 'list': list, 'map': map,
        'max': max, 'min': min, 'oct': oct, 'ord': ord,
        'pow': pow, 'print': print, 'range': range, 'repr': repr,
        'reversed': reversed, 'round': round, 'set': set,
        'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum,
        'tuple': tuple, 'type': type, 'zip': zip,
        '__import__': None,  # Disable imports for security
    }

    # Allow some safe modules
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Create execution namespace with math functions available
    exec_globals = {
        '__builtins__': safe_builtins,
        'math': math,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
        'exp': math.exp, 'pi': math.pi, 'e': math.e,
        'floor': math.floor, 'ceil': math.ceil,
        'json': json,  # Allow JSON parsing
    }
    exec_locals = {}

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals, exec_locals)

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Get the last expression value if no print output
        result = None
        if not stdout_output:
            # Try to get the last variable assigned
            for var_name in reversed(list(exec_locals.keys())):
                if not var_name.startswith('_'):
                    result = exec_locals[var_name]
                    break

        return {
            "success": True,
            "stdout": stdout_output if stdout_output else None,
            "stderr": stderr_output if stderr_output else None,
            "result": str(result) if result is not None else None,
            "variables": {k: str(v)[:200] for k, v in exec_locals.items() if not k.startswith('_')}
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def execute_calculator(expression: str) -> dict:
    """Safely evaluate a mathematical expression."""
    # Clean the expression
    expression = expression.strip()

    # Define safe math functions
    safe_dict = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'pow': pow,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
        'log2': math.log2, 'exp': math.exp,
        'floor': math.floor, 'ceil': math.ceil,
        'pi': math.pi, 'e': math.e, 'tau': math.tau,
        'degrees': math.degrees, 'radians': math.radians,
        'factorial': math.factorial, 'gcd': math.gcd,
    }

    try:
        # Only allow safe characters
        allowed_chars = set('0123456789+-*/().,:^ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}

        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')

        result = eval(expression, {"__builtins__": {}}, safe_dict)

        return {
            "expression": expression,
            "result": result,
            "formatted": f"{result:,.10g}" if isinstance(result, (int, float)) else str(result)
        }

    except Exception as e:
        return {
            "expression": expression,
            "error": str(e)
        }


def execute_analyze_text(text: str, analysis_type: str) -> dict:
    """Analyze text content."""
    results = {"text_length": len(text)}

    if analysis_type == "word_count":
        words = text.split()
        results["word_count"] = len(words)
        results["unique_words"] = len(set(w.lower() for w in words))
        results["avg_word_length"] = sum(len(w) for w in words) / len(words) if words else 0

    elif analysis_type == "char_count":
        results["total_chars"] = len(text)
        results["chars_no_spaces"] = len(text.replace(" ", ""))
        results["line_count"] = text.count('\n') + 1

    elif analysis_type == "find_emails":
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        results["emails"] = emails
        results["count"] = len(emails)

    elif analysis_type == "find_urls":
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        results["urls"] = urls
        results["count"] = len(urls)

    elif analysis_type == "find_numbers":
        number_pattern = r'-?\d+\.?\d*'
        numbers = re.findall(number_pattern, text)
        results["numbers"] = [float(n) if '.' in n else int(n) for n in numbers]
        results["count"] = len(numbers)
        if results["numbers"]:
            results["sum"] = sum(results["numbers"])
            results["average"] = results["sum"] / len(results["numbers"])

    elif analysis_type == "summarize_stats":
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        results["words"] = len(words)
        results["sentences"] = len([s for s in sentences if s.strip()])
        results["paragraphs"] = len([p for p in paragraphs if p.strip()])
        results["avg_sentence_length"] = len(words) / len(sentences) if sentences else 0

    return results


def execute_http_request(url: str, method: str = "GET") -> dict:
    """Make an HTTP request to an API."""
    import urllib.request
    import urllib.error

    # Security: only allow certain domains/patterns
    allowed_patterns = [
        'api.github.com',
        'api.openweathermap.org',
        'jsonplaceholder.typicode.com',
        'httpbin.org',
        'api.publicapis.org',
        'catfact.ninja',
        'dog.ceo',
        'api.coindesk.com',
        'api.exchangerate-api.com',
    ]

    # Check if URL is from an allowed domain
    is_allowed = any(pattern in url for pattern in allowed_patterns)

    if not is_allowed:
        return {
            "error": "URL not in allowed list for security",
            "allowed_domains": allowed_patterns,
            "tip": "For safety, API calls are limited to known public APIs"
        }

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'AgentForge/1.0', 'Accept': 'application/json'},
            method=method
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')

            # Try to parse as JSON
            try:
                data = json.loads(content)
                return {
                    "status": response.status,
                    "data": data,
                    "url": url
                }
            except json.JSONDecodeError:
                return {
                    "status": response.status,
                    "content": content[:2000],  # Limit response size
                    "url": url
                }

    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}", "url": url}
    except urllib.error.URLError as e:
        return {"error": f"URL Error: {str(e)}", "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}


def execute_get_time(timezone: str = "UTC") -> dict:
    """Get current date and time."""
    now = datetime.utcnow()

    return {
        "utc": now.isoformat(),
        "formatted": now.strftime("%A, %B %d, %Y at %H:%M:%S UTC"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timestamp": now.timestamp()
    }


def get_tools_for_agent(enabled_tools: list) -> list:
    """Get OpenAI tool definitions for an agent's enabled tools."""
    tools = []

    # Always include get_current_time as it's useful and safe
    tools.append(TOOL_DEFINITIONS["get_current_time"])

    for tool_name in enabled_tools:
        if tool_name in TOOL_DEFINITIONS:
            tools.append(TOOL_DEFINITIONS[tool_name])

    return tools
