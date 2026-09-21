
import os
import json
import requests
import yfinance as yf
from openai import OpenAI

# ── Tool functions ────────────────────────────────────────────────────────────

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=%C+%t"
    try:
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text.strip()}."
        return f"Could not retrieve weather for {city} (status {response.status_code})."
    except Exception as e:
        return f"Weather lookup failed: {e}"


def run_command(cmd: str) -> str:
    import subprocess
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip() or result.stderr.strip() or "(no output)"
        return output
    except subprocess.TimeoutExpired:
        return "Command timed out after 15 seconds."
    except Exception as e:
        return f"Command execution failed: {e}"


def get_stock_price(symbol: str) -> str:
    try:
        stock = yf.Ticker(symbol.upper())
        data = stock.history(period="1d")
        if data.empty:
            return f"No stock data found for '{symbol}'."
        price = data["Close"].iloc[-1]
        return f"Current price of {symbol.upper()} is ${price:.2f}"
    except Exception as e:
        return f"Stock lookup error: {e}"


AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "run_command": run_command,
    "get_stock_price": get_stock_price,
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a helpful AI Assistant specialized in resolving user queries.
You work in start → plan → action → observe → output mode.

Rules:
- Follow the Output JSON Format strictly.
- Always perform ONE step at a time and wait for input.
- Carefully analyse the user query.
- Never emit multiple steps in a single response.

Output JSON Format:
{
    "step": "plan" | "action" | "observe" | "output",
    "content": "string (for plan/output steps)",
    "function": "tool name (only for action step)",
    "input": "tool input (only for action step)"
}

Available Tools:
- "get_weather": Takes a city name, returns current weather.
- "run_command": Takes a Linux shell command string, executes it, returns output.
- "get_stock_price": Takes a stock ticker symbol, returns current price.

Example flow for "What is the weather in Tokyo?":
  {"step":"plan","content":"User wants weather data for Tokyo."}
  {"step":"plan","content":"I will use the get_weather tool."}
  {"step":"action","function":"get_weather","input":"Tokyo"}
  {"step":"observe","output":"Sunny +24°C"}
  {"step":"output","content":"The weather in Tokyo is sunny at 24°C."}
"""

# ── Agent runner (yields step dicts for streaming into UI) ────────────────────

def run_agent(query: str, model: str = "qwen2.5-coder:3b", base_url: str = "http://localhost:11434/v1/"):
    """
    Generator that yields step dicts:
      {"step": "plan"|"action"|"observe"|"output", ...}
    Raises ConnectionError if Ollama is unreachable.
    """
    client = OpenAI(base_url=base_url, api_key="ollama")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    max_iterations = 20
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        raw = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw})

        try:
            step = json.loads(raw)
        except json.JSONDecodeError:
            yield {"step": "error", "content": f"Model returned invalid JSON: {raw}"}
            return

        yield step

        if step.get("step") == "plan":
            continue

        if step.get("step") == "action":
            tool_name = step.get("function", "")
            tool_input = step.get("input", "")
            fn = AVAILABLE_TOOLS.get(tool_name)
            if fn:
                try:
                    observation = fn(tool_input)
                except Exception as e:
                    observation = f"Tool error: {e}"
            else:
                observation = f"Unknown tool: {tool_name}"

            observe_msg = json.dumps({"step": "observe", "output": observation})
            messages.append({"role": "user", "content": observe_msg})
            yield {"step": "observe", "output": observation}
            continue

        if step.get("step") == "output":
            return

        # unexpected step — stop
        yield {"step": "error", "content": f"Unexpected step value: {step.get('step')}"}
        return
