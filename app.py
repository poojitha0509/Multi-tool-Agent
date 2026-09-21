"""
AI Agent – Streamlit frontend
Run with:  streamlit run app.py
"""

import streamlit as st
import time
from agent_backend import run_agent, AVAILABLE_TOOLS

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #c9d1e0 !important;
}

/* ── Main background ── */
.main .block-container {
    padding: 2rem 2rem 4rem;
    max-width: 900px;
}

/* ── Chat bubbles ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 1rem 0;
}
.msg-user .bubble {
    background: #2563eb;
    color: #fff;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25);
}

.msg-agent {
    display: flex;
    justify-content: flex-start;
    margin: 0.3rem 0;
}
.msg-agent .bubble {
    background: #1a1d2e;
    color: #e2e8f0;
    padding: 0.65rem 1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.55;
    border: 1px solid #2a2f45;
}

/* Step badges */
.badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 2px 8px;
    border-radius: 999px;
    margin-right: 6px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    vertical-align: middle;
}
.badge-plan    { background:#1e3a5f; color:#60a5fa; }
.badge-action  { background:#1a3320; color:#4ade80; }
.badge-observe { background:#2d1f3d; color:#c084fc; }
.badge-output  { background:#1a2d1a; color:#86efac; }
.badge-error   { background:#3d1515; color:#f87171; }

/* Tool cards in sidebar */
.tool-card {
    background: #1a1d2e;
    border: 1px solid #2a2f45;
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
}
.tool-card .tool-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #60a5fa !important;
    font-weight: 600;
}
.tool-card .tool-desc {
    font-size: 0.78rem;
    color: #8892a4 !important;
    margin-top: 2px;
}

/* Input area */
.stTextInput input {
    background: #1a1d2e !important;
    border: 1px solid #2a2f45 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #2563eb !important;
}

/* Header */
.agent-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.agent-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.agent-header p {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 4px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #4a5568;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state h3 { color: #64748b; font-weight: 500; margin: 0 0 0.5rem; }
.empty-state p { font-size: 0.85rem; }

/* Divider */
hr.divider {
    border: none;
    border-top: 1px solid #1e2130;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, steps}
if "running" not in st.session_state:
    st.session_state.running = False

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    ollama_url = st.text_input(
        "Ollama base URL",
        value="http://localhost:11434/v1/",
        help="The URL where Ollama is running.",
    )
    model_name = st.text_input(
        "Model",
        value="qwen2.5-coder:3b",
        help="Any model available in your Ollama instance.",
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("## 🛠️ Available Tools")

    tool_meta = {
        "get_weather":    ("🌤️", "Current weather for any city"),
        "run_command":    ("💻", "Execute Linux shell commands"),
        "get_stock_price":("📈", "Live stock price by ticker"),
    }
    for name, (icon, desc) in tool_meta.items():
        st.markdown(f"""
        <div class='tool-card'>
            <div class='tool-name'>{icon} {name}</div>
            <div class='tool-desc'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Powered by Ollama · Local LLM")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class='agent-header'>
    <h1>🤖 AI Agent</h1>
    <p>Plan → Act → Observe · Powered by local LLM via Ollama</p>
</div>
""", unsafe_allow_html=True)

# ── Render helpers ────────────────────────────────────────────────────────────

STEP_LABELS = {
    "plan":    ("badge-plan",    "🧠 Plan"),
    "action":  ("badge-action",  "🛠️ Action"),
    "observe": ("badge-observe", "👁️ Observe"),
    "output":  ("badge-output",  "✅ Output"),
    "error":   ("badge-error",   "❌ Error"),
}

def render_step(step: dict):
    kind = step.get("step", "error")
    cls, label = STEP_LABELS.get(kind, ("badge-error", kind))

    if kind == "plan":
        text = step.get("content", "")
    elif kind == "action":
        text = f"<code style='font-family:JetBrains Mono,monospace;font-size:0.82rem'>{step.get('function')}({step.get('input')})</code>"
    elif kind == "observe":
        text = f"<pre style='margin:4px 0 0;font-size:0.8rem;white-space:pre-wrap'>{step.get('output','')}</pre>"
    elif kind == "output":
        text = step.get("content", "")
    else:
        text = step.get("content", str(step))

    st.markdown(f"""
    <div class='msg-agent'>
        <div class='bubble'>
            <span class='badge {cls}'>{label}</span>
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_history():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='msg-user'>
                <div class='bubble'>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for step in msg.get("steps", []):
                render_step(step)

# ── Chat area ─────────────────────────────────────────────────────────────────

chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class='empty-state'>
            <div class='icon'>💬</div>
            <h3>Start a conversation</h3>
            <p>Try: <em>"What's the weather in Mumbai?"</em><br>
               or  <em>"Stock price of TSLA"</em><br>
               or  <em>"List files in /tmp"</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        render_history()

# ── Input ─────────────────────────────────────────────────────────────────────

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([8, 1])
    user_input = cols[0].text_input(
        "Message",
        placeholder="Ask me anything…",
        label_visibility="collapsed",
    )
    submitted = cols[1].form_submit_button("Send", use_container_width=True)

# ── Agent execution ───────────────────────────────────────────────────────────

if submitted and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    # Collect all steps, streaming into a live placeholder
    steps_collected = []

    with st.spinner("Agent thinking…"):
        live = st.empty()

        def redraw_live():
            html_parts = []
            for s in steps_collected:
                kind = s.get("step", "error")
                cls, label = STEP_LABELS.get(kind, ("badge-error", kind))
                if kind == "plan":
                    text = s.get("content", "")
                elif kind == "action":
                    text = f"<code style='font-family:JetBrains Mono,monospace;font-size:0.82rem'>{s.get('function')}({s.get('input')})</code>"
                elif kind == "observe":
                    text = f"<pre style='margin:4px 0 0;font-size:0.8rem;white-space:pre-wrap'>{s.get('output','')}</pre>"
                elif kind == "output":
                    text = s.get("content", "")
                else:
                    text = s.get("content", str(s))
                html_parts.append(f"""
                <div class='msg-agent'>
                    <div class='bubble'>
                        <span class='badge {cls}'>{label}</span>
                        {text}
                    </div>
                </div>""")
            live.markdown("".join(html_parts), unsafe_allow_html=True)

        try:
            for step in run_agent(query, model=model_name, base_url=ollama_url):
                steps_collected.append(step)
                redraw_live()
                time.sleep(0.05)   # tiny pause so the UI repaints
        except Exception as e:
            steps_collected.append({"step": "error", "content": str(e)})
            redraw_live()

    # Persist to history and rerun to render properly
    st.session_state.messages.append({"role": "assistant", "steps": steps_collected})
    st.rerun()
