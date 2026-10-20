from __future__ import annotations

import tempfile
import uuid
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command


st.set_page_config(
    page_title="Atlas · Your research companion",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root { --ink: #172033; --muted: #647087; --violet: #6657e8; --paper: #f7f8fc; }
        .stApp { background: radial-gradient(circle at 83% 4%, #e6e1ff 0, transparent 24rem), #f7f8fc; color: var(--ink); }
        h1, h2, h3, [data-testid="stSidebar"] { font-family: 'Manrope', sans-serif; }
        .block-container { max-width: 1060px; padding-top: 2.4rem; padding-bottom: 7.5rem; }
        [data-testid="stSidebar"] { background: #171b2d; color: #f5f4ff; }
        [data-testid="stSidebar"] * { color: #f5f4ff; }
        [data-testid="stSidebar"] .stButton button { border-color: #7971e9; background: rgba(255,255,255,.08); }
        .github-link { display: flex; justify-content: flex-end; margin: -.35rem 0 -.25rem; }
        .github-link a { color: #bfc5dc !important; font-family: 'DM Mono', monospace; font-size: .72rem; text-decoration: none; }
        .github-link a:hover { color: #ffffff !important; text-decoration: underline; }
        [data-testid="stChatMessage"] { border: 1px solid #e5e7ef; border-radius: 16px; background: #ffffff; color: #172033 !important; }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span { color: #172033 !important; }
        [data-testid="stChatMessage"] code { color: #302989 !important; background: #f0efff !important; }
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"],
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] *,
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] a { color: #000000 !important; opacity: 1 !important; }
        .stChatInputContainer { border: 1px solid #d9ddec; border-radius: 18px; background: #ffffff; box-shadow: 0 8px 24px rgba(31, 38, 70, .08); }
        [data-testid="stChatInput"] textarea,
        .stChatInputContainer textarea { color: #172033 !important; caret-color: #6657e8; background: #ffffff !important; font-family: 'Manrope', sans-serif; }
        [data-testid="stChatInput"] textarea::placeholder,
        .stChatInputContainer textarea::placeholder { color: #758099 !important; opacity: 1; }
        [data-testid="stChatInput"] button { background: #6657e8 !important; color: #ffffff !important; border-radius: 10px; }
        [data-testid="stChatInput"] button svg { fill: #ffffff !important; }
        .thinking { display: inline-flex; align-items: center; gap: .55rem; color: #667085; font-family: 'Manrope', sans-serif; font-size: .88rem; padding: .2rem 0; }
        .thinking-dots { display: inline-flex; gap: .2rem; }
        .thinking-dots i { width: .38rem; height: .38rem; border-radius: 50%; background: #6657e8; animation: atlas-pulse 1.15s infinite ease-in-out; }
        .thinking-dots i:nth-child(2) { animation-delay: .16s; }
        .thinking-dots i:nth-child(3) { animation-delay: .32s; }
        @keyframes atlas-pulse { 0%, 70%, 100% { transform: translateY(0); opacity: .32; } 35% { transform: translateY(-.24rem); opacity: 1; } }
        .tool-status { color: #172033 !important; font-family: 'Manrope', sans-serif; font-size: .84rem; font-weight: 600; padding: .2rem 0 .45rem; }
        .doc-note { color: #788298; font-size: .83rem; }
        .email-preview { border: 1px solid #d9ddec; border-radius: 12px; background: #fbfbfe; padding: .9rem 1.1rem; margin: .4rem 0 1rem; font-family: 'Manrope', sans-serif; color: #172033; }
        .email-meta { font-size: .9rem; padding: .15rem 0; color: #172033; overflow-wrap: anywhere; }
        .email-meta span { display: inline-block; min-width: 4.6rem; color: #647087; font-weight: 600; }
        .email-body { border-top: 1px solid #e5e7ef; margin-top: .6rem; padding-top: .75rem; font-size: .93rem; line-height: 1.6; color: #172033; overflow-wrap: anywhere; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Connecting Atlas to its knowledge tools…")
def load_backend() -> Any:
    """Delay expensive model/database setup until the UI needs it."""
    import backend

    return backend


def new_thread() -> str:
    thread_id = str(uuid.uuid4())
    st.session_state.thread_id = thread_id
    st.session_state.messages = []
    st.session_state.pop("pending_approval", None)
    return thread_id


def display_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def render_email_preview(request: dict[str, Any]) -> None:
    """Show the complete email (recipient, subject and full body) awaiting approval."""
    body = request.get("body") or request.get("question") or ""
    rows = [
        (label, request[key])
        for label, key in (("To", "to"), ("Subject", "subject"))
        if request.get(key)
    ]
    meta = "".join(
        f"<div class='email-meta'><span>{label}</span>{escape(str(value))}</div>"
        for label, value in rows
    )
    # Newlines become <br> so blank lines can't end the HTML block and cut the email short.
    body_html = escape(str(body)).replace("\r\n", "\n").replace("\n", "<br>")
    st.markdown(
        f"<div class='email-preview'>{meta}<div class='email-body'>{body_html}</div></div>",
        unsafe_allow_html=True,
    )
    if not rows:  # older backend payload without separate to/subject fields
        st.caption(request.get("reason", "Atlas is waiting for your confirmation."))


def _read_workflow_result(result: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Return either a final reply or the approval payload from a paused workflow."""
    interrupts = result.get("__interrupt__", ())
    if interrupts:
        value = getattr(interrupts[0], "value", interrupts[0])
        return None, value if isinstance(value, dict) else {"reason": str(value)}

    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return display_content(message.content), None
    return "I completed the request, but the workflow did not return a text response.", None


def run_agent(backend: Any, prompt: str) -> tuple[str | None, dict[str, Any] | None]:
    """Start a LangGraph run and return a reply or an approval request."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = backend.workflow.invoke(
        {"messages": [HumanMessage(content=prompt)]}, config=config
    )
    return _read_workflow_result(result)


TOOL_LABELS = {
    "rag_tool": "Searching your documents…",
    "web_search": "Searching the web…",
    "read_inbox_emails_tool": "Reading your inbox…",
    "send_gmail_tool": "Preparing your email…",
    "yt_video_finder": "Finding YouTube videos…",
    "yt_video_summarizer": "Getting the video transcript…",
}


def _tool_names(value: Any) -> list[str]:
    """Find tool calls in a LangGraph stream update without relying on its shape."""
    names: list[str] = []

    def walk(item: Any) -> None:
        for tool_call in getattr(item, "tool_calls", []) or []:
            name = tool_call.get("name") if isinstance(tool_call, dict) else None
            if name and name not in names:
                names.append(name)
        if isinstance(item, dict):
            for nested in item.values():
                walk(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                walk(nested)

    walk(value)
    return names


def _interrupt_from_update(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or "__interrupt__" not in value:
        return None
    interrupts = value["__interrupt__"]
    if not interrupts:
        return None
    payload = getattr(interrupts[0], "value", interrupts[0])
    return payload if isinstance(payload, dict) else {"reason": str(payload)}


def _answer_from_update(value: Any) -> str | None:
    """Use the final graph update as a fallback when a provider does not token-stream."""
    if not isinstance(value, dict):
        return None
    for nested in value.values():
        messages = nested.get("messages", []) if isinstance(nested, dict) else []
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return display_content(message.content)
    return None


def run_agent_stream(
    backend: Any,
    prompt: str,
    on_token: Any,
    on_tool: Any,
) -> tuple[str | None, dict[str, Any] | None]:
    """Stream response tokens and graph tool activity to the Streamlit UI."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    answer = ""
    fallback_answer: str | None = None
    approval_request: dict[str, Any] | None = None

    for mode, payload in backend.workflow.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            message, metadata = payload
            if metadata.get("langgraph_node") != "chat_node":
                continue
            token = display_content(getattr(message, "content", ""))
            if token:
                answer += token
                on_token(answer)
        elif mode == "updates":
            approval_request = _interrupt_from_update(payload) or approval_request
            for tool_name in _tool_names(payload):
                on_tool(TOOL_LABELS.get(tool_name, f"Using {tool_name}…"))
            fallback_answer = _answer_from_update(payload) or fallback_answer

    if approval_request:
        return None, approval_request
    return answer or fallback_answer or "I completed the request, but no text response was returned.", None


def resume_agent(
    backend: Any, approval: bool
) -> tuple[str | None, dict[str, Any] | None]:
    """Resume the exact workflow that was paused by an approval interrupt."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = backend.workflow.invoke(Command(resume={"approved": approval}), config=config)
    return _read_workflow_result(result)


def ingest_upload(backend: Any, uploaded_file: Any) -> None:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = Path(temp_file.name)
        backend.ingest_document(str(temp_path))
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def get_thread_messages(backend: Any, thread_id: str) -> list[tuple[str, str]]:
    """Restore displayable messages saved in a LangGraph checkpoint."""
    snapshot = backend.workflow.get_state(
        {"configurable": {"thread_id": thread_id}}
    )
    history: list[tuple[str, str]] = []
    for message in snapshot.values.get("messages", []):
        if isinstance(message, HumanMessage):
            history.append(("user", display_content(message.content)))
        elif isinstance(message, AIMessage) and message.content:
            history.append(("assistant", display_content(message.content)))
    return history


def get_conversations(backend: Any) -> list[tuple[str, str]]:
    """Return saved threads with a title based on each first question."""
    try:
        thread_ids = backend.get_all_threads()
    except Exception:
        thread_ids = []

    conversations: list[tuple[str, str]] = []
    for thread_id in thread_ids:
        try:
            history = get_thread_messages(backend, thread_id)
            first_question = next(
                (content for role, content in history if role == "user"), "Untitled chat"
            )
            title = " ".join(first_question.split())[:38] or "Untitled chat"
            conversations.append((thread_id, title))
        except Exception:
            conversations.append((thread_id, "Previous conversation"))

    return conversations


def select_thread(backend: Any, thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.messages = get_thread_messages(backend, thread_id)


_apply_theme()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

try:
    backend = load_backend()
except Exception as error:
    st.error("Atlas could not connect to its backend.")
    st.code(str(error), language=None)
    st.info("Check the environment variables and external database/API connections, then reload the page.")
    st.stop()

with st.sidebar:
    st.markdown(
        "<div class='github-link'><a href='https://github.com/Soumya-techy' target='_blank' rel='noopener noreferrer'>GitHub ↗</a></div>",
        unsafe_allow_html=True,
    )
    st.markdown("## ✦ Atlas")
    st.caption("RESEARCH · EMAIL · WEB")
    if st.button("＋ New chat", use_container_width=True):
        new_thread()
        st.rerun()

    st.divider()
    st.markdown("#### Conversations")
    for thread_id, title in get_conversations(backend):
        active = thread_id == st.session_state.thread_id
        label = f"●  {title}" if active else title
        if st.button(label, key=f"thread-{thread_id}", use_container_width=True):
            select_thread(backend, thread_id)
            st.rerun()

    st.divider()
    st.markdown("#### Connected skills")
    st.caption("PDF retrieval\n\nWeb search\n\nGmail inbox & sending\n\nYouTube discovery & transcripts\n\nLong-term memory")

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

pending = st.session_state.pending_approval
if pending and pending["thread_id"] == st.session_state.thread_id:
    request = pending["request"]
    with st.chat_message("assistant"):
        st.markdown("**Send this email?**")
        render_email_preview(request)
        send_col, cancel_col = st.columns(2)
        with send_col:
            send = st.button("Send email", type="primary", use_container_width=True)
        with cancel_col:
            cancel = st.button("Cancel", use_container_width=True)

        if send or cancel:
            with st.spinner("Finalizing your choice…"):
                try:
                    answer, follow_up = resume_agent(backend, approval=send)
                except Exception as error:
                    answer, follow_up = f"I couldn't complete that request: `{error}`", None

            st.session_state.pending_approval = (
                {"thread_id": st.session_state.thread_id, "request": follow_up}
                if follow_up
                else None
            )
            if answer:
                st.session_state.messages.append(("assistant", answer))
            st.rerun()

submission = st.chat_input(
    "Message Atlas…",
    accept_file=True,
    file_type=["pdf"],
    disabled=bool(pending and pending["thread_id"] == st.session_state.thread_id),
)

if submission:
    prompt = getattr(submission, "text", submission)
    uploads = getattr(submission, "files", [])
    uploaded_names: list[str] = []
    for upload in uploads:
        upload_key = f"indexed:{upload.name}:{upload.size}"
        if not st.session_state.get(upload_key):
            try:
                with st.spinner(f"Indexing {upload.name}…"):
                    ingest_upload(backend, upload)
                st.session_state[upload_key] = True
                uploaded_names.append(upload.name)
            except Exception as error:
                st.error(f"Could not index {upload.name}: {error}")

    if uploaded_names:
        st.toast(f"Added {', '.join(uploaded_names)} to the knowledge base.", icon="📄")

    if not prompt:
        st.stop()

    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking = st.empty()
        tool_status = st.empty()
        response = st.empty()
        thinking.markdown(
            "<div class='thinking'><span class='thinking-dots'><i></i><i></i><i></i></span>Atlas is thinking</div>",
            unsafe_allow_html=True,
        )

        def show_token(text: str) -> None:
            thinking.empty()
            response.markdown(f"{text}▌")

        def show_tool(label: str) -> None:
            thinking.empty()
            tool_status.markdown(
                f"<div class='tool-status'>✦ {escape(label)}</div>",
                unsafe_allow_html=True,
            )

        try:
            answer, approval_request = run_agent_stream(
                backend, prompt, show_token, show_tool
            )
        except Exception as error:
            answer, approval_request = f"I couldn't complete that request: `{error}`", None
        thinking.empty()
        tool_status.empty()
        if approval_request:
            st.session_state.pending_approval = {
                "thread_id": st.session_state.thread_id,
                "request": approval_request,
            }
            st.rerun()
        if answer:
            response.markdown(answer)
    if answer:
        st.session_state.messages.append(("assistant", answer))