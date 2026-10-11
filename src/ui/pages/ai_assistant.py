"""
AI Assistant page for natural-language attendance queries.

Chat UI built on `st.chat_input`, so pressing ENTER submits the message and
the input clears automatically. No JavaScript is used.
"""

import streamlit as st
from typing import List

import config
from src.services.ai_assistant_service import AIAssistantService, ModelUnavailableError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_ai_assistant():
    """Render the AI Assistant chat page."""
    st.title("🤖 AI Assistant")
    st.caption("Ask about students, sessions, and attendance in plain language")

    _init_state()

    if not _ensure_service():
        return

    assistant: AIAssistantService = st.session_state.ai_assistant

    _render_capabilities(assistant)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------
    st.markdown("---")

    if not st.session_state.chat_history:
        st.info(
            "👋 Ask me anything about attendance, e.g. "
            "*Who was late in the CS101 session?* or *Summarize today's attendance.*"
        )
    for message in st.session_state.chat_history:
        _render_message(message["role"], message["content"])

    # ------------------------------------------------------------------
    # Input: ENTER submits, input clears automatically
    # ------------------------------------------------------------------
    prompt = st.chat_input(
        "Ask about attendance...",
        key="ai_assistant_chat_input",
    )

    if prompt:
        _handle_user_message(assistant, prompt)

    st.markdown("---")
    _render_footer_controls(assistant)


def _init_state() -> None:
    """Initialize session state keys (session-scoped, survives reruns)."""
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("ai_assistant", None)


def _ensure_service() -> bool:
    """
    Create the assistant once per session and surface configuration problems.

    Returns:
        True if the assistant is ready to use.
    """
    if st.session_state.ai_assistant is None:
        try:
            st.session_state.ai_assistant = AIAssistantService()
        except Exception as e:
            st.error(f"⚠️ Failed to initialize AI Assistant: {e}")
            logger.error(f"AI Assistant initialization failed: {e}")
            return False

    assistant: AIAssistantService = st.session_state.ai_assistant

    if not assistant.is_configured():
        st.warning(
            f"**AI Assistant is not configured.** Set `OMNIROUTE_API_KEY` in "
            f"`{config.ENV_FILE}` and restart the app."
        )
        return False

    return True


def _render_message(role: str, content: str) -> None:
    """Render a single chat bubble."""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)


def _handle_user_message(assistant: AIAssistantService, prompt: str) -> None:
    """
    Show the user message, run the assistant with a loading state, and show
    the reply. Never raises - errors become visible assistant messages.
    """
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    _render_message("user", prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                response = assistant.ask(prompt)
            except ModelUnavailableError as e:
                response = f"⚠️ The AI model is unavailable: {e}"
                logger.error(f"AI model unavailable: {e}")
            except Exception as e:
                response = "⚠️ Something went wrong while processing that. Please try again."
                logger.error(f"AI assistant error: {type(e).__name__}: {e}")

        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    logger.info(f"AI assistant answered a question ({len(prompt)} chars)")

    # Re-run so the transcript renders from session state on the next pass.
    st.rerun()


def _render_capabilities(assistant: AIAssistantService) -> None:
    """Show example questions and what the system cannot answer."""
    with st.expander("💡 What can I ask?", expanded=False):
        st.markdown("""
Try asking things like:

- "Who am I?"
- "How many students attended the latest session?"
- "Who was late in CS101 Week 1?"
- "What is Ahmed's punctuality percentage?"
- "Show me Ahmed's attendance history."
- "Who has the worst punctuality?"
- "How many sessions were held for CS101?"
- "Give me a summary of today's attendance."
""")

        st.markdown("**Not tracked by this system:**")
        st.markdown(
            "\n".join(f"- {item}" for item in assistant.get_unavailable_capabilities())
        )


def _render_footer_controls(assistant: AIAssistantService) -> None:
    """Clear-chat button, example shortcuts, and optional debug panel."""
    col_clear, col_examples = st.columns([1, 3])

    with col_clear:
        if st.button("🗑️ Clear Chat", width="stretch", key="ai_assistant_clear"):
            st.session_state.chat_history = []
            assistant.reset_conversation()
            st.rerun()

    with col_examples:
        st.markdown("**Try one:**")
        b1, b2, b3 = st.columns(3)
        examples = [
            ("Overall stats", "Show me overall attendance statistics", "ai_ex_1"),
            ("Latest session", "How many students attended the latest session?", "ai_ex_2"),
            ("Worst punctuality", "Who has the worst punctuality?", "ai_ex_3"),
        ]
        for column, (label, prompt, key) in zip((b1, b2, b3), examples):
            with column:
                if st.button(label, width="stretch", key=key):
                    st.session_state._pending_prompt = prompt

    pending = st.session_state.pop("_pending_prompt", None)
    if pending:
        _handle_user_message(assistant, pending)

    if st.checkbox("🔧 Show Debug Info", value=False, key="ai_assistant_debug"):
        st.json(assistant.debug_state())