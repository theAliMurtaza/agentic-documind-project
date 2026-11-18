"""Streamlit chat UI for DocuMind. Run with:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

from documind import config
from documind.ingest import get_collection, ingest

st.set_page_config(page_title="DocuMind", page_icon=":page_facing_up:")
st.title("DocuMind")
st.caption("Ask questions about your PDFs. Answers are cited, and unknown topics are refused.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Save and index documents"):
        config.DATA_DIR.mkdir(exist_ok=True)
        for upload in uploads or []:
            # Path(...).name strips any folder parts, so a file cannot escape data/.
            (config.DATA_DIR / Path(upload.name).name).write_bytes(upload.getbuffer())
        try:
            with st.spinner("Indexing..."):
                total = ingest()                   # Re-indexes every PDF in data/.
            st.success(f"Indexed {total} chunks.")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    try:
        st.metric("Chunks in index", get_collection().count())
    except Exception:
        st.metric("Chunks in index", 0)

    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

# ------------------------------------------------------------------- chat
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"question": str, "result": dict}


def show_result(result: dict) -> None:
    """Render one agent result: answer, sources and the step log."""
    answer = result["answer"]
    if not answer.found:
        st.warning("Not covered by your documents.")
    st.write(answer.text)
    if answer.citations:
        st.caption("Sources: " + ", ".join(f"{c.source} p.{c.page}" for c in answer.citations))
    label = f"Agent steps ({result['steps']} model calls, {result['seconds']}s)"
    with st.expander(label):
        if not result["tool_calls"]:
            st.write("No tools were used.")
        for call in result["tool_calls"]:
            st.markdown(f"**{call['tool']}**  `{call['args']}`")
            st.code(call["result"], language="text")


for turn in st.session_state.history:              # Redraw earlier turns on every rerun.
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        show_result(turn["result"])

question = st.chat_input("Ask about your documents")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            from documind.agent import ask   # imported here so a missing key shows an error, not a crash

            with st.spinner("Thinking..."):
                result = ask(question)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
        else:
            show_result(result)
            st.session_state.history.append({"question": question, "result": result})