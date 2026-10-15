"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Helper functions ──────────────────────────────────────────────────────
async def get_recent_sessions():
    """Query recent review sessions with calculated worst_risk and event count."""
    query = """
    SELECT thread_id, pr_url,
           MIN(timestamp) AS started,
           MAX(timestamp) AS last_event,
           CASE MAX(
               CASE risk_level
                   WHEN 'high' THEN 3
                   WHEN 'med' THEN 2
                   WHEN 'low' THEN 1
                   ELSE 0
               END
           )
               WHEN 3 THEN 'high'
               WHEN 2 THEN 'med'
               WHEN 1 THEN 'low'
               ELSE 'unknown'
           END AS worst_risk,
           COUNT(*) AS events
      FROM audit_events
     GROUP BY thread_id, pr_url
     ORDER BY last_event DESC
     LIMIT 25
    """
    async with db_conn() as conn:
        async with conn.execute(query) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def load_saved_session(thread_id: str):
    """Restore paused interrupt payload or completed state from checkpointer."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(cfg)
        if not snapshot.values:
            return None, None
        pending_payload = None
        for task in snapshot.tasks:
            for intr in task.interrupts:
                pending_payload = intr.value
                break
            if pending_payload is not None:
                break
        if pending_payload is not None:
            return pending_payload, None
        return None, dict(snapshot.values)


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    sessions = asyncio.run(get_recent_sessions())
    if not sessions:
        st.info("No review sessions yet.")
    else:
        st.dataframe(
            sessions,
            use_container_width=True,
            hide_index=True,
        )
        session_ids = [s["thread_id"] for s in sessions]
        selected_thread = st.selectbox(
            "Select a session",
            options=session_ids,
            format_func=lambda tid: (
                f"{tid[:8]}... — "
                f"{next(s['pr_url'] for s in sessions if s['thread_id'] == tid)}"
            ),
        )
        if st.button("Load selected session"):
            selected = next(
                s for s in sessions if s["thread_id"] == selected_thread
            )
            st.session_state.thread_id = selected["thread_id"]
            st.session_state.pr_url = selected["pr_url"]
            payload, final = asyncio.run(
                load_saved_session(selected["thread_id"])
            )
            st.session_state.interrupt_payload = payload
            st.session_state.final = final
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])
    questions = payload["questions"]

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, question in enumerate(questions):
            answers[question] = st.text_input(
                label=f"Q{i + 1}: {question}",
                key=f"escalation_q_{i}",
            )
        submitted = st.form_submit_button("Submit answers", type="primary")
        if submitted:
            if any(not ans.strip() for ans in answers.values()):
                st.warning("Please answer all escalation questions.")
                return None
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Run or resume the HITL review graph."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg
            )
        else:
            result = await app.ainvoke(
                Command(resume=resume_value), cfg
            )
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    analysis = final.get("analysis")
    if action in (
        "auto_committed",
        "committed",
        "committed_after_escalation",
        "auto_approved",
    ) or action.startswith("auto") or action.startswith("committed"):
        st.success(f"Review completed — {action}")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    elif "failed" in action:
        st.error(f"Review failed — {action}")
    else:
        st.info(f"Final action: {action}")

    if analysis is not None:
        st.metric("Final confidence", f"{analysis.confidence:.0%}")
        st.subheader("Review summary")
        st.write(analysis.summary)
        with st.expander("Review comments"):
            for comment in analysis.comments:
                st.markdown(
                    f"- **[{comment.severity}]** "
                    f"`{comment.file}:{comment.line or '?'}` "
                    f"— {comment.body}"
                )

    if action in (
        "auto_committed",
        "committed",
        "committed_after_escalation",
        "auto_approved",
    ) or action.startswith("auto") or action.startswith("committed"):
        st.link_button("View PR comment on GitHub", st.session_state.pr_url)

    st.caption(f"thread_id = {st.session_state.thread_id}")
    st.code(
        f"uv run python -m audit.replay --thread {st.session_state.thread_id}",
        language="bash",
    )
