"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 60%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import sys
import uuid

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer performing "
                    "a thorough pull request review.\n\n"
                    "Analyze the provided diff for:\n"
                    "- Security vulnerabilities\n"
                    "- Logic errors and bugs\n"
                    "- Data integrity issues\n"
                    "- Performance problems\n"
                    "- Missing validation and error handling\n"
                    "- Potential regressions\n\n"
                    "Return a structured PRAnalysis containing "
                    "summary, risk_factors, comments, confidence, "
                    "confidence_reasoning and escalation_questions.\n\n"
                    "Confidence represents how certain you are "
                    "that your review is complete and correct. "
                    "Do not confuse low confidence with proof "
                    "that the code is defective.\n\n"
                    f"When confidence is below {ESCALATE_THRESHOLD:.0%}, "
                    "you MUST generate 2 to 4 specific questions "
                    "in escalation_questions.\n"
                    "Each question must:\n"
                    "1. Reference a specific file, function, or code "
                    "section in the diff.\n"
                    "2. Identify missing context or uncertainty.\n"
                    "3. Ask something the author or reviewer can answer.\n"
                    "4. Help improve the accuracy of the final review.\n\n"
                    "Do NOT generate vague questions such as "
                    "'What is the purpose of this PR?'. "
                    "Prefer concrete questions about security, "
                    "intended behavior, data handling or deployment.\n\n"
                    "High-risk changes touching security (e.g. MD5 password hashing, plaintext tokens), authentication, raw SQL strings, or major architectural uncertainties must have confidence below 58% so they can be clarified.\n\n"
                    "When confidence is high and no clarification "
                    "is necessary, return an empty escalation_questions list."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Title: {state['pr_title']}\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(
        f"  [green]✓[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.escalation_questions)} question(s)"
    )
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions."""
    console.print("[yellow]→ escalate[/yellow]")
    a = state["analysis"]
    questions = a.escalation_questions
    if not questions:
        questions = [
            "What is the intent of this PR?",
            "Any migration concerns?",
        ]
    payload = {
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    }
    answers = interrupt(payload)
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Refine the initial review using reviewer answers."""
    console.print("[cyan]→ synthesize[/cyan]")
    initial = state["analysis"]
    answers = state["escalation_answers"] or {}

    # Format questions and answers
    qa_text = "\n\n".join(
        f"Question: {question}\nAnswer: {answer}"
        for question, answer in answers.items()
    )

    llm = get_llm().with_structured_output(PRAnalysis)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior software engineer refining "
                "an existing pull request review.\n\n"
                "You have access to:\n"
                "1. The original pull request diff.\n"
                "2. The initial code review.\n"
                "3. Additional context provided by a human reviewer.\n\n"
                "Your task is to produce a more accurate review.\n\n"
                "Instructions:\n"
                "- Re-evaluate all initial findings.\n"
                "- Use the reviewer answers to clarify uncertainties.\n"
                "- Correct any unsupported assumptions.\n"
                "- Preserve valid security and correctness concerns.\n"
                "- Remove or revise findings that are disproven.\n"
                "- Do not blindly trust reviewer answers if they "
                "contradict the code.\n"
                "- Produce specific, actionable review comments.\n"
                "- Recalculate confidence based on the available "
                "evidence. Do not automatically increase it.\n"
                "- Explain the reasoning for the final confidence.\n\n"
                "Return the final review as a structured PRAnalysis."
            ),
        },
        {
            "role": "user",
            "content": (
                f"PR Title:\n{state['pr_title']}\n\n"
                f"Original Diff:\n{state['pr_diff']}\n\n"
                "Initial Analysis:\n"
                f"{initial.model_dump_json(indent=2)}\n\n"
                "Reviewer Questions and Answers:\n"
                f"{qa_text}\n\n"
                "Generate the refined review."
            ),
        },
    ]

    with console.status("[dim]Refining review with human context...[/dim]"):
        refined = llm.invoke(messages)

    console.print(
        f"  [green]✓[/green] "
        f"confidence: {initial.confidence:.0%} "
        f"→ {refined.confidence:.0%}"
    )
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    # Escalation workflow
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
