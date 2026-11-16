import json
from pathlib import Path

import pytest

from papers_pipeline.config import TopicConfig
from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision, build_topic_gate
from topic_plugin import (
    CORE_PAPERS,
    SIMILARITY_THRESHOLD,
    accept_topic,
    core_embeddings,
    keyword_decision,
)

GATE_PAPERS = json.loads(
    (Path(__file__).parent / "fixtures" / "topic_gate_papers.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize(
    ("title", "abstract", "decision"),
    [
        (
            "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
            "Strong LLM judges like GPT-4 match human preferences.",
            TopicDecision(True, "accepted by title"),
        ),
        (
            "Four Ledgers, Not One Score: Responsible Communication of LLM-Judge"
            " Calibration",
            "How to report calibration.",
            TopicDecision(True, "accepted by title"),
        ),
        (
            "Large Language Models are not Fair Evaluators",
            "The order of candidate responses changes the verdict.",
            TopicDecision(True, "accepted by weak signal"),
        ),
        (
            "SWE-Prometheus: Measuring Engineering Governance Improvements",
            "Repository governance metrics.",
            TopicDecision(True, "accepted by weak signal"),
        ),
        (
            "Branch-Solve-Merge Improves Large Language Model Evaluation and"
            " Generation",
            "We apply our method to LLM response evaluation and improve"
            " human-LLM agreement.",
            TopicDecision(True, "accepted by weak signal"),
        ),
        (
            "Evalverse: Unified and Accessible Library for Large Language Model"
            " Evaluation",
            "A library for benchmarking LLMs.",
            TopicDecision(False, "missing LLM-as-a-judge signal"),
        ),
        (
            "On the Adversarial Robustness of Multimodal LLM Judges",
            "Attacks on MLLM-as-a-judge.",
            TopicDecision(False, "title matched excluded term: multimodal"),
        ),
        (
            "Amphista: Bi-directional Multi-head Decoding for Accelerating LLM Inference",
            "We use an LLM judge and report LLM-as-a-judge win rates.",
            TopicDecision(False, "title not about evaluation"),
        ),
    ],
)
def test_keyword_decision_applies_llm_as_a_judge_rule(
    paper: Paper, title: str, abstract: str, decision: TopicDecision
) -> None:
    candidate = paper.model_copy(update={"title": title, "abstract": abstract})

    assert keyword_decision(candidate) == decision


def test_every_core_paper_passes_the_keyword_rule(paper: Paper) -> None:
    core = json.loads(CORE_PAPERS.read_text(encoding="utf-8"))

    rejected = [
        entry["name"]
        for entry in core
        if not keyword_decision(
            paper.model_copy(
                update={"title": entry["title"], "abstract": entry["abstract"]}
            )
        ).accepted
    ]

    assert rejected == []


def test_every_core_paper_clears_the_threshold_against_the_others() -> None:
    similarity = core_embeddings() @ core_embeddings().T
    similarity[range(len(similarity)), range(len(similarity))] = 0

    assert similarity.max(axis=1).min() >= SIMILARITY_THRESHOLD


@pytest.mark.parametrize(
    ("fixture", "decision"),
    [
        ("on_topic", TopicDecision(True, "accepted by weak signal")),
        (
            "off_topic",
            TopicDecision(False, "similarity to core papers 0.38 below threshold"),
        ),
    ],
)
def test_accept_topic_gates_on_similarity_to_core_papers(
    paper: Paper, fixture: str, decision: TopicDecision
) -> None:
    entry = GATE_PAPERS[fixture]
    candidate = paper.model_copy(
        update={"title": entry["title"], "abstract": entry["abstract"]}
    )

    assert keyword_decision(candidate).accepted
    assert accept_topic(candidate) == decision


def test_papers_yml_plugin_reference_builds_repository_gate(paper: Paper) -> None:
    gate = build_topic_gate(TopicConfig(plugin="topic_plugin:accept_topic"))
    candidate = paper.model_copy(
        update={
            "title": "A Survey on LLM-as-a-Judge",
            "abstract": "",
        }
    )

    assert gate(candidate) == TopicDecision(True, "accepted by title")
