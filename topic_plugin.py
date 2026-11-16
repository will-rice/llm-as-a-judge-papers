"""LLM-as-a-judge relevance rule.

The repository covers large language models used to evaluate outputs: prompting
an LLM to score or compare responses (G-Eval, the MT-Bench judge), judge models
trained for it (PandaLM, JudgeLM, Prometheus), and studies of how reliable and
biased those judges are. Judges of images, video, audio or speech, and
multimodal judges, are out of scope.

Every accepted paper is compared with core_papers.json, the core papers
listed in docs/research/core-papers.md. A title that names the task
("LLM-as-a-judge", "LLM evaluators", "judge model", a judge bias) only needs to
clear a low similarity floor. A weaker signal must be close to a core paper: a
title that only says "judge" or "evaluator" or names a judge method, or an
abstract that names the task at least twice under a title about evaluation.
That keeps out papers merely evaluated with an LLM judge, or using "judge" in
another sense.
"""

import functools
import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision

LLM = (
    r"\b(llms?|large language models?|language models?|gpt-?4o?|chatgpt|lms?"
    r"|reasoning models?|lrms?)"
)

# Phrases that name the task.
TASK_PATTERN = re.compile(
    r"(llms?|language[- ]models?|agents?)[- ]as[- ](an?[- ])?(judges?|evaluators?|examiners?)"
    r"|(llm|model)[- ]based (evaluators?|judges?|evaluation)|llm[- ]judges?"
    r"|" + LLM + r" (judges?|evaluators?|graders?|raters?|assessors?)\b"
    r"|" + LLM + r" to (automatically )?(evaluate|judge|assess|score|grade)"
    r"|(judge|evaluator|critic) (models?|llms?)|generative (judges?|evaluators?)"
    r"|human-llm agreement|llm response evaluation|reference-free (reasoning )?evaluation"
    r"|(position|verbosity|length|self-preference|self-enhancement|egocentric|order) bias"
)

# Judge methods and benchmarks. Their names collide with other work
# ("SWE-Prometheus"), so they count as a weak signal.
METHOD_NAMES = re.compile(
    r"\b(g-eval|gptscore|pandalm|judgelm|auto-j|prometheus|critiquellm|chateval"
    r"|mt-bench|alpacaeval|judgebench|llmbar|instructscore)\b"
)

ABSTRACT_MENTIONS = 2

# Title phrases that suggest the task but need the similarity gate.
WEAK_TITLE = re.compile(
    r"\bjudg(e|es|ing)\b|\bevaluators?\b|alternative to human evaluations?"
    r"|(nlg|text generation|reasoning|response|automatic|open-ended) evaluation"
    r"|evaluat\w+ (instruction following|alignment)"
)

# Without a task phrase, an abstract-only paper's title must at least be
# about evaluation.
TOPICAL_TITLE = re.compile(
    r"\b(judg|evaluat|assess|grad(e|er|ers|ing)\b|rat(e|es|ing|ings)\b|bias"
    r"|critic|prefer|scor|examin|agree|reliab|metric)"
)

TITLE_EXCLUSION = re.compile(
    r"\b(image|video|audio|speech|vision|visual|diffusion|text-to-(image|video))\b"
    r"|multimodal|\bmllms?\b|\bvlms?\b|\blvlms?\b"
)

CORE_PAPERS = Path(__file__).with_name("core_papers.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Every core paper scores at least 0.61 against the other core papers. Below
# 0.58, weak-signal papers are mostly other work that mentions judges or
# evaluators: decoding, preference optimization, code detection.
SIMILARITY_THRESHOLD = 0.58
# Titles that name the task are almost always judge papers, even far from the
# core set ("Four Ledgers ... LLM-Judge Calibration" scores 0.42). Below this
# floor they only use an LLM judge as a baseline.
TITLE_SIMILARITY_FLOOR = 0.40


def accept_topic(paper: Paper) -> TopicDecision:
    """Apply the LLM-as-a-judge relevance rule to a paper's title and abstract.

    Args:
        paper: Normalized paper to classify.

    Returns:
        Acceptance decision with the rule that decided it.
    """
    decision = keyword_decision(paper)
    if not decision.accepted:
        return decision
    similarity = float(
        (embed([f"{paper.title}. {paper.abstract}"]) @ core_embeddings().T).max()
    )
    threshold = (
        TITLE_SIMILARITY_FLOOR
        if decision.reason == "accepted by title"
        else SIMILARITY_THRESHOLD
    )
    if similarity < threshold:
        return TopicDecision(
            False, f"similarity to core papers {similarity:.2f} below threshold"
        )
    return decision


def keyword_decision(paper: Paper) -> TopicDecision:
    """Apply the keyword part of the rule.

    Args:
        paper: Normalized paper to classify.

    Returns:
        Acceptance decision. "accepted by weak signal" still needs the
        similarity gate.
    """
    title = paper.title.lower()
    text = f"{title} {paper.abstract.lower()}"
    excluded = TITLE_EXCLUSION.search(title)
    if excluded is not None:
        return TopicDecision(False, f"title matched excluded term: {excluded[0]}")
    if TASK_PATTERN.search(title):
        return TopicDecision(True, "accepted by title")
    if WEAK_TITLE.search(title) or METHOD_NAMES.search(title):
        return TopicDecision(True, "accepted by weak signal")
    mentions = len(TASK_PATTERN.findall(text)) + len(METHOD_NAMES.findall(text))
    if mentions < ABSTRACT_MENTIONS:
        return TopicDecision(False, "missing LLM-as-a-judge signal")
    if not TOPICAL_TITLE.search(title):
        return TopicDecision(False, "title not about evaluation")
    return TopicDecision(True, "accepted by weak signal")


@functools.cache
def core_embeddings() -> np.ndarray:
    """Embed the core papers once per process.

    Returns:
        Unit-normalized embeddings, one row per core paper.
    """
    core = json.loads(CORE_PAPERS.read_text(encoding="utf-8"))
    return embed([f"{entry['title']}. {entry['abstract']}" for entry in core])


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts with the shared model.

    fastembed runs the model with ONNX Runtime and returns NumPy arrays; torch
    is only installed with the conversion extra, so the gate stays in NumPy.

    Args:
        texts: Texts to embed.

    Returns:
        Unit-normalized embeddings, one row per text.
    """
    vectors = np.stack(list(embedding_model().embed(texts)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@functools.cache
def embedding_model() -> TextEmbedding:
    """Load the embedding model once per process.

    Returns:
        The sentence embedding model.
    """
    return TextEmbedding(EMBEDDING_MODEL)
