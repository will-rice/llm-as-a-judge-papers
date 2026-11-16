---
identifier: arxiv:2609.06367v1
title: "Robust Conformal Consensus: Multi-Agent LLM-as-a-Judge Interval Evaluation with Conformal Prediction"
authors:
  - Lihui Liu
published: "2026-09-06T03:55:09+00:00"
url: https://arxiv.org/abs/2609.06367v1
source: arxiv
doi: null
arxiv_id: 2609.06367v1
categories:
  - cs.AI
  - cs.LG
---

# Robust Conformal Consensus: Multi-Agent LLM-as-a-Judge Interval Evaluation with Conformal Prediction

Lihui Liu Affiliation: Wayne State University Affiliation: Detroit,
Michigan, USA Email: [hw6926@wayne.edu](mailto:)

###### Abstract

LLM-as-a-Judge has emerged as a promising paradigm for evaluating
natural language generation. However, the uncertainty associated with
such evaluations remains largely unexplored, which limits their
reliability in real-world applications. Although conformal prediction
offers a principled framework for uncertainty quantification, existing
approaches typically apply it to a single LLM judge, overlooking the
variability introduced by using different LLM evaluators. In this work,
we propose a robust uncertainty estimation framework for multi-agent
LLM-as-a-Judge evaluation. Our approach constructs conformal prediction
intervals for LLM-based scores from multiple LLMs. By considering
intervals from different LLM judges, we obtain more stable and reliable
uncertainty estimates. Extensive experiments demonstrate that our method
produces valid prediction intervals with coverage guarantees, and that
interval-based aggregation across multiple judges leads to more stable
evaluation outcomes.

## 1 Introduction

Large language models (LLMs) are increasingly being adopted as automated
evaluators for natural language generation (NLG) tasks, a setting
commonly known as LLM-as-a-Judge. Prior studies show that LLM-based
evaluators correlate well with human assessments and achieve competitive
performance on established evaluation metrics such as ROUGE Lin (2004),
BLEU Papineni et al. (2002), and BERTScore Zhang et al. (2020). In
addition to metric-level agreement, LLM judges provide substantial
advantages in terms of flexibility and scalability: they can be readily
adapted to diverse evaluation objectives and offer a cost-effective
alternative to human annotation. As a result, LLM-as-a-Judge has been
successfully applied across a broad range of domains, including
healthcare Bedi et al. (2025), question answering Zheng and Chiang
(2023), cybersecurity Yuan and He (2024), and trafficking
detection Barbosa et al. (2025).

Despite their advantages, evaluations based on a single LLM judge can be
unreliable. LLM outputs are inherently random, so a single score may be
biased or unstable Gu et al. (2025). This unreliability is especially
concerning in high-stakes areas such as healthcare and finance  Bedi
et al. (2025). Although techniques like prompt engineering or
fine-tuning can reduce uncertainty, LLM judges may still produce
inconsistent or overly confident evaluations. Conformal prediction
 Shafer and Vovk (2008a) offers a clear way to address this issue. It
produces prediction intervals that indicate a range where the true score
is likely to fall, with statistical guarantees. Because it works without
assumptions about the model or data and requires only a small
calibration set, conformal prediction is well suited for measuring
uncertainty in LLM-based evaluations.

However, existing conformal prediction methods for LLM-based evaluation
are predominantly designed for a single LLM judge and are applied to the
scores produced by one model in isolation. In practice, different LLMs
often assign noticeably different scores to the same input due to
variations in training data, architectures, alignment strategies, and
inherent inductive biases. As a consequence, uncertainty estimates
derived from a single LLM can be highly model-dependent and may fail to
reflect the true variability of the evaluation process. This limitation
can lead to misleadingly narrow prediction intervals, overconfident
assessments, and poor robustness when the choice of LLM judge changes.
To obtain more reliable and stable uncertainty estimates, it is
therefore necessary to explicitly account for inter-LLM variability by
incorporating scores from multiple LLM judges. Leveraging multiple LLMs
as independent evaluators enables the construction of uncertainty
estimates that better capture disagreement across judges, resulting in
more robust prediction intervals and more trustworthy LLM-as-a-Judge
evaluations.

In this paper, we introduce a robust interval adaptation framework that
can be integrated with existing conformal prediction methods. Instead of
relying on a single LLM judge, our approach explicitly incorporates
multiple LLMs and recalibrates prediction intervals to account for
inter-LLM variability. We apply the proposed framework to nine conformal
prediction techniques to quantify uncertainty in rating-based
LLM-as-a-Judge evaluations, where each method produces a prediction
interval over LLM-generated scores. For each approach, we assess both
efficiency, measured by the average interval width, and coverage,
defined as the probability that the true rating lies within the
predicted interval. Our analysis shows that the quality of prediction
intervals depends on the LLM used and the size of the calibration set.
Using multiple LLMs provides more stable and reliable intervals than
relying on a single model. These findings emphasize the value of
multi-LLM, uncertainty-aware evaluation for more robust assessments.

In summary, our contributions are:

- •
  We propose a multi-LLM, robust framework that quantifies uncertainty
  in LLM-as-a-judge evaluations using conformal prediction for
  rating-based tasks.
- •
  We run experiments with multiple LLMs, showing that our method yields
  more stable uncertainty estimates than using a single LLM.

## 2 Preliminaries

##### LLM-as-a-Judge.

Large language models (LLMs) are increasingly used as automatic
evaluators for natural language generation, a paradigm commonly called
LLM-as-a-judge. In rating-based evaluations, an LLM assigns a numerical
score $`y_{0}`$ to a candidate text $`x`$ according to a predefined
scale, such as a Likert scale  elangovan2024beyond. Formally, given a
prompt $`p`$, an LLM judge $`M`$ produces

```math
M(p,x)=(z,y_{0}), \tag{1}
```

where $`z`$ represents the token-level logits and $`y_{0}`$ is the final
score. To obtain $`y_{0}`$, we focus on the logits corresponding to the
tokens that encode the discrete rating options (e.g., 1–5). In addition
to scalar scores, LLMs can handle more complex evaluation setups, such
as pairwise comparisons or ranking tasks, by scoring multiple candidate
outputs and deriving relative judgments based on these scores. This
flexibility makes LLMs versatile evaluators across diverse NLG
scenarios.

##### Conformal Prediction.

Conformal prediction is a flexible, model-agnostic technique for
measuring uncertainty in predictions  Shafer and Vovk (2008a). Instead
of giving a single point estimate, it produces a prediction interval
that is statistically guaranteed to contain the true value with a
pre-specified probability. Two key features make it appealing for LLM
evaluation: it works post-hoc without modifying the model, and it does
not rely on assumptions about the data distribution.

In this paper, we use split conformal prediction, which estimates
uncertainty using a separate calibration set. For each calibration
example, we compute a non-conformity score that quantifies how far the
model’s prediction $`\hat{y}`$ is from the true value $`y`$:

```math
s(z,y)=|\hat{y}-y|, \tag{2}
```

where $`\hat{y}=f(z)`$ is the predicted score derived from model logits
$`z`$.

Given a desired error rate $`\alpha`$, we take the
$`\lceil(n+1)(1-\alpha)\rceil`$-th largest non-conformity score
$`\hat{q}`$ from the calibration set to define the prediction interval
for a new test input $`z_{\text{test}}`$:

```math
C(z_{\text{test}},\hat{y}_{\text{test}})=[\hat{y}_{\text{test}}-\hat{q},\hat{y}_{\text{test}}+\hat{q}]. \tag{3}
```

This interval satisfies a formal coverage guarantee, meaning that the
true score $`y_{\text{test}}`$ will lie inside it at least $`1-\alpha`$
of the time, assuming calibration and test points are exchangeable
 Shafer and Vovk (2008a):

```math
1-\alpha\leq\mathbb{P}\big(y_{\text{test}}\in C(z_{\text{test}},\hat{y}_{\text{test}})\big)\leq 1-\alpha+\frac{1}{n+1}. \tag{4}
```

By constructing such intervals, conformal prediction allows us to
quantify the uncertainty of LLM-generated ratings in a principled and
interpretable way.

## 3 Methodology

Figure 1: Overview of uncertainty quantification in rating-based
multi-agent LLM-as-a-Judge evaluation. We apply conformal prediction to
multiple LLM judges to construct prediction intervals for their ratings.
These intervals are then aggregated, and the width of the resulting
interval is used as a measure of evaluation uncertainty.

Existing conformal prediction approaches for LLM-based evaluation
typically apply uncertainty quantification to a single judging agent,
treating the LLM’s output as fixed given one prompt and one response.
This setting overlooks the well-documented sensitivity of LLM judgments
to prompt wording, instruction style, and sampling randomness,
effectively collapsing diverse judging behaviors into a single agent
view. For example, when the same generated text is evaluated by two
judging agents instantiated through semantically equivalent Likert-scale
prompts, such as “Rate the overall quality on a scale from 1 to 5” or
“How good is this response? (1 = very poor, 5 = excellent)”, different
LLMs may assign different ratings (e.g., 3 vs. 4), despite evaluating
identical content. Applying conformal prediction to only one such agent
produces a prediction interval that reflects agent-specific variability
rather than intrinsic evaluation uncertainty, leading to unstable and
potentially misleading uncertainty estimates.

To solve this problem, in this paper, we study uncertainty
quantification for multi-agent LLM-as-a-judge in discrete rating-based
evaluations (e.g., Likert scales) using conformal prediction. Instead of
relying on a single judging agent, we instantiate multiple agents
through semantically equivalent prompts. Each agent independently
produces a judgment, which is then transformed into a conformal
prediction interval. These agent-level intervals are subsequently
aggregated to yield a final, uncertainty-aware evaluation that captures
model uncertainty. An overview of the workflow is shown in Figure  1.

### 3.1 LLM Rating Extraction and Conformal Prediction

We focus on evaluation tasks where ratings are discrete, such as Likert
scales (1–5). For each input text, we collect responses from $`m`$
different LLM judges, where each judge independently provides a
single-turn rating. This setup captures variability across models, which
is important since different LLMs may assign different scores to the
same text.

From each LLM response, we extract the token logits corresponding to
candidate rating labels. In practice, outputs may not strictly follow a
fixed format, so we adopt a simple rule-based strategy: identify the
rating token at its most frequent position across responses from the
same LLM. Once the position is located, we extract the log probabilities
for all possible labels and merge tokens with equivalent meanings (e.g.,
“two” and “2”) to maintain semantic consistency.

This process produces, for each LLM $`j`$, a $`K`$-dimensional feature
vector:

```math
\mathbf{z}^{(j)}\in\mathbb{R}^{K},
```

representing logits for each rating label. Combined with the observed
rating $`y^{(j)}`$, each pair $`(\mathbf{z}^{(j)},y^{(j)})`$ is used as
input to conformal prediction. We assume that pairs from different LLMs
are exchangeable, allowing uncertainty to be estimated across the
ensemble of judges.

##### Constructing Prediction Intervals.

For each LLM, we generate a prediction interval for its rating.
Intervals are preferred over sets because they naturally respect the
ordinal structure of Likert ratings. To accommodate different modeling
strategies, we apply a variety of conformal methods.

For regression-based approaches, where logits often exhibit
heteroscedasticity, we use Conformalized Quantile Regression (CQR)
 Romano et al. (2019) and its variants, including asymmetric CQR, CHR
 Sesia and Romano (2021), LVD  Lin et al. (2021), Boosted CQR Xie et al.
(2024), Boosted LCP Xie et al. (2024), and R2CCP Guha et al. (2024). For
ordinal classification, ratings are treated as ordered categories, and
we apply ordinal conformal methods such as Ordinal APS Lu et al. (2022)
and Ordinal Risk Control Xu et al. (2023), using softmax probabilities
as class likelihoods. Raw logits are retained for regression-based
methods. Each LLM $`j`$ thus produces a conformal prediction interval
$`\mathcal{I}^{(j)}_{\alpha}`$ with coverage $`1-\alpha`$.

##### Boundary Adjustment.

In rating-based evaluations, the labels are discrete numbers (for
example, 1 to 5), but the prediction intervals produced by conformal
prediction are often continuous values. Continuous boundaries can be
confusing or hard to interpret, following  Sheng et al. (2025), we
convert the interval $`C(\mathbf{z}_{\text{test}})=[l,u]`$ into a range
of valid integer ratings:

```math
[l,u]\;\longrightarrow\;[l^{\prime},u^{\prime}],\quad l^{\prime}=\lceil l\rceil,\,u^{\prime}=\lfloor u\rfloor.
```

This ensures that the interval only contains valid ratings and removes
parts that do not correspond to any possible label.

To make the intervals more robust and reduce the risk of excluding the
true rating, we can optionally expand the interval to include
neighboring labels. For instance, if the original continuous interval is
$`[2.2,3.9]`$, it would initially cover only label 3, but expanding it
to $`[2,4]`$ gives a safer range. This adjustment preserves the order of
the ratings and provides a more reliable measure of uncertainty.

### 3.2 Aggregating Intervals Across Multiple Judges

After obtaining the individual conformal intervals from each LLM judge,
we need a way to combine them into a single, unified uncertainty
estimate. Each LLM can be treated as an independent “expert”, producing
a calibrated prediction interval

```math
\mathcal{I}^{(m)}_{\alpha}=[L_{m},U_{m}]
```

with nominal coverage $`1-\alpha`$. A straightforward approach is to
take the union or intersection of these intervals. However, these simple
strategies have drawbacks: taking the union often results in overly wide
intervals that are too conservative, while taking the intersection can
produce intervals that are too narrow, risking missed true
ratings—particularly when some LLMs are overconfident or generate
occasional outlier predictions.

To overcome these issues, we propose a Weighted Conformal Consensus
(WCC) method. WCC assigns weights to each LLM judge based on its past
reliability and calibration performance. More trustworthy agents have a
larger influence on the merged interval, while less reliable agents
contribute less. Using this weighted approach, we construct a consensus
interval that balances coverage and efficiency, producing a more robust
and informative uncertainty estimate. This multi-LLM aggregation allows
users to capture both the agreement and variability among different
models, leading to more reliable and interpretable evaluation of text
quality.

#### 3.2.1 Weighted Consensus Aggregation

Instead of relying solely on the median, which ignores subtle
differences in agent reliability, we first calculate a confidence weight
$`w_{m}`$ for each LLM agent $`m`$ based on its historical concordance
with the collective. This weight quantifies the agent’s trustworthiness
relative to its peers.

The preliminary consensus interval
$`C_{\text{weighted}}=[L_{\text{weighted}},U_{\text{weighted}}]`$ is
then derived via a weighted average of the individual interval
endpoints:

```math
\hat{L}_{\text{weighted}}=\frac{\sum_{m=1}^{M}w_{m}\cdot\hat{L}_{m}}{\sum w_{m}}
```

```math
\hat{U}_{\text{weighted}}=\frac{\sum_{m=1}^{M}w_{m}\cdot\hat{U}_{m}}{\sum w_{m}}
```

This aggregation step is crucial: by down-weighting agents that
frequently produce outlier or overconfident intervals, we mitigate the
negative impact of "hallucinating" agents and enhance the efficiency
(narrowness) of the consensus.

### 3.3 Meta-Calibration for Coverage Guarantee

While the weighted average provides an improved, more compact
preliminary interval, it inherently lacks the formal statistical
coverage guarantee. To restore this validity, we introduce a
meta-calibration step.

On a small, independent meta-calibration set
$`\mathcal{D}_{\text{meta}}`$, we compute meta-nonconformity scores
$`S^{\prime}_{\text{meta},i}`$ that measure how far the true label
$`Y_{i}`$ deviates from our weighted consensus interval
$`[\hat{L}_{\text{weighted},i},\hat{U}_{\text{weighted},i}]`$:

```math
S^{\prime}_{\text{meta},i}=\max\big(\hat{L}_{\text{weighted},i}-Y_{i},\;Y_{i}-\hat{U}_{\text{weighted},i}\big).
```

Finally, we compute the $`(1-\alpha)`$ empirical quantile of the
meta-scores, denoted $`Q^{\prime}_{\text{meta}}`$, and construct the
final merged interval:

```math
C_{\text{final}}=[\hat{L}_{\text{weighted}}-Q^{\prime}_{\text{meta}},\;\hat{U}_{\text{weighted}}+Q^{\prime}_{\text{meta}}].
```

### 3.4 Coverage Guarantee for WCC

In this subsection, we provide a theoretical analysis showing that our
proposed method satisfies the desired coverage guarantee, ensuring that
the constructed prediction intervals contain the true rating with
probability at least $`1-\alpha`$.

Let $`\mathcal{Z}=\mathcal{X}\times\mathcal{Y}`$ denote the data space,
where $`\mathcal{X}`$ is the input space and $`\mathcal{Y}`$ the label
space. Let

```math
D_{\mathrm{meta}}=\{(X_{i},Y_{i})\}_{i=1}^{n}
```

be a held-out _meta-calibration_ set used exclusively to compute
meta-level nonconformity scores for calibrating the aggregated LLM
consensus interval. Let $`(X_{n+1},Y_{n+1})`$ denote a test point.

###### Theorem 1 (Marginal Coverage Guarantee of WCC).

Suppose $`M`$ LLMs produce prediction intervals

```math
\hat{C}_{m}(X)=[\hat{L}_{m}(X),\hat{U}_{m}(X)],\quad m=1,\dots,M,
```

for an input $`X`$, with fixed, pre-computed weights
$`\{w_{m}\}_{m=1}^{M}`$ based on historical concordance. Define the
weighted consensus interval as

```math
[\hat{L}_{\mathrm{w}}(X),\hat{U}_{\mathrm{w}}(X)]=\mathcal{A}_{\mathrm{w}}(\hat{C}_{1},\dots,\hat{C}_{M}),
```

where

```math
\hat{L}_{\mathrm{w}}(X)=\frac{\sum_{m=1}^{M}w_{m}\hat{L}_{m}(X)}{\sum_{m=1}^{M}w_{m}},
```

```math
\hat{U}_{\mathrm{w}}(X)=\frac{\sum_{m=1}^{M}w_{m}\hat{U}_{m}(X)}{\sum_{m=1}^{M}w_{m}}.
```

Let $`Q^{\prime}_{\mathrm{meta}}`$ be the $`(1-\alpha)`$ empirical
quantile of the meta-nonconformity scores computed on
$`D_{\mathrm{meta}}`$. Then the final interval

```math
C_{\mathrm{final}}(X_{n+1})=\begin{aligned} \Big[&\hat{L}_{\mathrm{w}}(X_{n+1})-Q^{\prime}_{\mathrm{meta}},\\
&\hat{U}_{\mathrm{w}}(X_{n+1})+Q^{\prime}_{\mathrm{meta}}\Big]\end{aligned}
```

satisfies the marginal coverage guarantee

```math
\mathbb{P}\big(Y_{n+1}\in C_{\mathrm{final}}(X_{n+1})\big)\geq 1-\alpha.
```

###### Proof.

The proof follows the standard split conformal prediction framework,
treating the weighted aggregation operator $`\mathcal{A}_{\mathrm{w}}`$
as a fixed and deterministic predictor.

##### Meta-nonconformity score.

Define the meta-level nonconformity score

```math
S^{\prime}(X,Y)=\max\!\left(\hat{L}_{\mathrm{w}}(X)-Y,\;Y-\hat{U}_{\mathrm{w}}(X)\right),
```

which measures the deviation of the true label $`Y`$ from the weighted
consensus interval.

##### Exchangeability.

Assume the data are i.i.d. Then the meta-calibration samples
$`(X_{1},Y_{1}),\dots,(X_{n},Y_{n})`$ and the test point
$`(X_{n+1},Y_{n+1})`$ are exchangeable. Consequently, the scores

```math
S^{\prime}_{1},\dots,S^{\prime}_{n},S^{\prime}_{n+1},\quad\text{where }S^{\prime}_{i}=S^{\prime}(X_{i},Y_{i}),
```

are exchangeable random variables.

##### Quantile guarantee.

Let $`Q^{\prime}_{\mathrm{meta}}`$ be the
$`\lceil(n+1)(1-\alpha)\rceil`$-th smallest value of the multiset
$`\{S^{\prime}_{1},\dots,S^{\prime}_{n},+\infty\}`$. By the conformal
prediction quantile lemma (Romano et al., 2019),

```math
\mathbb{P}\big(S^{\prime}_{n+1}\leq Q^{\prime}_{\mathrm{meta}}\big)\geq 1-\alpha.
```

The event $`S^{\prime}_{n+1}\leq Q^{\prime}_{\mathrm{meta}}`$ is
equivalent to

```math
\max\!\left(\hat{L}_{\mathrm{w}}(X_{n+1})-Y_{n+1},\;Y_{n+1}-\hat{U}_{\mathrm{w}}(X_{n+1})\right)\leq Q^{\prime}_{\mathrm{meta}}.
```

This holds if and only if

```math
\hat{L}_{\mathrm{w}}(X_{n+1})-Q^{\prime}_{\mathrm{meta}}\leq Y_{n+1}\leq\hat{U}_{\mathrm{w}}(X_{n+1})+Q^{\prime}_{\mathrm{meta}}.
```

Therefore,
$`\mathbb{P}\big(Y_{n+1}\in C_{\mathrm{final}}(X_{n+1})\big)\geq 1-\alpha`$,
which completes the proof. ∎

## 4 Experiments

##### Datasets.

We evaluate our approach on benchmarks spanning text summarization,
dialogue summarization, and reasoning. For summarization, we use
SummEval  Fabbri et al. (2021) and DialSumm  Goo and Chen (2018). Each
instance is annotated by three human raters using Likert-scale scores
along four dimensions—consistency, coherence, fluency, and relevance.
Following standard practice, we use the average of the three ratings as
the ground-truth label on a GPA-style scale. For reasoning, we adopt the
overall-quality Likert annotations provided in ROSCOE  Golovneva et al.
(2023), covering CosmosQA  Huang et al. (2019), DROP  Dua et al. (2019),
e-SNLI  Camburu et al. (2018), and GSM8K  Zeng et al. (2024), each
containing approximately 200 annotated examples.

##### Multi-Judge LLM Evaluation.

All evaluations are conducted using multiple LLM judges rather than a
single evaluator. We primarily adopt GEval  Liu et al. (2023b) with
chain-of-thought (CoT) prompting, and additionally include SocREval  He
et al. (2024) for reasoning tasks. The full set of prompts is provided
in Appendix. Our judge pool consists of GPT-4o mini (2024-07-18),
DeepSeek-R1-Distill-Qwen-32B  DeepSeek-AI (2025), and
Qwen2.5-72B-Instruct  Qwen et al. (2025). These models are selected
because they expose token-level logits required for conformal
calibration and represent diverse model families and training regimes.

##### Data Splits and Meta-Calibration.

To support both judge-level calibration and robust multi-judge
aggregation, we adopt a three-way split for each dataset. Specifically,
we allocate 40% of the data to a per-judge calibration set, used
independently by each LLM judge to construct conformal prediction
intervals; 10% to a held-out meta-calibration set $`D_{\mathrm{meta}}`$,
used exclusively to compute meta non-conformity scores for aggregating
intervals across judges; and the remaining 50% to a test set. The
meta-calibration set is never used to train, fine-tune, or calibrate any
individual LLM judge.

##### Conformal Prediction Methods.

We apply conformal prediction independently to each judge and then
aggregate the resulting intervals using our Robust Conformal Consensus
framework. We compare seven regression-based methods—CQR  Romano et al.
(2019), Asymmetric CQR  Romano et al. (2019), CHR  Sesia and Romano
(2021), LVD  Lin et al. (2021), Boosted CQR, Boosted LCP  Xie et al.
(2024), and R2CCP—as well as two ordinal classification-based methods,
Ordinal APS  Lu et al. (2022) and Ordinal Risk Control  Xu et al.
(2023). Ordinal methods are evaluated after applying the boundary
adjustment.

##### Evaluation Protocol.

All experiments are repeated over 30 random seeds (1–30). For each seed,
we resample the three-way data split, perform judge-level conformal
calibration, meta-calibration on $`D_{\mathrm{meta}}`$, and RCC-based
aggregation, and then evaluate on the test set. We report the average
empirical coverage and mean interval width of the final aggregated
prediction intervals; for summarization tasks, results are also reported
per evaluation dimension.

|                              |                                |               |               |               |                                |               |               |               |
| ---------------------------- | ------------------------------ | ------------- | ------------- | ------------- | ------------------------------ | ------------- | ------------- | ------------- |
| Method                       | SummEval Evaluated with G-Eval |               |               |               | ROSCOE Evaluated with SocREval |               |               |               |
|                              | Consistency                    | Coherence     | Fluency       | Relevance     | CosmosQA                       | DROP          | e-SNLI        | GSM8K         |
| GPT-4o mini                  |                                |               |               |               |                                |               |               |               |
| CQR                          | 1.15 / 94.16%                  | 2.87 / 93.15% | 1.44 / 92.92% | 2.09 / 90.92% | 3.53 / 95.27%                  | 3.82 / 96.70% | 3.04 / 96.62% | 3.53 / 95.67% |
| Asym CQR                     | 1.25 / 94.97%                  | 2.91 / 93.76% | 1.60 / 93.75% | 2.13 / 91.42% | 3.90 / 98.71%                  | 3.91 / 98.60% | 2.87 / 96.67% | 3.89 / 98.80% |
| CHR                          | 0.67 / 88.99%                  | 2.41 / 82.96% | 0.94 / 88.86% | 1.74 / 82.62% | 2.54 / 73.06%                  | 1.86 / 68.92% | 1.36 / 72.24% | 1.98 / 78.67% |
| LVD                          | 1.01 / 92.35%                  | 2.73 / 89.76% | 1.11 / 90.59% | 2.02 / 89.55% | 3.10 / 83.95%                  | 2.49 / 83.05% | 2.17 / 86.18% | 3.08 / 89.57% |
| Boosted CQR                  | 1.01 / 87.75%                  | 2.73 / 87.80% | 1.54 / 88.68% | 2.00 / 87.42% | 3.15 / 80.07%                  | 2.63 / 78.57% | 1.82 / 80.26% | 3.08 / 82.50% |
| Boosted LCP                  | 0.76 / 89.22%                  | 2.67 / 87.34% | 0.92 / 89.18% | 1.91 / 87.19% | 3.60 / 83.91%                  | 2.92 / 85.40% | 1.88 / 81.23% | 3.36 / 85.93% |
| R2CCP                        | 0.69 / 90.88%                  | 2.62 / 89.63% | 0.92 / 89.36% | 1.97 / 89.70% | 2.96 / 85.85%                  | 2.43 / 84.73% | 1.75 / 84.02% | 2.15 / 85.07% |
| DeepSeek-R1-Distill-Qwen-32B |                                |               |               |               |                                |               |               |               |
| CQR                          | 1.16 / 93.88%                  | 2.67 / 92.50% | 1.31 / 93.01% | 2.13 / 91.05% | 3.48 / 96.70%                  | 3.83 / 96.35% | 2.97 / 96.36% | 3.46 / 95.60% |
| Asym CQR                     | 1.30 / 95.13%                  | 2.72 / 92.86% | 1.49 / 94.52% | 2.21 / 92.06% | 3.84 / 99.08%                  | 3.95 / 99.27% | 2.86 / 96.05% | 3.85 / 98.43% |
| CHR                          | 0.82 / 91.17%                  | 2.23 / 87.07% | 0.90 / 89.24% | 1.87 / 86.38% | 2.66 / 76.50%                  | 1.95 / 78.06% | 1.38 / 71.97% | 2.01 / 81.60% |
| LVD                          | 0.97 / 92.93%                  | 2.43 / 91.10% | 1.00 / 91.10% | 2.04 / 90.14% | 3.25 / 88.10%                  | 2.62 / 88.06% | 2.24 / 90.96% | 3.02 / 90.63% |
| Boosted CQR                  | 1.10 / 89.30%                  | 2.36 / 88.98% | 1.16 / 89.46% | 2.00 / 88.98% | 3.17 / 82.72%                  | 2.47 / 81.11% | 1.79 / 80.96% | 2.94 / 79.83% |
| Boosted LCP                  | 0.77 / 89.20%                  | 2.32 / 86.70% | 0.93 / 89.10% | 1.91 / 86.89% | 3.48 / 81.60%                  | 2.79 / 85.46% | 1.84 / 80.61% | 3.43 / 85.23% |
| R2CCP                        | 0.69 / 90.44%                  | 2.30 / 90.12% | 0.89 / 90.09% | 2.00 / 89.84% | 2.94 / 86.97%                  | 2.29 / 86.35% | 1.85 / 87.87% | 1.88 / 85.33% |
| Qwen2.5-72B-Instruct         |                                |               |               |               |                                |               |               |               |
| CQR                          | 0.98 / 93.10%                  | 2.73 / 92.25% | 1.44 / 93.73% | 2.11 / 91.30% | 3.37 / 94.80%                  | 3.79 / 97.02% | 3.01 / 97.37% | 3.35 / 95.33% |
| Asym CQR                     | 1.11 / 94.47%                  | 2.80 / 93.13% | 1.63 / 94.79% | 2.17 / 92.21% | 3.86 / 99.01%                  | 3.89 / 98.67% | 2.77 / 96.84% | 3.87 / 98.97% |
| CHR                          | 0.61 / 89.04%                  | 2.14 / 80.93% | 0.98 / 88.93% | 1.61 / 79.61% | 2.44 / 72.65%                  | 2.08 / 75.87% | 1.22 / 69.69% | 1.81 / 77.50% |
| LVD                          | 0.85 / 92.82%                  | 2.55 / 90.49% | 1.09 / 90.94% | 1.94 / 89.27% | 3.05 / 84.29%                  | 2.67 / 90.57% | 1.91 / 85.96% | 2.83 / 90.13% |
| Boosted CQR                  | 0.80 / 88.28%                  | 2.46 / 87.82% | 1.24 / 89.22% | 1.88 / 87.17% | 3.05 / 79.08%                  | 2.56 / 81.17% | 1.51 / 77.11% | 2.81 / 80.67% |
| Boosted LCP                  | 0.67 / 88.81%                  | 2.43 / 86.92% | 0.94 / 89.26% | 1.86 / 87.51% | 3.46 / 80.41%                  | 2.81 / 85.75% | 1.74 / 77.50% | 3.38 / 86.23% |
| R2CCP                        | 0.61 / 90.73%                  | 2.44 / 89.54% | 0.95 / 90.18% | 1.98 / 90.45% | 2.90 / 85.34%                  | 2.39 / 86.25% | 1.59 / 84.50% | 2.00 / 86.73% |
| Multi-Agent LLM-as-a-Judge   |                                |               |               |               |                                |               |               |               |
| CQR                          | 1.09 / 93.7%                   | 2.75 / 92.6%  | 1.40 / 93.3%  | 2.11 / 91.1%  | 3.46 / 95.5%                   | 3.81 / 96.7%  | 3.00 / 96.7%  | 3.42 / 95.5%  |
| Asym CQR                     | 1.22 / 94.8%                   | 2.80 / 93.2%  | 1.57 / 94.4%  | 2.17 / 91.9%  | 3.87 / 98.9%                   | 3.92 / 98.9%  | 2.84 / 96.5%  | 3.86 / 98.7%  |
| CHR                          | 0.70 / 89.7%                   | 2.24 / 83.5%  | 0.94 / 89.0%  | 1.74 / 82.9%  | 2.54 / 74.0%                   | 1.98 / 75.3%  | 1.33 / 71.4%  | 1.92 / 79.4%  |
| LVD                          | 0.94 / 92.7%                   | 2.56 / 90.5%  | 1.06 / 90.9%  | 2.00 / 89.7%  | 3.13 / 85.4%                   | 2.61 / 88.1%  | 2.13 / 88.0%  | 2.95 / 90.3%  |
| Boosted CQR                  | 0.96 / 88.4%                   | 2.50 / 88.2%  | 1.29 / 89.2%  | 1.96 / 87.9%  | 3.12 / 80.5%                   | 2.54 / 80.6%  | 1.72 / 79.7%  | 2.91 / 80.6%  |
| Boosted LCP                  | 0.73 / 89.1%                   | 2.46 / 87.0%  | 0.93 / 89.2%  | 1.89 / 87.2%  | 3.51 / 82.0%                   | 2.82 / 85.6%  | 1.83 / 80.0%  | 3.40 / 85.8%  |
| R2CCP                        | 0.66 / 90.7%                   | 2.44 / 89.8%  | 0.92 / 90.0%  | 1.98 / 90.0%  | 2.93 / 86.0%                   | 2.36 / 86.0%  | 1.75 / 85.7%  | 1.97 / 85.9%  |

Table 1: Interval width and coverage on SummEval evaluated by G-Eval and
ROSCOE evaluated by SocREval before boundary adjustment. We mark
coverage $`<85\%`$ with gray text, coverage between $`85\%-90\%`$ with
underline and coverage with the smallest interval width $`\geq 90\%`$ in
bold.

### 4.1 Continuous Intervals Indicate Uncertainty

Table 1 reports the interval width and empirical coverage for different
conformal prediction methods on SummEval evaluated with G-Eval and on
ROSCOE evaluated with SocREval. The results show a clear trade-off
between interval width and coverage. Methods like CQR and Asym CQR
consistently achieve the target coverage (at or near 90%) across nearly
all evaluation dimensions and LLM judges, but they produce the widest
prediction intervals. In contrast, methods such as CHR and Boosted CQR
often yield narrower intervals but frequently fail to reach the 90%
coverage target, with many results falling below 85%, as indicated by
the gray text.

For the SummEval benchmark, most methods achieve coverage close to or
above 90% for the GPT-4o mini and Multi-Agent judge, particularly for
Fluency and Consistency. However, for other judges like Qwen2.5, methods
like CHR show coverage dips below 85% for Coherence and Relevance. On
the more challenging ROSCOE reasoning tasks, achieving the target
coverage is more difficult for most methods. While CQR and Asym CQR
again maintain high coverage, they do so with very wide intervals (e.g.,
often \>3.8). Other methods frequently result in coverage below 85%,
especially for the CosmosQA, DROP, and e-SNLI datasets. The LVD and
R2CCP methods often present a middle ground, achieving coverage between
85%-90% (underlined) with moderately sized intervals, and in several
cases (shown in bold) they achieve the smallest interval width while
still maintaining coverage of at least 90%.

The Multi-Agent judge demonstrates greater robustness by consistently
avoiding the lowest coverage failures seen in single models. For
example, with the CHR method on ROSCOE/CosmosQA, it achieves 74.0%
coverage where single models like Qwen2.5 fall to 72.65%. This indicates
that combining multiple LLMs yields more stable and reliable uncertainty
estimates across different tasks.

### 4.2 All Coverages Improve after Adjustment

Table 2 presents the interval width and coverage after boundary
adjustment for SummEval under G-Eval and for ROSCOE under SocREval. We
observe a consistent improvement in coverage across all settings after
applying boundary adjustment. In most cases, the adjusted prediction
intervals achieve coverage close to or exceeding the desired 90% level
across datasets, methods, and LLM judges. The gains are especially
notable for reasoning tasks, where coverage was often below 90% before
adjustment. For instance, LVD on e-SNLI evaluated with
Qwen2.5-72B-Instruct under SocREval increases from 85.96% to 95.53%.
Multi-LLM ensembles further stabilize uncertainty estimates: aggregating
intervals from multiple judges generally preserves or improves coverage
while maintaining relatively small interval widths. Overall, boundary
adjustment combined with multi-LLM aggregation produces more stable
uncertainty quantification across diverse tasks.

|                              |                                |               |               |               |                                |               |               |               |
| ---------------------------- | ------------------------------ | ------------- | ------------- | ------------- | ------------------------------ | ------------- | ------------- | ------------- |
| Method                       | SummEval Evaluated with G-Eval |               |               |               | ROSCOE Evaluated with SocREval |               |               |               |
|                              | Consistency                    | Coherence     | Fluency       | Relevance     | CosmosQA                       | DROP          | e-SNLI        | GSM8K         |
| GPT-4o mini                  |                                |               |               |               |                                |               |               |               |
| CQR                          | 1.15 / 95.45%                  | 2.87 / 94.94% | 1.44 / 93.80% | 2.09 / 93.56% | 3.53 / 95.34%                  | 3.82 / 97.05% | 3.04 / 96.89% | 3.53 / 95.67% |
| Asym CQR                     | 1.25 / 96.02%                  | 2.90 / 95.41% | 1.60 / 94.57% | 2.14 / 94.14% | 3.90 / 98.84%                  | 3.91 / 98.73% | 2.87 / 96.89% | 3.89 / 98.80% |
| CHR                          | 0.70 / 91.79%                  | 2.41 / 87.78% | 0.94 / 90.60% | 1.74 / 88.10% | 2.56 / 82.45%                  | 1.87 / 78.86% | 1.34 / 83.46% | 1.94 / 83.23% |
| LVD                          | 1.01 / 94.11%                  | 2.73 / 93.72% | 1.12 / 92.70% | 2.03 / 93.82% | 3.13 / 91.53%                  | 2.52 / 90.22% | 2.17 / 94.82% | 3.09 / 93.37% |
| Boosted CQR                  | 0.99 / 92.81%                  | 2.73 / 93.02% | 1.54 / 94.38% | 2.00 / 92.93% | 3.20 / 93.40%                  | 2.63 / 89.65% | 1.82 / 92.15% | 3.09 / 91.17% |
| Boosted LCP                  | 0.74 / 91.90%                  | 2.68 / 93.53% | 0.90 / 90.88% | 1.91 / 92.70% | 3.60 / 95.48%                  | 3.01 / 91.27% | 1.90 / 91.80% | 3.26 / 92.17% |
| R2CCP                        | 0.68 / 92.15%                  | 2.62 / 92.81% | 0.91 / 90.99% | 1.97 / 93.38% | 2.93 / 89.46%                  | 2.41 / 89.21% | 1.71 / 90.11% | 2.09 / 86.93% |
| OrdinalAPS                   | 2.28 / 71.48%                  | 1.88 / 64.84% | 1.78 / 13.65% | 2.36 / 87.94% | 0.73 / 47.52%                  | 0.83 / 55.08% | 0.72 / 52.76% | 0.58 / 73.90% |
| OrdinalRC                    | 2.41 / 75.19%                  | 2.02 / 67.38% | 1.93 / 14.58% | 2.51 / 90.30% | 0.82 / 49.46%                  | 0.91 / 57.11% | 0.80 / 54.61% | 0.60 / 74.43% |
| DeepSeek-R1-Distill-Qwen-32B |                                |               |               |               |                                |               |               |               |
| CQR                          | 1.15 / 95.02%                  | 2.67 / 94.34% | 1.32 / 94.44% | 2.13 / 93.67% | 3.48 / 96.80%                  | 3.82 / 96.54% | 2.99 / 96.80% | 3.46 / 95.63% |
| Asym CQR                     | 1.31 / 95.99%                  | 2.72 / 94.83% | 1.49 / 95.57% | 2.21 / 94.53% | 3.84 / 99.08%                  | 3.95 / 99.27% | 2.88 / 96.45% | 3.84 / 98.47% |
| CHR                          | 0.87 / 93.96%                  | 2.23 / 91.42% | 0.91 / 91.98% | 1.87 / 90.84% | 2.69 / 86.80%                  | 1.97 / 85.90% | 1.39 / 85.96% | 2.01 / 86.60% |
| LVD                          | 0.97 / 95.01%                  | 2.44 / 94.58% | 1.00 / 93.21% | 2.04 / 94.12% | 3.28 / 95.27%                  | 2.67 / 93.75% | 2.24 / 96.36% | 3.03 / 94.40% |
| Boosted CQR                  | 1.08 / 93.55%                  | 2.37 / 93.96% | 1.15 / 93.48% | 2.01 / 93.72% | 3.20 / 95.71%                  | 2.52 / 93.30% | 1.79 / 93.25% | 2.94 / 92.23% |
| Boosted LCP                  | 0.76 / 92.03%                  | 2.32 / 92.37% | 0.93 / 91.34% | 1.92 / 92.81% | 3.46 / 95.95%                  | 2.80 / 91.94% | 1.87 / 92.89% | 3.36 / 93.63% |
| R2CCP                        | 0.68 / 91.57%                  | 2.30 / 93.22% | 0.89 / 91.80% | 1.99 / 92.96% | 2.91 / 90.58%                  | 2.25 / 89.97% | 1.80 / 92.35% | 1.82 / 86.93% |
| OrdinalAPS                   | 2.51 / 90.06%                  | 2.52 / 90.64% | 3.76 / 91.08% | 2.13 / 89.98% | 1.32 / 60.00%                  | 1.26 / 78.22% | 1.46 / 87.85% | 1.50 / 85.67% |
| OrdinalRC                    | 2.54 / 90.11%                  | 2.56 / 91.18% | 3.73 / 89.53% | 2.14 / 90.07% | 1.44 / 62.35%                  | 1.33 / 78.22% | 1.52 / 88.33% | 1.55 / 86.07% |
| Qwen2.5-72B-Instruct         |                                |               |               |               |                                |               |               |               |
| CQR                          | 0.98 / 94.35%                  | 2.72 / 94.18% | 1.45 / 94.79% | 2.10 / 94.02% | 3.36 / 95.07%                  | 3.79 / 97.08% | 3.01 / 97.68% | 3.34 / 95.33% |
| Asym CQR                     | 1.10 / 95.47%                  | 2.79 / 94.70% | 1.64 / 95.63% | 2.17 / 94.85% | 3.85 / 99.18%                  | 3.89 / 98.67% | 2.77 / 97.06% | 3.87 / 98.97% |
| CHR                          | 0.66 / 92.21%                  | 2.14 / 86.10% | 0.98 / 91.16% | 1.61 / 85.78% | 2.49 / 82.14%                  | 2.05 / 82.89% | 1.18 / 84.56% | 1.79 / 85.27% |
| LVD                          | 0.85 / 95.11%                  | 2.56 / 94.05% | 1.09 / 93.45% | 1.95 / 93.86% | 3.07 / 92.01%                  | 2.67 / 93.87% | 1.91 / 95.53% | 2.87 / 93.43% |
| Boosted CQR                  | 0.81 / 92.36%                  | 2.47 / 93.06% | 1.25 / 93.66% | 1.88 / 92.81% | 3.10 / 94.01%                  | 2.56 / 90.79% | 1.49 / 92.11% | 2.82 / 92.03% |
| Boosted LCP                  | 0.65 / 91.26%                  | 2.44 / 92.26% | 0.93 / 91.20% | 1.86 / 92.57% | 3.40 / 94.90%                  | 2.84 / 92.41% | 1.79 / 91.84% | 3.33 / 92.90% |
| R2CCP                        | 0.59 / 91.83%                  | 2.43 / 92.78% | 0.95 / 92.12% | 1.98 / 93.72% | 2.88 / 89.29%                  | 2.34 / 90.00% | 1.55 / 90.20% | 1.96 / 88.57% |
| OrdinalAPS                   | 2.86 / 90.18%                  | 3.01 / 90.59% | 3.05 / 45.43% | 2.75 / 90.29% | 0.71 / 55.99%                  | 0.25 / 56.83% | 0.67 / 77.68% | 0.46 / 70.87% |
| OrdinalRC                    | 2.85 / 90.00%                  | 2.96 / 89.35% | 3.21 / 53.31% | 2.75 / 90.14% | 0.75 / 57.28%                  | 0.29 / 56.83% | 0.80 / 79.74% | 0.49 / 71.37% |
| Multi-Agent LLM-as-a-Judge   |                                |               |               |               |                                |               |               |               |
| CQR                          | 1.09 / 94.91%                  | 2.74 / 94.43% | 1.40 / 94.43% | 2.11 / 93.76% | 3.45 / 95.70%                  | 3.81 / 96.86% | 3.01 / 97.07% | 3.42 / 95.51% |
| Asym CQR                     | 1.22 / 95.81%                  | 2.80 / 94.93% | 1.58 / 95.35% | 2.18 / 94.54% | 3.86 / 99.04%                  | 3.92 / 98.92% | 2.85 / 96.76% | 3.86 / 98.73% |
| CHR                          | 0.74 / 92.64%                  | 2.24 / 88.29% | 0.95 / 91.30% | 1.74 / 88.27% | 2.58 / 83.69%                  | 1.98 / 83.27% | 1.32 / 84.76% | 1.90 / 85.52% |
| LVD                          | 0.94 / 94.76%                  | 2.57 / 94.14% | 1.07 / 93.19% | 2.01 / 93.94% | 3.16 / 92.87%                  | 2.64 / 93.09% | 2.13 / 95.63% | 2.97 / 93.83% |
| Boosted CQR                  | 0.95 / 92.89%                  | 2.51 / 93.35% | 1.29 / 93.77% | 1.96 / 93.18% | 3.16 / 94.34%                  | 2.56 / 91.56% | 1.72 / 92.58% | 2.91 / 91.99% |
| Boosted LCP                  | 0.71 / 91.71%                  | 2.46 / 92.63% | 0.92 / 91.17% | 1.90 / 92.69% | 3.49 / 95.42%                  | 2.86 / 91.99% | 1.86 / 92.25% | 3.33 / 93.10% |
| R2CCP                        | 0.65 / 91.85%                  | 2.44 / 92.93% | 0.92 / 91.74% | 1.98 / 93.35% | 2.91 / 89.75%                  | 2.32 / 89.83% | 1.70 / 91.03% | 1.92 / 87.64% |
| OrdinalAPS                   | 2.56 / 84.21%                  | 2.55 / 83.82% | 2.98 / 53.29% | 2.42 / 89.53% | 0.91 / 54.42%                  | 0.77 / 64.96% | 1.00 / 73.64% | 0.91 / 77.54% |
| OrdinalRC                    | 2.61 / 85.34%                  | 2.58 / 84.16% | 3.08 / 56.26% | 2.46 / 90.16% | 0.99 / 56.25%                  | 0.83 / 65.37% | 1.09 / 75.00% | 0.95 / 78.00% |

Table 2: Interval width and coverage on SummEval evaluated by G-Eval and
ROSCOE evaluated by SocREval after boundary adjustment. We mark coverage
$`<85\%`$ with gray text, coverage between $`85\%\textendash 90\%`$ with
underline and coverage $`\geq 90\%`$ with the smallest interval width in
bold.

## 5 Related work

##### Conformal Prediction and Robustness.

Conformal Prediction (CP) is a distribution-free framework for
uncertainty quantification that provides finite-sample marginal coverage
guarantees Shafer and Vovk (2008b). It has been widely used to construct
prediction intervals in regression and prediction sets in
classification, without making assumptions about the underlying data
distribution, which has a broader application in artificial intelligence
 Liu et al. (2019); Liu et al. (2021b); Liu et al. (2021a); Liu et al.
(2022a); Liu et al. (2022b); Liu et al. (2022c); Liu et al. (2023a); Liu
et al. (2024d); Liu et al. (2024b); Liu et al. (2024c); Liu et al.
(2024a); Liu et al. (2025b); Liu et al. (2025c); Liu (2025c); Liu
(2025b); Liu et al. (2025a); Liu (2024); Liu and Tong (2026d); Liu and
Tong (); Liu and Tong (2026a); Liu and Tong (2026b); Liu and Tong
(2026c); Liu and Tong (2026e); Liu and Shu (2025); Liu and Yan (2026);
Liu and Yang (2026); Liu and Tong (2026d); Wu and Liu (2026); Liu
(2026a); Liu (2026b); Liu and Tong (2023); Liu (2025a) . Recent studies
show that standard CP can fail to maintain coverage when inputs are
adversarially perturbed Gendler et al. (2022). A small number of works
have explored robust variants of CP. Early approaches lack formal
guarantees Hechtlinger et al. (2018), while more recent methods, such as
Randomly Smoothed Conformal Prediction (RSCP), combine CP with
randomized smoothing to provide robustness guarantees Gendler et al.
(2022). Although effective in theory, RSCP relies on inflating conformal
quantiles in proportion to the perturbation radius, often resulting in
overly conservative and large prediction sets. Overall, robust conformal
prediction remains an underexplored area, especially in complex
decision-making settings.

##### Uncertainty Quantification for LLM-as-a-Judge.

Uncertainty quantification for LLM-based judges is an emerging but still
limited area of research. Existing approaches typically rely on
token-level probabilities Wagner et al. (2024); Xie et al. (2025),
self-reported confidence (Yona et al., 2024; Xu et al., 2024), or
consistency across multiple generations (Tian et al., 2023). These
methods often suffer from bias, instability, overconfidence, or high
computational cost.

Conformal prediction has recently attracted attention as a principled,
post-hoc uncertainty quantification tool for LLMs due to its
distribution-free guarantees (Ye et al., 2024; Campos et al., 2024).
Most existing work applies CP to classification-style tasks, such as
multiple-choice question answering or response selection, where the goal
is to construct unordered prediction sets that include the correct
answer with high probability. The most closely related work to ours
applies conformalized risk control to align LLM judgments with human
preferences in pairwise comparisons (Jung et al., 2024).

## 6 Conclusion

This work addresses a key limitation of LLM-as-a-Judge evaluation by
moving beyond uncertainty estimates derived from a single LLM. We
propose a robust, multi-LLM interval adaptation framework that
integrates seamlessly with existing conformal prediction methods and
explicitly accounts for inter-LLM variability in rating-based evaluation
tasks. Through extensive experiments across multiple LLMs and conformal
techniques, we show that incorporating multiple judges produces more
stable prediction intervals with improved robustness, while maintaining
strong coverage guarantees.

## 7 Ethical Considerations

We have carefully assessed the potential risks associated with our work
and do not foresee any significant ethical concerns. Our framework is
designed with a strong emphasis on usability and ease of implementation,
lowering adoption barriers while minimizing operational complexities.
Additionally, our research is built upon an open-source dataset,
ensuring transparency, fostering collaboration, and promoting ethical
integrity by providing accessible and reproducible data.

## 8 Limitations

Despite its effectiveness, our approach has certain limitations. First,
our training data is limited in scope, primarily covering specific
domains rather than offering broad generalization across diverse topics.
This constraint may impact the model’s performance when handling queries
beyond these predefined areas. Furthermore, while our knowledge graph
provides valuable contextual information, it remains inherently
incomplete. Certain regions may lack sufficient data or relational
links, potentially leading to gaps in the model’s reasoning and
inference capabilities.

## References

- Barbosa et al. (2025) Juliana Silva Barbosa, Ulhas Gondhali, Gohar
  Petrossian, Kinshuk Sharma, Sunandan Chakraborty, Jennifer Jacquet,
  and Juliana Freire. 2025. A cost-effective llm-based approach to
  identify wildlife trafficking in online marketplaces. _Proceedings of
  the ACM on Management of Data_, 3(3):1–23.
- Bedi et al. (2025) Suhana Bedi, Yutong Liu, Lucy Orr-Ewing, Dev Dash,
  Sanmi Koyejo, Alison Callahan, Jason A Fries, Michael Wornow, Akshay
  Swaminathan, Lisa Soleymani Lehmann, et al. 2025. Testing and
  evaluation of health care applications of large language models: a
  systematic review. _Jama_.
- Camburu et al. (2018) Oana-Maria Camburu, Tim Rocktäschel, Thomas
  Lukasiewicz, and Phil Blunsom. 2018. e-snli: natural language
  inference with natural language explanations. In _Proceedings of the
  32nd International Conference on Neural Information Processing
  Systems_, NIPS’18, page 9560–9572, Red Hook, NY, USA. Curran
  Associates Inc.
- Campos et al. (2024) Margarida M Campos, António Farinhas, Chrysoula
  Zerva, Mário AT Figueiredo, and André FT Martins. 2024. Conformal
  prediction for natural language processing: A survey. _arXiv preprint
  arXiv:2405.01976_.
- DeepSeek-AI (2025) DeepSeek-AI. 2025. [Deepseek-r1: Incentivizing
  reasoning capability in llms via reinforcement
  learning](http://arxiv.org/abs/2501.12948).
- Dua et al. (2019) Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel
  Stanovsky, Sameer Singh, and Matt Gardner. 2019. DROP: A reading
  comprehension benchmark requiring discrete reasoning over paragraphs.
  In _Proceedings of the 2019 Conference of the North American Chapter
  of the Association for Computational Linguistics: Human Language
  Technologies, Volume 1 (Long and Short Papers)_, Minneapolis,
  Minnesota. Association for Computational Linguistics.
- Fabbri et al. (2021) Alexander R. Fabbri, Wojciech Kryściński, Bryan
  McCann, Caiming Xiong, Richard Socher, and Dragomir Radev. 2021.
  [Summeval: Re-evaluating summarization
  evaluation](http://arxiv.org/abs/2007.12626).
- Gendler et al. (2022) Asaf Gendler, Tsui-Wei Weng, Luca Daniel, and
  Yaniv Romano. 2022. Adversarially robust conformal prediction. In
  _International Conference on Learning Representations (ICLR)_.
- Golovneva et al. (2023) Olga Golovneva, Moya Chen, Spencer Poff,
  Martin Corredor, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli
  Celikyilmaz. 2023. [Roscoe: A suite of metrics for scoring
  step-by-step reasoning](http://arxiv.org/abs/2212.07919).
- Goo and Chen (2018) Chih-Wen Goo and Yun-Nung Chen. 2018. Abstractive
  dialogue summarization with sentence-gated modeling optimized by
  dialogue acts. In _Proceedings of 7th IEEE Workshop on Spoken Language
  Technology_.
- Gu et al. (2025) Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
  Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao
  Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and
  Jian Guo. 2025. [A survey on
  llm-as-a-judge](http://arxiv.org/abs/2411.15594).
- Guha et al. (2024) Etash Guha, Shlok Natarajan, Thomas Möllenhoff,
  Mohammad Emtiyaz Khan, and Eugene Ndiaye. 2024. Conformal prediction
  via regression-as-classification. _arXiv preprint arXiv:2404.08168_.
- He et al. (2024) Hangfeng He, Hongming Zhang, and Dan Roth. 2024.
  [SocREval: Large language models with the socratic method for
  reference-free reasoning
  evaluation](https://doi.org/10.18653/v1/2024.findings-naacl.175). In
  _Findings of the Association for Computational Linguistics: NAACL
  2024_, pages 2736–2764. Association for Computational Linguistics.
- Hechtlinger et al. (2018) Yotam Hechtlinger, Barnabás Póczos, and
  Larry Wasserman. 2018. Cautious deep learning. _arXiv preprint
  arXiv:1805.09460_.
- Huang et al. (2019) Lifu Huang, Ronan Le Bras, Chandra Bhagavatula,
  and Yejin Choi. 2019. [Cosmos qa: Machine reading comprehension with
  contextual commonsense reasoning](http://arxiv.org/abs/1909.00277).
- Jung et al. (2024) Jaehun Jung, Faeze Brahman, and Yejin Choi. 2024.
  Trust or escalate: Llm judges with provable guarantees for human
  agreement.
- Lin (2004) Chin-Yew Lin. 2004. [ROUGE: A package for automatic
  evaluation of summaries](https://aclanthology.org/W04-1013/). In _Text
  Summarization Branches Out_, pages 74–81, Barcelona, Spain.
  Association for Computational Linguistics.
- Lin et al. (2021) Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2021.
  Locally valid and discriminative prediction intervals for deep
  learning models. _Advances in Neural Information Processing Systems_,
  34:8378–8391.
- Liu (2024) Lihui Liu. 2024. _Knowledge graph reasoning and its
  applications: A pathway towards neural symbolic AI_. Ph.D. thesis,
  University of Illinois at Urbana-Champaign.
- Liu (2025a) Lihui Liu. 2025a. Graph-o1: Monte carlo tree search with
  reinforcement learning for text-attributed graph reasoning. _arXiv
  preprint arXiv:2512.17912_.
- Liu (2025b) Lihui Liu. 2025b. Hyperkgr: Knowledge graph reasoning in
  hyperbolic space with graph neural network encoding symbolic path. In
  _Proceedings of the 2025 Conference on Empirical Methods in Natural
  Language Processing_, pages 25188–25199.
- Liu (2025c) Lihui Liu. 2025c. Monte carlo tree search for graph
  reasoning in large language model agents. In _Proceedings of the 34th
  ACM International Conference on Information and Knowledge Management_,
  pages 4966–4970.
- Liu (2026a) Lihui Liu. 2026a. Multi-hop reasoning and retrieval in
  embedding space: Leveraging large language models with knowledge.
  _arXiv preprint arXiv:2603.13266_.
- Liu (2026b) Lihui Liu. 2026b. Neural-symbolic logic query answering in
  non-euclidean space. _arXiv preprint arXiv:2603.15633_.
- Liu et al. (2023a) Lihui Liu, Yuzhong Chen, Mahashweta Das, Hao Yang,
  and Hanghang Tong. 2023a. Knowledge graph question answering with
  ambiguous query. In _Proceedings of the ACM Web Conference 2023_,
  pages 2477–2486.
- Liu et al. (2025a) Lihui Liu, Jiayuan Ding, Subhabrata Mukherjee, and
  Carl J Yang. 2025a. Mixrag: Mixture-of-experts retrieval-augmented
  generation for textual graph understanding and question answering.
  _arXiv preprint arXiv:2509.21391_.
- Liu et al. (2021a) Lihui Liu, Boxin Du, Yi Ren Fung, Heng Ji, Jiejun
  Xu, and Hanghang Tong. 2021a. Kompare: A knowledge graph comparative
  reasoning system. In _Proceedings of the 27th ACM SIGKDD Conference on
  Knowledge Discovery & Data Mining_, pages 3308–3318.
- Liu et al. (2021b) Lihui Liu, Boxin Du, Heng Ji, ChengXiang Zhai, and
  Hanghang Tong. 2021b. Neural-answering logical queries on knowledge
  graphs. In _Proceedings of the 27th ACM SIGKDD Conference on Knowledge
  Discovery & Data Mining_, pages 1087–1097.
- Liu et al. (2019) Lihui Liu, Boxin Du, Hanghang Tong, et al. 2019.
  G-finder: Approximate attributed subgraph matching. In _2019 IEEE
  International Conference on Big Data (Big Data)_, pages 513–522. IEEE.
- Liu et al. (2022a) Lihui Liu, Boxin Du, Jiejun Xu, Yinglong Xia, and
  Hanghang Tong. 2022a. Joint knowledge graph completion and question
  answering. In _Proceedings of the 28th ACM SIGKDD Conference on
  Knowledge Discovery and Data Mining_, pages 1098–1108.
- Liu et al. (2024a) Lihui Liu, Blaine Hill, Boxin Du, Fei Wang, and
  Hanghang Tong. 2024a. Conversational question answering with language
  models generated reformulations over knowledge graph. In _Findings of
  the Association for Computational Linguistics ACL 2024_, pages
  839–850.
- Liu et al. (2022b) Lihui Liu, Houxiang Ji, Jiejun Xu, and Hanghang
  Tong. 2022b. Comparative reasoning for knowledge graph fact checking.
  In _2022 IEEE International Conference on Big Data (Big Data)_, pages
  2309–2312. IEEE.
- Liu et al. (2024b) Lihui Liu, Jinha Kim, and Vidit Bansal. 2024b. Can
  contrastive learning refine embeddings. _arXiv preprint
  arXiv:2404.08701_.
- Liu and Shu (2025) Lihui Liu and Kai Shu. 2025. Unifying knowledge in
  agentic llms: Concepts, methods, and recent advancements. _ACM SIGKDD
  Explorations Newsletter_, 27(2):88–96.
- (35) Lihui Liu and Hanghang Tong. Neural symbolic knowledge graph
  reasoning.
- Liu and Tong (2023) Lihui Liu and Hanghang Tong. 2023. Knowledge graph
  reasoning and its applications. In _Proceedings of the 29th ACM SIGKDD
  Conference on Knowledge Discovery and Data Mining_, pages 5813–5814.
- Liu and Tong (2026a) Lihui Liu and Hanghang Tong. 2026a. Accurate
  query answering with symbolic reasoning over complete kgs. In _Neural
  Symbolic Knowledge Graph Reasoning: A Pathway Towards Neural Symbolic
  AI_, pages 17–36. Springer.
- Liu and Tong (2026b) Lihui Liu and Hanghang Tong. 2026b. Ambiguous
  query answering with neural symbolic reasoning over incomplete kg. In
  _Neural Symbolic Knowledge Graph Reasoning: A Pathway Towards Neural
  Symbolic AI_, pages 89–106. Springer.
- Liu and Tong (2026c) Lihui Liu and Hanghang Tong. 2026c. Dynamic query
  answering with neural symbolic reasoning over incomplete kg. In
  _Neural Symbolic Knowledge Graph Reasoning: A Pathway Towards Neural
  Symbolic AI_, pages 121–136. Springer.
- Liu and Tong (2026d) Lihui Liu and Hanghang Tong. 2026d. _Neural
  Symbolic Knowledge Graph Reasoning: A Pathway Towards Neural Symbolic
  AI_. Springer Nature.
- Liu and Tong (2026e) Lihui Liu and Hanghang Tong. 2026e. Symbolic
  reasoning for inconsistency detection over complete kg. In _Neural
  Symbolic Knowledge Graph Reasoning: A Pathway Towards Neural Symbolic
  AI_, pages 37–54. Springer.
- Liu et al. (2024c) Lihui Liu, Zihao Wang, Jiaxin Bai, Yangqiu Song,
  and Hanghang Tong. 2024c. New frontiers of knowledge graph reasoning:
  Recent advances and future trends. In _Companion Proceedings of the
  ACM Web Conference 2024_, pages 1294–1297.
- Liu et al. (2024d) Lihui Liu, Zihao Wang, Ruizhong Qiu, Yikun Ban,
  Eunice Chan, Yangqiu Song, Jingrui He, and Hanghang Tong. 2024d. Logic
  query of thoughts: Guiding large language models to answer complex
  logic queries with knowledge graphs. _arXiv preprint
  arXiv:2404.04264_.
- Liu et al. (2025b) Lihui Liu, Zihao Wang, and Hanghang Tong. 2025b.
  Neural-symbolic reasoning over knowledge graphs: A survey from a query
  perspective. _ACM SIGKDD Explorations Newsletter_, 27(1):124–136.
- Liu et al. (2025c) Lihui Liu, Zihao Wang, Dawei Zhou, Ruijie Wang,
  Yuchen Yan, Bo Xiong, Sihong He, and Hanghang Tong. 2025c. Few-shot
  knowledge graph completion via transfer knowledge from similar tasks.
  In _Proceedings of the 34th ACM International Conference on
  Information and Knowledge Management_, pages 4960–4965.
- Liu and Yan (2026) Lihui Liu and Yuchen Yan. 2026. Morgan: To bridge
  mixture of experts and spectral graph neural network. In _Proceedings
  of the AAAI Conference on Artificial Intelligence_, volume 40, pages
  23783–23791.
- Liu and Yang (2026) Lihui Liu and Carl Yang. 2026. Prompt-tuning with
  attribute guidance for low-resource entity matching. _arXiv preprint
  arXiv:2603.19321_.
- Liu et al. (2022c) Lihui Liu, Ruining Zhao, Boxin Du, Yi Ren Fung,
  Heng Ji, Jiejun Xu, and Hanghang Tong. 2022c. Knowledge graph
  comparative reasoning for fact checking: Problem definition and
  algorithms. _IEEE Data Eng. Bull._, 45(4):19–38.
- Liu et al. (2023b) Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
  Ruochen Xu, and Chenguang Zhu. 2023b. G-eval: NLG evaluation using
  gpt-4 with better human alignment. In _Proceedings of the 2023
  Conference on Empirical Methods in Natural Language Processing_,
  Singapore. Association for Computational Linguistics.
- Lu et al. (2022) Charles Lu, Anastasios N Angelopoulos, and Stuart
  Pomerantz. 2022. Improving trustworthiness of ai disease severity
  rating in medical imaging with ordinal conformal prediction sets. In
  _International conference on medical image computing and
  computer-assisted intervention_, pages 545–554. Springer.
- Papineni et al. (2002) Kishore Papineni, Salim Roukos, Todd Ward, and
  Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine
  translation. In _Proceedings of the 40th Annual Meeting on Association
  for Computational Linguistics_, ACL ’02, page 311–318, USA.
  Association for Computational Linguistics.
- Qwen et al. (2025) Qwen, :, An Yang, Baosong Yang, and Beichen
  Zhang. 2025. [Qwen2.5 technical
  report](http://arxiv.org/abs/2412.15115).
- Romano et al. (2019) Yaniv Romano, Evan Patterson, and Emmanuel
  Candes. 2019. Conformalized quantile regression. _Advances in neural
  information processing systems_, 32.
- Sesia and Romano (2021) Matteo Sesia and Yaniv Romano. 2021. Conformal
  prediction using conditional histograms. _Advances in neural
  information processing systems_, 34:6304–6315.
- Shafer and Vovk (2008a) Glenn Shafer and Vladimir Vovk. 2008a. A
  tutorial on conformal prediction. _Journal of Machine Learning
  Research_, 9(3).
- Shafer and Vovk (2008b) Glenn Shafer and Vladimir Vovk. 2008b. A
  tutorial on conformal prediction. _Journal of Machine Learning
  Research_, 9(3).
- Sheng et al. (2025) Huanxin Sheng, Xinyi Liu, Hangfeng He, Jieyu Zhao,
  and Jian Kang. 2025. Analyzing uncertainty of LLM-as-a-judge: Interval
  evaluations with conformal prediction. In _Proceedings of the 2025
  Conference on Empirical Methods in Natural Language Processing_,
  Suzhou, China. Association for Computational Linguistics.
- Tian et al. (2023) Katherine Tian, Eric Mitchell, Allan Zhou, Archit
  Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D
  Manning. 2023. Just ask for calibration: Strategies for eliciting
  calibrated confidence scores from language models fine-tuned with
  human feedback.
- Wagner et al. (2024) Nico Wagner, Michael Desmond, Rahul Nair, Zahra
  Ashktorab, Elizabeth M Daly, Qian Pan, Martin Santillan Cooper,
  James M Johnson, and Werner Geyer. 2024. Black-box uncertainty
  quantification method for llm-as-a-judge.
- Wu and Liu (2026) Yukun Wu and Lihui Liu. 2026. Mixture of
  demonstrations for textual graph understanding and question answering.
  _arXiv preprint arXiv:2603.23554_.
- Xie et al. (2025) Qiujie Xie, Qingqiu Li, Zhuohao Yu, Yuejie Zhang,
  Yue Zhang, and Linyi Yang. 2025. An empirical analysis of uncertainty
  in large language model evaluations. _arXiv preprint_.
- Xie et al. (2024) Ran Xie, Rina Barber, and Emmanuel Candes. 2024.
  Boosted conformal prediction intervals. _Advances in Neural
  Information Processing Systems_, 37:71868–71899.
- Xu et al. (2024) Tianyang Xu, Shujin Wu, Shizhe Diao, Xiaoze Liu,
  Xingyao Wang, Yangyi Chen, and Jing Gao. 2024. Sayself: Teaching llms
  to express confidence with self-reflective rationales.
- Xu et al. (2023) Yunpeng Xu, Wenge Guo, and Zhi Wei. 2023. Conformal
  risk control for ordinal classification. In _Uncertainty in Artificial
  Intelligence_, pages 2346–2355. PMLR.
- Ye et al. (2024) Fanghua Ye, Mingming Yang, Jianhui Pang, Longyue
  Wang, Derek F Wong, Emine Yilmaz, Shuming Shi, and Zhaopeng Tu. 2024.
  Benchmarking llms via uncertainty quantification.
- Yona et al. (2024) Gal Yona, Roee Aharoni, and Mor Geva. 2024. Can
  large language models faithfully express their intrinsic uncertainty
  in words?
- Yuan and He (2024) Tongxin Yuan and Zhiwei He. 2024. R-judge:
  Benchmarking safety risk awareness for llm agents. _arXiv preprint
  arXiv:2401.10019_.
- Zeng et al. (2024) Zhongshen Zeng, Pengguang Chen, Shu Liu, Haiyun
  Jiang, and Jiaya Jia. 2024. [Mr-gsm8k: A meta-reasoning benchmark for
  large language model evaluation](http://arxiv.org/abs/2312.17080).
- Zhang et al. (2020) Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
  Weinberger, and Yoav Artzi. 2020. [Bertscore: Evaluating text
  generation with bert](http://arxiv.org/abs/1904.09675).
- Zheng and Chiang (2023) Lianmin Zheng and Wei-Lin Chiang. 2023.
  Judging llm-as-a-judge with mt-bench and chatbot arena. _Advances in
  neural information processing systems_, 36:46595–46623.
