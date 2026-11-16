---
identifier: arxiv:2604.02621v2
title: Reinforcement Learning-based Semi-supervised Knowledge Distillation with LLM-as-a-Judge
authors:
  - Yiyang Shen
  - Lifu Tu
  - Weiran Wang
published: "2026-04-03T01:23:04+00:00"
url: https://arxiv.org/abs/2604.02621v2
source: arxiv
doi: null
arxiv_id: 2604.02621v2
categories:
  - cs.CL
  - cs.LG
---

# Reinforcement Learning-based Semi-supervised Knowledge Distillation with LLM-as-a-Judge

Yiyang Shen    Lifu Tu\corresponding ^(†)^(†)thanks: Work done prior to
joining Oracle.    Weiran Wang\corresponding

###### Abstract

Reinforcement Learning (RL) substantially improves the reasoning
capabilities of language models, but most existing RL fine-tuning
approaches rely entirely on ground-truth verifiable rewards and thus
labeled datasets with verifiable answers. To overcome this, we propose a
RL framework for reasoning distillation that leverages continuous,
LLM-based rewards. Our method employs an efficient mechanism that
computes a continuous CoT reward (CCR) directly from a single-token
logit of a judge LLM, evaluating the student model’s reasoning
trajectory. This formulation provides an effective and scalable online
training signal that can be applied to large volumes of unlabeled data.
We demonstrate that, when paired with a strong judge, simply using CCR
achieves performance comparable to that of ground-truth or pseudo-label
verifiable rewards, and even surpasses them as the amount of unlabeled
data increases. Furthermore, we find that combining them in a
semi-supervised setup is highly synergistic: verifiable rewards help
stabilize the CCR, while CCR improves the generalizability of verifiable
rewards to related tasks. We also provide a comprehensive empirical
comparison of various reward sources across multiple model architectures
and dataset sizes. Our results show that this semi-supervised approach
consistently enhances mathematical reasoning, yielding an absolute
improvement of 5-10% on multiple reasoning tasks.

¹University of Iowa ²Oracle AI

{yiyang-shen,weiran-wang}@uiowa.edu, lifu.tu@oracle.com

## 1 Introduction

Recent advances have shown that reinforcement learning (RL) can
significantly improve reasoning capabilities of large language models
(LLMs), in particular on mathematical reasoning tasks where accurate
reward signals can be obtained from verifiable answers (Sutton et al.
1999; Schulman et al. 2017; Ouyang et al. 2022; Rafailov et al. 2023;
Trung et al. 2024; Guo et al. 2025). However, RL-based reasoning
training methods rely on labeled datasets and automatic verification,
and their applicability remains constrained in settings where ground
truth is unavailable (e.g., LLM generated question-answer pair not
verified by human experts) and abundant training signals are thus
difficult to obtain.

Knowledge distillation (KD) transfers reasoning ability from large
teacher models to smaller students (Hinton et al. 2015; Kim and Rush
2016). Mainstream methods such as supervised fine-tuning (SFT) rely on
students attempting to reproduce teacher’s reasoning trajectories (Gu et
al. 2026; Deng et al. 2024). To overcome the lack of labeled data, a
strong LLM can generate synthetic problems and reasoning trajectories
for both SFT and RL with verifiable rewards (RLVR) (Guo et al. 2025).
While effective, SFT approaches require large volumes of teacher
demonstrations and primarily optimize next-token prediction rather than
the underlying reasoning behavior. Naturally, reinforcement learning
allows the student to improve its own reasoning ability by receiving
feedback from a stronger teacher.

To address the reliance on ground truth, a growing line of research
explores learning from weaker or learned supervision signals, including
human preferences or model-based feedback (Ouyang et al. 2022; Bai et
al. 2022a; Bai et al. 2022b; Zhu et al. 2023; Rafailov et al. 2023).
Concurrently, the paradigm of utilizing an “LLM-as-a-Judge” has rapidly
gained traction. However, this paradigm has been predominantly
restricted to post-hoc evaluation—using a capable LLM to assess and rank
the quality of generations (Zheng et al. 2023b; Wang et al. 2023b).
While these LLM evaluators correlate highly with human judgment, their
potential to serve as an active, automated reward signal during RL
fine-tuning remains underexplored. Existing methods that attempt to
incorporate model-based feedback into training often distill it offline
into static preference datasets (Bai et al. 2022b; Lee et al. 2024) or
involve complex, separate reward modeling pipelines that scale poorly to
the massive sampling demands of continuous RL fine-tuning (Rafailov et
al. 2023; Yuan et al. 2024). Therefore, we ask the following question:

> Can we effectively use a capable LLM to provide active, continuous
> reward signals for training student models on unlabeled data?

In this work, we propose a semi-supervised RL method for mathematical
reasoning distillation. It uses both labeled and unlabeled data for
small LM fine-tuning, where a judge LLM evaluates the full reasoning
trajectory of each unlabeled example to provide a reward signal.

A key feature of our framework is an efficient reward formulation based
on single-token knowledge distillation, which we refer to as continuous
chain-of-thought reward (CCR). To avoid complex reward modeling
pipelines, we restrict the LLM judge to produce a binary decision (“yes”
or “no”) evaluating the student’s reasoning trajectory, and the reward
is computed directly from the logits of these two tokens. Unlike a hard
binary reward, the continuous probability derived from its logits
contains nuanced information about reasoning quality. Furthermore,
requiring only a single forward step for reward makes large-scale online
RL practical.

Beyond introducing this reward, we perform an empirical comparison of
different reward formulations prior to tuning, establishing that our
method correlates well with verifiable rewards. Through subsequent
training experiments, we demonstrate that CCR provides diverse and
effective training signals beyond mere correctness. We find CCR alone
yields performance comparable to that of ground-truth or pseudo-label
verifiable rewards and even surpasses them as the amount of unlabeled
data increases. Across mathematical reasoning benchmarks, combining
verifiable supervision on labeled data with LLM judge rewards on
unlabeled data consistently yields the strongest performance. Verifiable
rewards help stabilize CCR, while CCR improves generalizability to
related tasks, collectively demonstrating the effectiveness of
semi-supervised RL for reasoning problems.

Our contributions to reasoning knowledge distillation are summarized as
follows.

1.  1.  We propose a single-token judge reward formulation that efficiently
        distills reasoning behavior by providing continuous online reward
        signals from binary next-token probabilities on full reasoning
        trajectories, a departure from mainstream approaches performed on
        offline datasets.
2.  2.  We introduce a semi-supervised RL framework that combines verifiable
        reward on labeled data with customized reward formulations on
        unlabeled math problems, substantially improving reasoning
        performance.
3.  3.  We provide a comprehensive empirical evaluation and comparison of
        different reward sources, demonstrating their applicability to
        reasoning distillation. On tiny models (\<1B), our RL framework
        achieves a 5–10 point absolute improvement on mathematical reasoning
        tasks compared to using only verifiable rewards. Moreover, it yields
        even larger gains on out-of-domain evaluations, demonstrating
        improved robustness. On a larger model (6.7B), our method still
        provides an improvement of approximately 1.5 points, indicating that
        the proposed approach scales effectively to larger architectures.

## 2 Method

Reinforcement Learning with Verifiable Rewards (RLVR) has gained
prominence following its use in DeepSeek-R1 (Guo et al. 2025) for
o1-style (Learning to reason with LLMs 2024) long chain-of-thought
(CoT, Wei et al. 2022) reasoning. Yet, because RLVR’s benefits scale
with model size, smaller architectures experience only modest reasoning
improvements (Guo et al. 2025). In this work, we demonstrate that a
reinforcement learning framework guided by a carefully designed reward
system can effectively elicit latent knowledge from a large model to
enhance the reasoning capabilities of a smaller counterpart.

### 2.1 Problem setup

A supervised dataset consists of (question, CoT, answer) tuples denoted
as $`(x,e,y)`$, where the chain-of-thought (CoT) (Wei et al. 2023) $`e`$
represents the intermediate reasoning process that leads to the correct
answer $`y`$. The language model provides conditional probabilities of
the form $`\pi_{\theta}(a|s_{t})`$ where $`a`$ is a token in the
vocabulary, and $`s_{t}`$ denotes the “state” at step $`t`$. The state
can be interpreted as the generation context, including all tokens in
the question and all tokens generated so far, based on which the LM
predicts the next token. Once the token
$`a_{t+1}\sim\pi_{\theta}(a|s_{t})`$ is generated, it is appended to the
exist context to form $`s_{t+1}=[s_{t},a_{t+1}]`$ for generating the
next token. This process is repeated until the model generates an
end-of-sentence token $`<\!\!\text{eos}\!\!>`$. The initial state is
$`s_{0}=x`$.

### 2.2 Training Pipeline

#### Warm-up.

Depending on the base model’s initial capabilities and the target task
format, an initial Supervised Fine-Tuning (SFT) phase may be necessary
to equip the LLM with complex reasoning capabilities. For instance, our
GSM8K experiments require an SFT warm-up, whereas our AIME experiments
bypass this stage. During SFT, the model is trained to maximize the
log-likelihood of CoT sequences. Denote by
$`e=[a_{1},\dots,a_{L-1},a_{L}=<\!\!\text{eos}\!\!>]`$ the CoT token
sequence of length $`L`$. The SFT loss is defined as
$`{\mathcal{L}}_{\text{SFT}}(\theta)=-\mathbb{E}_{e\sim D}\left[\sum_{t=1}^{L}\log(\pi_{\theta}(a_{t}|s_{t-1}))\right]`$.

We can boost the model’s reasoning capability by having the model learn
from its experience using reinforcement learning (Sutton and Barto
2018). We view $`\pi_{\theta}(a|s_{t})`$ as a _policy_ which generates
tokens, or “actions” in the RL terminology, which evolves the state
$`s_{t}`$ until reaching the terminal state (when
$`<\!\!\text{eos}\!\!>`$ is generated), at which point a _raw_ reward is
given based on the state trajectory. All intermediate steps have a raw
reward of $`0`$.

#### Proximal Policy Optimization (PPO, Schulman et al. 2017).

The goal of RL is to maximize the expected reward given the policy. To
this end, we adopt the Proximal Policy Gradient (PPO) method as
implemented by Trung et al. (2024) for RL fine-tuning of a smaller LM.
Let $`r(s_{t-1},a_{t},s_{t})`$ be the _raw_ reward for generating
$`a_{t}`$ at step $`t`$; the design of raw reward is detailed in the
next subsection. For stability, PPO regularizes the learned policy to
stay close to the reference model $`\pi_{\text{ref}}`$ (e.g., a model
obtained by SFT warm-up) using a KL divergence penalty, and it employs a
value model $`V_{\phi}(\cdot)`$ to predict the expected reward for each
state.

#### Group Relative Policy Optimization (GRPO, Shao et al. 2024).

We also explore GRPO, a variant of PPO that eliminates the need for a
value model by sampling a group of responses and computing relative
advantages. The formal definitions for PPO and GRPO are provided in
Appendix A. While GRPO inherently demands higher computational cost due
to the generation of multiple hypotheses per question (group size
$`G=8`$ for GSM8K, and $`G=7`$ for AIME in our experiments), we provide
an empirical comparison in Section 5.1 demonstrating that both
algorithms yield comparable results on our tasks. Throughout this study,
we follow the method of choice in prior works for each respective
dataset.

### 2.3 Reward Design

#### Existing Reward Formulations.

The majority of previous works rely on verifiable rewards (VR), using
binary or partial rewards (Le et al. 2022; Trung et al. 2024) to
mitigate sparse signals (Riedmiller et al. 2018; Trott et al. 2019). For
instance, the reward can be defined as:

```math
r(s_{t-1},a_{t},s_{t})=\begin{cases}0,&s_{L}=\emptyset\\
0.1,&s_{L}\not=\tilde{y}\;\text{and}\;s_{L}\not=\emptyset\\
1,&s_{L}=\tilde{y},\end{cases}
```

where $`\tilde{y}`$ is the ground truth answer and $`s_{L}`$ is the
final answer string. In a natural language CoT setup, the model receives
a reward of 0 if the final answer cannot be extracted, 0.1 if it can be
extracted but is incorrect, and 1 if the extracted answer is correct. In
a programming language (programming of thought, PoT) setup (Gao et al.
2023), the reasoning process is represented as a Python program, which
is more rigorous and has lower perplexity than NL. Accordingly, we
assign a reward of 0 if the code fails to compile or times out, 0.1 if
it executes successfully but yields an incorrect answer, and a reward of
1 when the output matches the ground truth. We denote the loss function
associated with this reward as $`{\mathcal{L}}_{\text{VR}}`$.

Additionally, pretrained rerankers (PRR), typically used to select
best-of-$`k`$ inference results (Uesato et al. 2022), output a
correctness probability. We can repurpose this probability as a
continuous reward on unlabeled data, with the associated loss denoted
$`{\mathcal{L}}_{\text{PRR}}`$. However, relying solely on VR requires
extensive labeled data, while PRR necessitates training a separate
classifier for each specific task.

#### Our Novel Continuous CoT Reward (CCR).

To overcome these limitations and effectively leverage abundant
unlabeled data, we propose the Continuous CoT Reward (CCR). Rather than
using a teacher LLM to generate complete CoTs—which is computationally
heavy and prone to reasoning errors—we deploy it strictly as an online
evaluator of the student’s reasoning trajectories.

During RL fine-tuning, the student model’s generated candidate CoT is
appended to an evaluation prompt forcing the judge LLM to answer only
“Yes” or “No”. We extract the logits corresponding to these two tokens
to compute a continuous scalar reward. This provides a nuanced “degree
of correctness” that serves as a rich training signal, drawing
inspiration from the future constraint satisfaction score (Tu et al.
2024). Specifically, the score is calculated as:

```math
\displaystyle s(p(\texttt{Yes}),p(\texttt{No}))=\sigma\left(u\right)\cdot\mathbb{I}_{\{\sigma(u)\geq a\}}, \tag{1}
```

```math
\displaystyle\text{where}\;u=\frac{\log p(\texttt{Yes})-\log p(\texttt{No})}{\tau}.
```

$`\tau\geq 0`$ is a temperature parameter,
$`\sigma(u)=\frac{1}{1+\exp(-u)}`$ is the logistic sigmoid function, and
$`a`$ is a threshold so that scores smaller than $`a`$ will be set to 0.
We emphasize the novelty and efficiency of CCR: the reward depends
solely on the LLM logits from a single inference step without executing
code or generating full trajectories, and it remains largely
task-invariant with minimal prompt adjustments.

#### Novel Semi-Supervised Training Objective.

A key contribution of our work is formulating a novel semi-supervised RL
framework that balances ground-truth supervision with scalable,
unsupervised feedback. Verifiable rewards (VR) ensure reliable grounding
on labeled data, while unsupervised rewards—such as CCR and PRR, though
other reward formulations can readily be incorporated—provide rich,
continuous signals on unlabeled data. The overall objective is an convex
combination of these losses:

```math
\displaystyle{\mathcal{L}}_{\text{overall}}=\lambda\cdot{\mathcal{L}}_{\text{VR}}+\mu\cdot{\mathcal{L}}_{\text{CCR}}+\rho\cdot{\mathcal{L}}_{\text{PRR}}
```

where $`{\mathcal{L}}_{r}`$ denotes the loss (e.g., PPO or GRPO) using
reward $`r`$, with non-negative coefficients satisfying
$`\lambda+\mu+\rho=1`$.

## 3 Related Work

#### Knowledge Distillation.

Knowledge distillation (KD) transfers knowledge from a large teacher
model to a smaller student model (Hinton et al. 2015). Early work in NLP
extends KD to sequence generation tasks by distilling sequence-level
outputs (Kim and Rush 2016). A common approach is to minimize the
Kullback–Leibler (KL) divergence between teacher and student
distributions, which has been widely adopted in subsequent studies (Kim
and Rush 2016; Gu et al. 2026; Wu et al. 2025). Beyond likelihood-based
objectives, Tu and Gimpel (2018); Tu et al. (2020) proposed training an
inference network to match the teacher by minimizing its energy
function. Recent advancements have shifted focus toward distilling
complex reasoning capabilities from LLMs. For instance, Deng et al.
(2023) explore a novel paradigm of implicit chain-of-thought
distillation, transferring the explicit, token-by-token reasoning steps
of a teacher directly into the vertical hidden states of a student to
eliminate the need for generating intermediate text. Building on this
push for extreme model compression and efficiency, recent frameworks
like CODI (Shen et al. 2025) utilize self-distillation to compress
natural language reasoning into a continuous latent space by aligning
hidden states across tasks, successfully matching explicit reasoning
performance while drastically reducing computational overhead. Recent
research has integrated RL to further elicit and distill these complex
reasoning behaviors, as seen in DeepSeek-R1 (Guo et al. 2025). However,
conventional knowledge distillation via supervised fine-tuning (SFT) is
limited to surface-level imitation, failing to capture underlying
reasoning processes and resulting in poor generalization.

On-policy distillation (OPD) (Agarwal et al. 2024; Gu et al. 2024; Zhao
et al. 2026; He et al. 2026) trains the student using its own rollouts,
leveraging the teacher’s per-token log-probabilities as a dense reward.
In contrast, our method utilizes an online reward computed over the
entire reasoning chain on unlabeled data, avoiding token-level
supervision.

#### Reinforcement Learning.

Reinforcement learning (RL) has become a standard approach for aligning
large language models (LLMs) with desired behaviors. Early work on
reinforcement learning from human feedback (RLHF) relies on reward
models trained from costly human annotations (Ouyang et al. 2022; Bai et
al. 2022a). To reduce this dependence, subsequent methods introduce
model-based feedback, where LLMs replace human labelers (Bai et al.
2022b; Lee et al. 2024). However, these approaches typically construct
static preference datasets and train separate reward models, resulting
in complex and computationally expensive pipelines. To simplify
training, methods such as Direct Preference Optimization (DPO) (Rafailov
et al. 2023) and related techniques (Zhu et al. 2023; Yang et al. 2024)
optimize policies directly from pairwise preferences, while
Self-Rewarding Language Models (Yuan et al. 2024) iteratively generate
their own preference data for DPO training. Nonetheless, these
approaches remain limited by offline preference data. More recent work
explores richer reward signals beyond binary feedback (Damani et al.
2026; Gunjal et al. 2026), and integrates RL with distillation to
transfer reasoning capabilities (Guo et al. 2025). Here, we propose a
reinforcement learning–based knowledge distillation framework that
leverages multiple reward signals derived from LLM judges. Our approach
produces lightweight, real-time rewards from unlabeled data, reducing
the need for offline preference datasets and ground-truth supervision
while achieving strong performance on reasoning tasks.

## 4 Pre-training Evaluation of Reward Signals

Optimal distillation requires a high-quality judge and reward
formulation on reasoning trajectories. For fair comparison, we select
baselines requiring minimal token generation.

[TABLE]

Table 1: Datasets Statistics. Left: Grade School-level Math. Right:
Competition-level Math.

#### Likert.

We use the reference-Likert approach (Zheng et al. 2023a), which
provides the judge LLM with both a reference response and a candidate
generation, prompting it to output a discrete score (e.g., 1–10). A
notable drawback of this method is it creates extra CoTs, which
increases latency and the risk of parsing invalid scores. Its
reference-free version, direct-Likert, has been empirically shown to
have poor performance as a reward signal (Gunjal et al. 2026).

#### RaR.

This baseline adapts the implicit Rubrics-as-Rewards (RaR)
framework (Gunjal et al. 2026). Originally, RaR requires the LLM to
generate detailed rubrics composed of criteria and weights for every
question, incurring high computational cost. For fair comparison, we use
GPT-5.6 to generate one static rubric for GSM and another for AIME
(Appendix B). The judge LLM then implicitly aggregates these criteria to
score each response (1–10).

#### Token Likelihood (LL).

This approach utilizes the average token likelihood of the student’s CoT
as evaluated by the teacher model, measuring whether the student’s
reasoning trajectory aligns with the teacher’s distribution. However,
this method imposes significant computational overhead, as it requires
calculating logits across the entire generated trajectory. This makes
calculating reward for complex reasoning tasks that elicit long
reasoning chains expensive.

### 4.1 Empirical Comparison

We perform experiments on both grade-level math and competition-level
math, using datasets listed in Table 1.

|        |       |                |       |       |
| ------ | ----- | -------------- | ----- | ----- |
| Signal | Label | Judge          | GSM8K | SVAMP |
| VR     | ✓     | —              | 100.0 | 100.0 |
| Likert |       | Qwen3-8B       | 86.5  | 87.5  |
|        | ✓     | Llama3.1-8B    | 72.0  | 80.5  |
|        |       | Gemma3-12B     | 79.5  | 85.5  |
| RaR    |       | Qwen3-8B       | 74.5  | 82.0  |
|        | ✗     | Llama3.1-8B    | 57.5  | 62.0  |
|        |       | Gemma3-12B     | 70.0  | 72.5  |
| PRR    | ✗     | Galactica-6.7B | 87.5  | 80.5  |
| LL     |       | Qwen3-8B       | 81.5  | 79.0  |
|        | ✗     | Llama3.1-8B    | 81.0  | 79.5  |
|        |       | Gemma3-12B     | 49.5  | 64.0  |
| CCR    |       | Qwen3-8B       | 88.0  | 93.0  |
|        | ✗     | Llama3.1-8B    | 82.0  | 87.5  |
|        |       | Gemma3-12B     | 85.0  | 87.0  |

Table 2: Judge quality on Grade School-level Math.

| Judge Candidate | CCR  | LL   | RaR  | Likert |
| --------------- | ---- | ---- | ---- | ------ |
| Qwen3-8B        | 91.0 | 93.6 | 89.7 | 97.4   |
| Qwen3.5-9B      | 94.9 | 92.3 | 71.8 | 95.0   |
| Gemma3-12B      | 84.6 | 37.2 | 61.5 | 96.2   |

Table 3: Judge quality on AIME.

To evaluate the quality of the unsupervised reward signals, we sample
200 questions from the GSM8K, SVAMP, and the aggregated AIME datasets.
For each question, we generate a paired correct ($`y_{w}`$) and
incorrect ($`y_{l}`$) reasoning trajectory using the initial student
model. We then prompt the candidate judge models to evaluate these
trajectories and compute the respective rewards. A judgment is
considered correct if the reward assigned to the correct trajectory
exceeds that of the incorrect trajectory, i.e.,
$`r(x,y_{w})>r(x,y_{l})`$.

As shown in Table 2, Qwen3-8B consistently outperforms alternative
models across reward formulations on GSM8K chain-of-thought
trajectories; we thus adopt it as our primary grade school math judge.
Furthermore, CCR demonstrates competitive reward accuracy while
requiring far less runtime than generation-based rewards. Although PRR
achieves exceptional in-domain accuracy on GSM8K (where it was
explicitly trained on Galactica-6.7B trajectories), its performance
degrades out-of-distribution on SVAMP.

Table 3 presents results on AIME. Although reference-based Likert reward
yields the highest accuracy, its reliance on ground-truth labels
precludes its use in unsupervised training. Among reference-free
methods, combining Qwen3.5-9B and CCR achieves the best balance of
reward accuracy and minimal computational overhead, so we select it as
the AIME judge.

In Appendix C, we show that CCR takes the least amount of time among
different formulations to be computed.

## 5 Reasoning KD on Grade School-level Math

#### Supervised Datasets.

GSM8K (Cobbe et al. 2021) and SVAMP (Patel et al. 2021) are widely
adopted grade school-level math datasets for training and evaluating
reasoning LMs. We use their labeled splits for SFT and RL finetuning.
Across all our grade school-level math experiments, models are trained
to generate executable Python programs as reasoning (Program of
Thoughts, PoT).

#### Unsupervised Datasets.

We deploy our proposed CCR reward signal across two unlabeled datasets:
GSM8K-aug (Deng et al. 2024) and GSM-Plus (Li et al. 2024). GSM8K-aug
consists of questions generated by GPT-4; however, because it lacks
rigorous quality filtering, the provided answers are occasionally
incorrect, making it an ideal testbed for unsupervised reward signals.
Although GSM-Plus typically serves as an evaluation benchmark, we
repurpose its questions for unsupervised training by strictly masking
all ground-truth labels during the RL phase.¹¹ 1 We obtained explicit
permission from Li et al. to use GSM-Plus for training on this project.
We selected GSM-Plus because its expert-reviewed, perturbed problems
offer the necessary volume and structural complexity to push the
boundaries of the model’s reasoning capabilities.

#### Evaluation Datasets.

Several challenging benchmarks, including GSM-IC (Shi et al. 2023),
GSM-Plus (Li et al. 2024), GSM-Symbolic (Mirzadeh et al. 2025), and
GSM-$`\infty`$ (Zhou et al. 2025), have been proposed to evaluate the
robustness of reasoning LMs. In this study, we focus on GSM-Plus and
GSM-Symbolic. Of the 10,552 samples in GSM-Plus, we exclude 1,319
instances labeled as “None” because they lack a valid ground-truth
answer. GSM-Symbolic perturbs numeric variables from the original GSM8K
dataset via templated generation. To further escalate the difficulty, it
appends one (p1) or two (p2) additional clauses to each question,
thereby increasing context length and structural complexity.

#### Models.

We evaluate our distillation framework on two small LMs:
Galactica-125M (Taylor et al. 2022) and CodeGen-350M (Nijkamp et al.
2023). We use Qwen3-8B as the CCR generator and pretrained
Galactica-6.7B reward model as the PRR generator. While computational
constraints required us to employ fewer training epochs and a smaller
global batch size compared to Trung et al. (2024), we successfully
reproduce their core RLVR performance baseline. Full implementation
details are provided in Appendix D.

|                              |       |          |              |       |      |       |
| ---------------------------- | ----- | -------- | ------------ | ----- | ---- | ----- |
| Setup                        | GSM8K | GSM-Plus | GSM-Symbolic |       |      | SVAMP |
|                              |       |          | main         | p1    | p2   |       |
| Original GSM8K sup           |       |          |              |       |      |       |
| SFT                          | 25.25 | 15.14    | 22.88        | 7.40  | 3.64 | 32.30 |
| V                            | 30.48 | 18.38    | 30.96        | 8.42  | 0.96 | 31.30 |
| Pseudo-labeled GSM8K-aug sup |       |          |              |       |      |       |
| Base 1                       | 28.89 | 16.78    | 25.00        | 6.48  | 2.16 | 44.60 |
| Base 2                       | 36.01 | 26.64    | 37.20        | 14.58 | 1.96 | 51.00 |
| GSM8K-aug unsup              |       |          |              |       |      |       |
| C                            | 33.13 | 21.28    | 30.38        | 11.36 | 3.32 | 44.70 |
| P                            | 32.98 | 20.92    | 31.12        | 10.36 | 2.60 | 46.30 |
| C+P                          | 40.71 | 25.33    | 37.74        | 14.62 | 3.00 | 50.70 |
| GSM8K sup + GSM8K-aug unsup  |       |          |              |       |      |       |
| V+C                          | 35.10 | 22.14    | 34.08        | 12.76 | 3.32 | 40.70 |
| V+P                          | 39.27 | 25.04    | 36.86        | 12.84 | 1.44 | 43.30 |
| V+C+P                        | 40.11 | 25.26    | 37.50        | 15.70 | 2.24 | 42.60 |

Table 4: Results of Galactica-125M as measured by accuracy (%) on GSM8K.
V = VR, C = CCR, and P = PRR. We apply equal weights (0.5) when
combining two rewards, and set $`\lambda=0.5`$, $`\mu=0.25`$, and
$`\rho=0.25`$ for V+C+P.

### 5.1 Results on GSM8K

All RL experiments initialize from an SFT warm-up model achieving 21.83%
accuracy. A fully converged SFT model reaches 25.25% accuracy on GSM8K,
slightly exceeding Trung et al. (2024)’s 23.70% baseline.

#### Comparison against offline pseudo-labeling baselines.

We evaluate two strong baselines where GSM8K-aug labels are generated
offline by a teacher model and trained using conventional RLVR. In Base
1, we constrain our teacher to complete the answer in a maximum of 40
tokens without reasoning. In Base 2, we use the full CoT trajectories
generated by GPT-4 as pseudo-labels. Because these labels are
model-generated and lack human verification, they contain noise. While
Base 2 achieves a strong 36.01% on GSM8K (outperforming the original
supervised baseline), it falls short of our proposed rewards,
highlighting the superiority of online continuous evaluation over static
offline pseudo-labels.

#### Unsupervised reward signals surpass supervised baselines.

A compelling result of our framework is the strong performance of fully
unsupervised RL training. As shown in the “GSM8K-aug unsup” section of
Table 4, utilizing CCR (C) or PRR (P) independently achieves 33.13% and
32.98% accuracy on GSM8K, respectively, without relying on any
ground-truth labels during RL. Strikingly, both unsupervised methods
outperform the supervised verifiable reward baseline (V) trained on the
original labeled GSM8K dataset, which achieves only 30.48%. This
superiority is largely driven by the scalability of our unsupervised
approach: because our reward formulation circumvents the need for
ground-truth labels, we can leverage vastly larger sets of unlabeled
questions (e.g., GSM8K-aug) than would be possible with strictly
supervised datasets. When combining both unsupervised signals (C+P), the
performance jumps significantly to 40.71%, highlighting that CCR and PRR
can provide synergistic and diverse training signals. This demonstrates
that scaling rich reward signals over abundant unlabeled data may
effectively substitute costly annotations.

|                                  |          |              |       |      |       |
| -------------------------------- | -------- | ------------ | ----- | ---- | ----- |
| GSM8K                            | GSM-Plus | GSM-Symbolic |       |      | SVAMP |
|                                  |          | main         | p1    | p2   |       |
| GSM8K sup + 5% GSM8K-aug unsup   |          |              |       |      |       |
| 36.54                            | 22.85    | 31.96        | 10.24 | 0.96 | 43.10 |
| GSM8K sup + 25% GSM8K-aug unsup  |          |              |       |      |       |
| 38.36                            | 24.61    | 36.42        | 13.74 | 1.16 | 42.30 |
| GSM8K sup + 100% GSM8K-aug unsup |          |              |       |      |       |
| 40.11                            | 25.26    | 37.50        | 15.70 | 2.24 | 42.60 |
| GSM8K sup + GSM-Plus unsup       |          |              |       |      |       |
| 44.66                            | 30.67    | 38.74        | 10.88 | 0.84 | 32.50 |

Table 5: More and higher quality unsupervised data improves performance
on GSM8K. $`\lambda=0.5,\mu=0.25,\rho=0.25`$.

#### Semi-supervised RL provides superior performance and training stability.

As shown in the bottom section of Table 4, combining labeled and
unlabeled data under a semi-supervised setup yields the strongest
results across GSM evaluation benchmarks. Integrating all three reward
signals (VR, CCR, and PRR) achieves 40.11% accuracy on GSM8K,
representing a nearly 10% absolute gain over conventional supervised
RLVR (30.48%). Furthermore, semi-supervised training is more stable than
unsupervised training alone: while unsupervised models can experience
model collapse (where actual task accuracy plummets despite high proxy
rewards) after reaching peak accuracy, semi-supervised models maintain
steady reward progression throughout training (a trend even more
pronounced in Section 5.2 below). This demonstrates that anchoring rich,
continuous CoT rewards with verifiable ground-truth supervision provides
both synergistic learning signals and robust optimization dynamics.

#### Strong out-of-distribution generalization.

Beyond in-domain performance on GSM8K, our proposed continuous reward
formulations exhibit generalization to out-of-distribution (OOD)
benchmarks, including GSM-Plus, GSM-Symbolic, and SVAMP. As shown in
Table 4, the standard supervised verifiable reward (V) baseline
struggles to generalize, yielding only marginal OOD improvements or even
regressions (e.g., dropping from 32.30% to 31.30% on SVAMP) over the SFT
model. In contrast, our unsupervised (C+P) and semi-supervised (V+C+P)
methods demonstrate robustness. For instance, the fully unsupervised C+P
model improves GSM-Plus accuracy from 15.14% (SFT) to 25.33%, and
achieves a leap on SVAMP from 32.30% to 50.70%. This indicates that
reward signals generated by teacher models mitigate overfitting to the
training distribution and foster deeper mathematical reasoning
capabilities that transfer across diverse problem structures.

#### Scaling both quantity and quality of unsupervised data improves performance.

Table 5 demonstrates that scaling the amount of unsupervised data
consistently improves model performance. Furthermore, scaling data
quality proves equally impactful: although GSM-Plus contains
significantly fewer entries (9,233) than GSM8K-aug (385,461), training
on it yields highly competitive results.²² 2 GSM-Plus labels are
strictly used for evaluation, not fine-tuning (a valid transductive
setup). This indicates that utilizing a high-quality, human-reviewed
dataset provides immense value, confirming that optimizing both data
quantity and data quality are highly effective strategies.

|                                      |       |          |              |      |      |       |
| ------------------------------------ | ----- | -------- | ------------ | ---- | ---- | ----- |
| Setup                                | GSM8K | GSM-Plus | GSM-Symbolic |      |      | SVAMP |
|                                      |       |          | main         | p1   | p2   |       |
| Reported by Trung et al. (2024)      |       |          |              |      |      |       |
| V                                    | 68.9  |          |              |      |      |       |
| Continuing from the above checkpoint |       |          |              |      |      |       |
| V                                    | 69.1  | 52.0     | 64.2         | 54.8 | 23.4 | 69.4  |
| \+ voting                            | 71.7  | 53.3     | 67.4         | 57.9 | 27.0 | 70.8  |
| V+C                                  | 70.4  | 53.4     | 66.3         | 53.3 | 25.7 | 74.2  |
| \+ voting                            | 72.0  | 55.5     | 68.1         | 57.3 | 28.4 | 74.9  |

Table 6: Results of finetuning Galactica-6.7B on GSM8K. Here
$`\lambda=0.5,\mu=0.5`$ for V+C.

#### Scalability to Larger Architectures.

To verify that our reward framework generalizes to larger student
architectures, we apply our semi-supervised method to a 6.7B parameter
model. As detailed in Table 6, online continuous rewards boost RL
performance even when the student and judge models are of comparable
scale. Initializing from the Galactica-6.7B checkpoint provided by Trung
et al. (2024), we compare conventional RLVR against our semi-supervised
objective ($`V+C`$).³³ 3 Full $`V+C+P`$ on 6.7B requires ZeRO-3
offloading, making training prohibitively slow; we thus omit PRR.
Although the baseline RLVR performance is competitive at this scale,
incorporating our continuous CoT reward over augmented data yields a
notable gain, improving GSM8K accuracy from 69.1% to 70.4%. Applying
majority voting (Wang et al. 2023a) with 100 hypotheses further elevates
this to 72.0%. These findings confirm that our distillation framework
remains effective for larger models by capitalizing on vast amounts of
unlabeled data.

|                             |       |          |              |       |      |       |
| --------------------------- | ----- | -------- | ------------ | ----- | ---- | ----- |
| Setup                       | GSM8K | GSM-Plus | GSM-Symbolic |       |      | SVAMP |
|                             |       |          | main         | p1    | p2   |       |
| Original GSM8K only         |       |          |              |       |      |       |
| SFT                         | 22.97 | 14.07    | 23.40        | 8.34  | 2.96 | 28.20 |
| V                           | 26.91 | 16.35    | 29.90        | 10.54 | 4.16 | 34.30 |
| GSM8K sup + GSM8K-aug unsup |       |          |              |       |      |       |
| V+C                         | 31.01 | 19.11    | 33.86        | 11.54 | 4.72 | 38.50 |
| V+P                         | 33.13 | 19.84    | 31.62        | 13.68 | 3.60 | 36.80 |
| V+C+P                       | 31.31 | 19.63    | 30.38        | 10.74 | 3.56 | 35.90 |
| GSM8K sup + GSM-Plus unsup  |       |          |              |       |      |       |
| V+C                         | 35.63 | 22.98    | 33.12        | 12.08 | 2.08 | 32.80 |
| V+P                         | 36.24 | 24.12    | 29.94        | 12.34 | 3.12 | 32.70 |
| V+C+P                       | 37.00 | 24.99    | 33.00        | 14.50 | 3.08 | 29.90 |

Table 7: Results of CodeGen-350M on GSM8K.

#### CodeGen-350M Results.

In Table 7, we show that similar performance gains are observed using
another small model, CodeGen-350M. With semi-supervised training on
GSM8K-aug, we obtain 33.13% on GSM8K test set when combining VR with
PRR, showing a substantial increase from 26.91% which uses only VR on
labeled data. With higher quality unsupervised data (GSM-Plus), the
accuracy of this setting is further improved to 36.24%. Clearly, the
pre-trained rerank model is effective for a wide range of model sizes
and different model accuracies on this dataset. Our best result 37% is
achieved by semi-supervised training on GSM-Plus with both CCR and PRR.
We note that using only CCR or PRR on unlabeled data may lead to the
reward hacking phenomenon, causing model collapse. However, VR helps
stabilize semi-supervised training.

### 5.2 Results on SVAMP

We evaluate our framework on SVAMP using both small LMs, applying
optimal hyperparameters from GSM8K. As shown in Table 8, our
semi-supervised framework yields significant performance gains on SVAMP
across both small LMs. For Galactica-125M, incorporating augmented data
with all three rewards improves accuracy from 40.1% (supervised RLVR) to
44.3%. Likewise, for CodeGen-350M, accuracy increases from 41.4% to
46.6%. Notably, pure unsupervised training without VR suffers from the
aforementioned model collapse (see Appendix G for detailed analysis).
Integrating VR resolves this collapse, demonstrating stability of the
semi-supervised setup. Finally, we note an asymmetric generalization
effect: training on GSM8K transfers effectively to SVAMP, whereas
training on SVAMP yields poor performance on GSM8K (results not shown).

|                             |                |              |
| --------------------------- | -------------- | ------------ |
| Setup                       | Galactica-125M | CodeGen-350M |
| Original SVAMP only         |                |              |
| SFT                         | 37.40          | 35.80        |
| V                           | 40.10          | 41.40        |
| GSM8K-aug unsup             |                |              |
| C                           | 44.50          | $`\times`$   |
| P                           | $`\times`$     | $`\times`$   |
| SVAMP sup + GSM8K-aug unsup |                |              |
| V+C                         | 47.00          | 41.80        |
| V+P                         | 42.80          | 43.20        |
| V+C+P                       | 44.30          | 46.60        |

Table 8: Results on SVAMP. $`\times`$ indicates model collapse. We apply
equal weights (0.5) when combining two rewards, and set $`\lambda=0.5`$,
$`\mu=0.25`$, and $`\rho=0.25`$ for V+C+P.

## 6 Reasoning KD on Competition-level Math

We show below that our framework generalizes to harder natural language
problems and newer model families eliciting much more sophisticated
CoTs. We collect American Invitational Mathematics Examination (AIME)
problems from 1983 to 2023 as our primary supervised dataset, where
solutions typically require CoTs on the order of 10K tokens. Because
AIME answers are strictly positive integers, we pair it with the English
subset of DAPO-Math-17K (Yu et al. 2025). We select DAPO-Math-17K
because it underwent rigorous LLM-as-a-judge filtering and shares the
positive-integer answer format. We train student models to output pure
natural language reasoning for AIME and DAPO-Math, as these
competition-level problems necessitate complex, abstract conceptual
planning. Based on findings in Section 4, we adopt Qwen3.5-9B as our
judge model. We employ Qwen3-0.6B operating in thinking mode as the
student model. Given the difficulty of AIME problems and the variance
across reasoning paths, we report average@32 accuracy.

#### Results.

Our AIME results in Table 9 corroborate our earlier findings. First,
purely supervised RLVR on limited labeled data yields mixed results,
improving AIME ’24 but degrading performance on AIME ’25 and ’26.
Second, purely unsupervised training with CCR provides consistent gains
that scale directly with the volume of unlabeled data. Finally,
combining both in our semi-supervised framework (V+C) resolves these
inconsistencies, achieving the strongest overall performance across all
evaluation sets, highlighted by a 7.5% absolute gain on AIME ’24.

|                                           |         |         |         |
| ----------------------------------------- | ------- | ------- | ------- |
| Setup                                     | AIME’24 | AIME’25 | AIME’26 |
| HF Ckpt                                   | 9.2     | 16.5    | 11.5    |
| Supervised                                |         |         |         |
| V                                         | 10.7    | 15.1    | 11.0    |
| Unsupervised (until ’23)                  |         |         |         |
| C                                         | 12.0    | 16.9    | 12.7    |
| Unsupervised (until ’23 + DAPO)           |         |         |         |
| C                                         | 15.5    | 18.3    | 13.0    |
| Semi-supervised ($`\lambda=0.1,\mu=0.9`$) |         |         |         |
| V+C                                       | 16.7    | 20.7    | 15.2    |

Table 9: Avg@32 results of Qwen3-0.6B-Thinking on AIME.

## 7 Conclusions

We introduced a semi-supervised reasoning distillation framework powered
by our novel Continuous CoT Reward (CCR). By combining sparse verifiable
rewards with this continuous, inference-efficient teacher signal, our
approach effectively circumvents model collapse and reduces dependence
on labeled data, achieving significant gains on complex reasoning tasks.
Future directions include integrating more advanced policy gradient
algorithms (Yu et al. 2025) into our distillation framework, and
extending our methodology to fine-tune emerging diffusion language
models (Lou et al. 2024; Ye et al. 2024; Nie et al. 2025; Zekri and
Boullé 2025; Zhu et al. 2025).

## References

- Agarwal et al. (2024) R. Agarwal, N. Vieillard, Y. Zhou, P.
  Stanczyk, S. Ramos, M. Geist, and O. Bachem On-policy distillation of
  language models: learning from self-generated mistakes. In ICLR,
  External Links: [Link](https://openreview.net) Cited by: §3.
- Bai et al. (2022a) Y. Bai, A. Jones, K. Ndousse, A. Askell, A.
  Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, N.
  Joseph, S. Kadavath, J. Kernion, T. Conerly, S. El-Showk, N.
  Elhage, Z. Hatfield-Dodds, D. Hernandez, T. Hume, S. Johnston, S.
  Kravec, L. Lovitt, N. Nanda, C. Olsson, D. Amodei, T. Brown, J.
  Clark, S. McCandlish, C. Olah, B. Mann, and J. Kaplan Training a
  helpful and harmless assistant with reinforcement learning from human
  feedback. Note: arXiv 2204.05862 \[cs.CL\] External Links:
  [Link](https://arxiv.org/abs/2204.05862) Cited by: §1, §3.
- Bai et al. (2022b) Y. Bai, S. Kadavath, S. Kundu, A. Askell, J.
  Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, C.
  Chen, C. Olsson, C. Olah, D. Hernandez, D. Drain, D. Ganguli, D.
  Li, E. Tran-Johnson, E. Perez, J. Kerr, J. Mueller, J. Ladish, J.
  Landau, K. Ndousse, K. Lukosuite, L. Lovitt, M. Sellitto, N.
  Elhage, N. Schiefer, N. Mercado, N. DasSarma, R. Lasenby, R.
  Larson, S. Ringer, S. Johnston, S. Kravec, S. E. Showk, S. Fort, T.
  Lanham, T. Telleen-Lawton, T. Conerly, T. Henighan, T. Hume, S. R.
  Bowman, Z. Hatfield-Dodds, B. Mann, D. Amodei, N. Joseph, S.
  McCandlish, T. Brown, and J. Kaplan Constitutional AI: Harmlessness
  from AI feedback. Note: arXiv 2212.08073 \[cs.CL\] External Links:
  [Link](https://arxiv.org/abs/2212.08073) Cited by: §1, §3.
- Cobbe et al. (2021) K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H.
  Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C.
  Hesse, and J. Schulman Training verifiers to solve math word problems.
  External Links: 2110.14168, [Link](https://arxiv.org/abs/2110.14168)
  Cited by: §5.
- Damani et al. (2026) M. Damani, I. Puri, S. Slocum, I. Shenfeld, L.
  Choshen, Y. Kim, and J. Andreas Beyond binary rewards: Training LMs to
  reason about their uncertainty. In ICLR, External Links:
  [Link](https://openreview.net/forum?id=ASQ649zdHm) Cited by: §3.
- Deng et al. (2023) Y. Deng, K. Prasad, R. Fernandez, P. Smolensky, V.
  Chaudhary, and S. Shieber Implicit chain of thought reasoning via
  knowledge distillation. External Links: 2311.01460,
  [Link](https://arxiv.org/abs/2311.01460) Cited by: §3.
- Deng et al. (2024) Y. Deng, K. Prasad, R. Fernandez, P. Smolensky, V.
  Chaudhary, and S. Shieber Implicit chain of thought reasoning via
  knowledge distillation. In ICLR, Cited by: §D.1, §1, §5.
- Gao et al. (2023) L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y.
  Yang, J. Callan, and G. Neubig PAL: Program-aided language models. In
  ICML, Cited by: §2.3.
- Gu et al. (2026) Y. Gu, L. Dong, F. Wei, and M. Huang MiniLLM:
  on-policy distillation of large language models. External Links:
  2306.08543, [Link](https://arxiv.org/abs/2306.08543) Cited by: §1, §3.
- Gu et al. (2024) Y. Gu, L. Dong, F. Wei, and Z. Sui MiniLLM: knowledge
  distillation of large language models. In ICLR, External Links:
  [Link](https://openreview.net/forum?id=5h0qf7IBZZ) Cited by: §3.
- Gunjal et al. (2026) A. Gunjal, A. Wang, E. Lau, V. Nath, Y. He, B.
  Liu, and S. Hendryx Rubrics as rewards: reinforcement learning beyond
  verifiable domains. In ICLR, Cited by: §3, §4, §4.
- Guo et al. (2025) D. Guo, D. Yang, H. Zhang, J. Song, and P. Wang
  DeepSeek-R1: incentivizes reasoning in LLMs through reinforcement
  learning. Nature 645 (8081), pp. 633–638. External Links: ISSN
  1476-4687, [Link](http://dx.doi.org/10.1038/s41586-025-09422-z),
  [Document](https://dx.doi.org/10.1038/s41586-025-09422-z) Cited by:
  §A.2, §1, §1, §2, §3, §3.
- He et al. (2026) Y. He, S. Kaur, A. Bhaskar, Y. Yang, J. Liu, N.
  Ri, L. Fowl, A. Panigrahi, D. Chen, and S. Arora Self-distillation
  zero: self-revision turns binary rewards into dense supervision.
  External Links: 2604.12002, [Link](https://arxiv.org/abs/2604.12002)
  Cited by: §3.
- Hinton et al. (2015) G. Hinton, O. Vinyals, and J. Dean Distilling the
  knowledge in a neural network. External Links: 1503.02531,
  [Link](https://arxiv.org/abs/1503.02531) Cited by: §1, §3.
- Kim and Rush (2016) Y. Kim and A. M. Rush Sequence-level knowledge
  distillation. In EMNLP, External Links:
  [Link](https://aclanthology.org/D16-1139/),
  [Document](https://dx.doi.org/10.18653/v1/D16-1139) Cited by: §1, §3.
- Le et al. (2022) H. Le, Y. Wang, A. D. Gotmare, S. Savarese,
  and S. C. H. Hoi CodeRL: mastering code generation through pretrained
  models and deep reinforcement learning. In NeurIPS, External Links:
  [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/8636419dea1aa9fbd25fc4248e702da4-Paper-Conference.pdf)
  Cited by: §2.3.
- Learning to reason with LLMs (2024) External Links:
  [Link](https://openai.com/index/learning-to-reason-with-llms/) Cited
  by: §2.
- Lee et al. (2024) H. Lee, S. Phatale, H. Mansoor, T. Mesnard, J.
  Ferret, K. Lu, C. Bishop, E. Hall, V. Carbune, A. Rastogi, and S.
  Prakash RLAIF vs. RLHF: Scaling reinforcement learning from human
  feedback with AI feedback. In ICML, Cited by: §1, §3.
- Li et al. (2024) Q. Li, L. Cui, X. Zhao, L. Kong, and W. Bi GSM-plus:
  a comprehensive benchmark for evaluating the robustness of LLMs as
  mathematical problem solvers. In ACL, External Links:
  [Link](https://aclanthology.org/2024.acl-long.163/),
  [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.163) Cited by:
  §5, §5, footnote 1.
- Lou et al. (2024) A. Lou, C. Meng, and S. Ermon Discrete diffusion
  modeling by estimating the ratios of the data distribution. In ICML,
  Cited by: §7.
- Mirzadeh et al. (2025) S. I. Mirzadeh, K. Alizadeh, H. Shahrokhi, O.
  Tuzel, S. Bengio, and M. Farajtabar GSM-symbolic: understanding the
  limitations of mathematical reasoning in large language models. In
  ICLR, Cited by: §5.
- Nie et al. (2025) S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J.
  Zhou, Y. Lin, J. Wen, and C. Li Large language diffusion models. arXiv
  preprint arXiv:2502.09992. Cited by: §7.
- Nijkamp et al. (2023) E. Nijkamp, B. Pang, H. Hayashi, L. Tu, H.
  Wang, Y. Zhou, S. Savarese, and C. Xiong CodeGen: an open large
  language model for code with multi-turn program synthesis. In ICLR,
  Cited by: §5.
- Ouyang et al. (2022) L. Ouyang, J. Wu, X. Jiang, D. Almeida, C.
  Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al.
  Training language models to follow instructions with human feedback.
  In NeurIPS, Cited by: §1, §1, §3.
- Patel et al. (2021) A. Patel, S. Bhattamishra, and N. Goyal Are NLP
  models really able to solve simple math word problems?. In Proceedings
  of the 2021 Conference of the North American Chapter of the
  Association for Computational Linguistics: Human Language
  Technologies, External Links:
  [Link](https://aclanthology.org/2021.naacl-main.168/),
  [Document](https://dx.doi.org/10.18653/v1/2021.naacl-main.168) Cited
  by: §5.
- Rafailov et al. (2023) R. Rafailov, A. Sharma, E. Mitchell, S.
  Ermon, C. D. Manning, and C. Finn Direct preference optimization: your
  language model is secretly a reward model. arXiv preprint
  arXiv:2305.18290. Cited by: §1, §1, §3.
- Riedmiller et al. (2018) M. Riedmiller, R. Hafner, T. Lampe, M.
  Neunert, J. Degrave, T. van de Wiele, V. Mnih, N. Heess, and J. T.
  Springenberg Learning by playing solving sparse reward tasks from
  scratch. In ICML, External Links:
  [Link](https://proceedings.mlr.press/v80/riedmiller18a.html) Cited by:
  §2.3.
- Schulman et al. (2018) J. Schulman, P. Moritz, S. Levine, M. Jordan,
  and P. Abbeel High-dimensional continuous control using generalized
  advantage estimation. External Links: 1506.02438,
  [Link](https://arxiv.org/abs/1506.02438) Cited by: §A.1.
- Schulman et al. (2017) J. Schulman, F. Wolski, P. Dhariwal, A.
  Radford, and O. Klimov Proximal policy optimization algorithms.
  External Links: 1707.06347, [Link](https://arxiv.org/abs/1707.06347)
  Cited by: §1, §2.2.
- Shao et al. (2024) Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H.
  Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo DeepSeekMath: Pushing the
  limits of mathematical reasoning in open language models. External
  Links: 2402.03300, [Link](https://arxiv.org/abs/2402.03300) Cited by:
  §2.2.
- Shen et al. (2025) Z. Shen, H. Yan, L. Zhang, Z. Hu, Y. Du, and Y. He
  CODI: compressing chain-of-thought into continuous space via
  self-distillation. In EMNLP, External Links:
  [Link](https://aclanthology.org/2025.emnlp-main.36/),
  [Document](https://dx.doi.org/10.18653/v1/2025.emnlp-main.36) Cited
  by: §3.
- Shi et al. (2023) F. Shi, X. Chen, K. Misra, N. Scales, D.
  Dohan, E. H. Chi, N. Schärli, and D. Zhou Large language models can be
  easily distracted by irrelevant context. In ICML, External Links:
  [Link](https://proceedings.mlr.press/v202/shi23a.html) Cited by: §5.
- Skalse et al. (2025) J. Skalse, N. H. R. Howe, D. Krasheninnikov,
  and D. Krueger Defining and characterizing reward hacking. External
  Links: 2209.13085, [Link](https://arxiv.org/abs/2209.13085) Cited by:
  Appendix G.
- Sutton et al. (1999) R. S. Sutton, D. McAllester, S. Singh, and Y.
  Mansour Policy gradient methods for reinforcement learning with
  function approximation. In NeurIPS, Cited by: §1.
- Sutton and Barto (2018) R. S. Sutton and A. G. Barto Reinforcement
  learning: an introduction. 2nd edition, MIT Press. Cited by: §2.2.
- Taylor et al. (2022) R. Taylor, M. Kardas, G. Cucurull, T. Scialom, A.
  Hartshorn, E. Saravia, A. Poulton, V. Kerkez, and R. Stojnic
  Galactica: a large language model for science. External Links:
  2211.09085, [Link](https://arxiv.org/abs/2211.09085) Cited by: §5.
- Trott et al. (2019) A. Trott, S. Zheng, C. Xiong, and R. Socher
  Keeping your distance: solving sparse reward tasks using
  self-balancing shaped rewards. In NeurIPS, Cited by: §2.3.
- Trung et al. (2024) L. Trung, X. Zhang, Z. Jie, P. Sun, X. Jin, and H.
  Li ReFT: Reasoning with reinforced fine-tuning. In ACL, Cited by:
  §D.1, Table 15, Appendix G, §1, §2.2, §2.3, §5, §5.1, §5.1, Table 6.
- Tu and Gimpel (2018) L. Tu and K. Gimpel Learning approximate
  inference networks for structured prediction. In ICLR, Cited by: §3.
- Tu et al. (2020) L. Tu, R. Y. Pang, S. Wiseman, and K. Gimpel ENGINE:
  energy-based inference networks for non-autoregressive machine
  translation. In ACL, External Links:
  [Link](https://aclanthology.org/2020.acl-main.251/),
  [Document](https://dx.doi.org/10.18653/v1/2020.acl-main.251) Cited by:
  §3.
- Tu et al. (2024) L. Tu, S. Yavuz, J. Qu, J. Xu, R. Meng, C. Xiong,
  and Y. Zhou Unlocking anticipatory text generation: a constrained
  approach for large language models decoding. In EMNLP, External Links:
  [Link](https://aclanthology.org/2024.emnlp-main.870/),
  [Document](https://dx.doi.org/10.18653/v1/2024.emnlp-main.870) Cited
  by: §2.3.
- Uehara et al. (2024) M. Uehara, Y. Zhao, K. Black, E.
  Hajiramezanali, G. Scalia, N. L. Diamant, A. M. Tseng, T. Biancalani,
  and S. Levine Fine-tuning of continuous-time diffusion models as
  entropy-regularized control. External Links: 2402.15194,
  [Link](https://arxiv.org/abs/2402.15194) Cited by: Appendix G.
- Uesato et al. (2022) J. Uesato, N. Kushman, R. Kumar, F. Song, N.
  Siegel, L. Wang, A. Creswell, G. Irving, and I. Higgins Solving math
  word problems with process- and outcome-based feedback. External
  Links: 2211.14275, [Link](https://arxiv.org/abs/2211.14275) Cited by:
  §2.3.
- Wang et al. (2023a) X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H.
  Chi, S. Narang, A. Chowdhery, and D. Zhou Self-consistency improves
  chain of thought reasoning in language models. In ICLR, Cited by:
  §5.1.
- Wang et al. (2023b) Y. Wang, Z. Yu, Z. Zeng, L. Yang, C. Wang, et al.
  PandaLM: An automatic evaluation benchmark for llm instruction tuning
  optimization. arXiv preprint arXiv:2306.05087. Cited by: §1.
- Wei et al. (2022) J. Wei, X. Wang, D. Schuurmans, M. Bosma, brian
  ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou Chain of thought
  prompting elicits reasoning in large language models. In NeurIPS,
  Cited by: §2.
- Wei et al. (2023) J. Wei, X. Wang, D. Schuurmans, M. Bosma, B.
  Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou Chain-of-thought prompting
  elicits reasoning in large language models. External Links:
  2201.11903, [Link](https://arxiv.org/abs/2201.11903) Cited by: §2.1.
- Wu et al. (2025) T. Wu, C. Tao, J. Wang, R. Yang, Z. Zhao, and N. Wong
  Rethinking Kullback-Leibler divergence in knowledge distillation for
  large language models. In Proceedings of the 31st International
  Conference on Computational Linguistics, Cited by: §3.
- Yang et al. (2024) K. Yang, D. Klein, A. Celikyilmaz, N. Peng, and Y.
  Tian RLCD: reinforcement learning from contrastive distillation for LM
  alignment. In ICLR, External Links:
  [Link](https://openreview.net/forum?id=v3XXtxWKi6) Cited by: §3.
- Ye et al. (2024) J. Ye, S. Gong, L. Chen, L. Zheng, J. Gao, H. Shi, C.
  Wu, Z. Li, W. Bi, and L. Kong Diffusion of thoughts: chain-of-thought
  reasoning in diffusion language models. In NeurIPS, Cited by: §7.
- Yu et al. (2025) Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W.
  Dai, T. Fan, G. Liu, L. Liu, X. Liu, H. Lin, Z. Lin, B. Ma, G.
  Sheng, Y. Tong, C. Zhang, M. Zhang, W. Zhang, H. Zhu, J. Zhu, J.
  Chen, J. Chen, C. Wang, H. Yu, Y. Song, X. Wei, H. Zhou, J. Liu, W.
  Ma, Y. Zhang, L. Yan, M. Qiao, Y. Wu, and M. Wang DAPO: An open-source
  llm reinforcement learning system at scale. External Links:
  2503.14476, [Link](https://arxiv.org/abs/2503.14476) Cited by: §6, §7.
- Yuan et al. (2024) W. Yuan, R. Y. Pang, K. Cho, X. Li, S.
  Sukhbaatar, J. Xu, and J. Weston Self-rewarding language models. In
  ICML, Cited by: §1, §3.
- Zekri and Boullé (2025) O. Zekri and N. Boullé Fine-tuning discrete
  diffusion models with policy gradient methods. In NeurIPS, Cited by:
  §7.
- Zhao et al. (2026) S. Zhao, Z. Xie, M. Liu, J. Huang, P. Pang, F.
  Chen, and A. Grover Self-distilled reasoner: on-policy
  self-distillation for large language models. In ICML, External Links:
  [Link](https://arxiv.org/abs/2601.18734) Cited by: §3.
- Zheng et al. (2023a) L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z.
  Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E.
  Gonzalez, and I. Stoica Judging llm-as-a-judge with mt-bench and
  chatbot arena. In NeurIPS, Cited by: §4.
- Zheng et al. (2023b) L. Zheng, W. Chiang, Y. S. Ying, S. Hao, S.
  Macavaney, et al. Judging llm-as-a-judge with mt-bench and chatbot
  arena. In NeurIPS, Cited by: §1.
- Zhou et al. (2025) Y. Zhou, H. Liu, Z. Chen, Y. Tian, and B. Chen
  GSM-$`\infty`$: how do your llms behave over infinitely increasing
  reasoning complexity and context length?. In ICML, Cited by: §5.
- Zhu et al. (2023) B. Zhu, J. Jiao, and M. I. Jordan Principled
  reinforcement learning with human feedback from pairwise or $`K`$-wise
  comparisons. arXiv preprint arXiv:2301.11270. Cited by: §1, §3.
- Zhu et al. (2025) F. Zhu, R. Wang, S. Nie, X. Zhang, C. Wu, J. Hu, J.
  Zhou, J. Chen, Y. Lin, J. Wen, and C. Li LLaDA 1.5: variance-reduced
  preference optimization for large language diffusion models. External
  Links: 2505.19223, [Link](https://arxiv.org/abs/2505.19223) Cited by:
  §7.

## Appendix A Policy Gradient Algorithms for RL

### A.1 Proximal Policy Optimization (PPO)

For stability, PPO regularizes the learned policy to stay close to the
reference model $`\pi_{\text{ref}}`$ (e.g., a model obtained by SFT
warm-up), and its total reward takes into account the KL divergence
between the learned model and the reference model:

```math
\displaystyle r_{\text{total}}(s_{t-1},a_{t},s_{t}) \displaystyle=r(s_{t-1},a_{t},s_{t})
```

```math
\displaystyle-\beta\cdot\mathbb{D}_{\mathrm{KL}}\!\left(\pi_{\theta}(\cdot\mid s_{t-1}),\pi_{\text{ref}}(\cdot\mid s_{t-1})\right).
```

Additionally, PPO employs a value model $`V_{\phi}(\cdot)`$ to predict
the expected reward for each state. Architecturally, $`V_{\phi}`$ shares
most parameters with the policy $`\pi_{\theta}`$ by adding a value head
atop the final hidden layer. We perform Temporal Difference (TD)
estimation of state values:

```math
\displaystyle\hat{A}_{t}=\sum_{l=0}^{L-t}(\gamma\lambda)^{l}\delta_{t+l},\quad\text{where}\quad\delta_{t^{\prime}}=-V_{\phi}(s_{t^{\prime}})+
```

```math
\displaystyle r_{\text{total}}(s_{t^{\prime}},a_{t^{\prime}+1},s_{t^{\prime}+1})+\gamma V_{\phi}(s_{t^{\prime}+1})
```

with the terminal state value $`V_{\phi}(s_{L+1}):=0`$,
$`\gamma\in(0,1]`$ is the discount factor for rewards, and
$`\lambda\in[0,1]`$ is the discount factor for TD (Schulman et al.
2018). The final estimated state value for $`s_{t}`$ is
$`\hat{R}_{t}=\hat{A}_{t}+V_{\phi}(s_{t})`$. The loss for step $`t`$
involves two terms, one for the expected reward and the other for value
estimation:

```math
\displaystyle\mathcal{L}_{\text{policy}}^{t}(\theta) \displaystyle=\begin{aligned} &-\mathbb{E}_{e\sim\pi_{\theta_{\text{old}}}}\Bigg[\min\Biggl(\frac{\pi_{\theta}(a_{t}\mid s_{t-1})}{\pi_{\theta_{\text{old}}}(a_{t}\mid s_{t-1})}\hat{A}_{t},\\
&\operatorname{clip}\left(\frac{\pi_{\theta}(a_{t}\mid s_{t-1})}{\pi_{\theta_{\text{old}}}(a_{t}\mid s_{t-1})},1-\epsilon,1+\epsilon\right)\hat{A}_{t}\Biggr)\Bigg]\end{aligned}
```

```math
\displaystyle\mathcal{L}_{\text{value}}^{t}(\phi) \displaystyle=\begin{aligned} &\frac{1}{2}\mathbb{E}_{e\sim\pi_{\theta_{\text{old}}}}\Bigg[\max\Biggl(\left\lVert V_{\phi}(s_{t})-\hat{R}_{t}\right\rVert^{2},\\
&\left\lVert\operatorname{clip}\left(\hat{R}_{t}-V_{\phi}(s_{t}),\hat{A}_{t}-\epsilon,\hat{A}_{t}+\epsilon\right)\right\rVert^{2}\Biggr)\Bigg]\end{aligned}
```

where $`\pi_{\theta_{\text{old}}}`$ and $`V_{\phi_{\text{old}}}`$ denote
earlier models that are used to sample the “rollout” CoTs and compute
$`\hat{A}_{t}`$ and $`\hat{R}_{t}`$. The final RL loss is the weighted
combination:

```math
\displaystyle{\mathcal{L}}_{\text{PPO}}(\theta,\phi)=\sum_{t=1}^{L}\left(\mathcal{L}_{\text{policy}}^{t}+\alpha\mathcal{L}_{\text{value}}^{t}\right)
```

where $`\alpha\geq 0`$ is the coefficient for the value objective.

### A.2 Group Relative Policy Optimization (GRPO)

Due to the success of DeepSeek-R1 (Guo et al. 2025), GRPO is currently
the most popular RL algorithm for reasoning tasks, and one may wonder
how it compares with PPO on our tasks. Here, we briefly review the
algorithm.

A key architectural divergence lies in the absence of a value module in
GRPO. Unlike PPO, which relies on a critic to estimate the advantage and
mitigate the variance of the policy gradient, GRPO eliminates this
requirement. Instead, GRPO samples a group of CoTs for the same
question, denoted as $`\{e^{1},\dots,e^{G}\}`$ where
$`e^{i}=[a_{1}^{i},\dots,a_{|e^{i}|}^{i}]`$, using the old policy model
$`\pi_{\theta_{\text{old}}}`$. These trajectories receive raw rewards
$`\{r^{1},\dots,r^{G}\}`$ at their corresponding terminal states.
Outcome supervision provides the normalized reward that is shared by all
tokens within each CoT:

```math
\displaystyle\hat{A}^{i}=\frac{r^{i}-\text{mean}(r)}{\text{std}(r)}.
```

GRPO optimizes the following objective:

```math
\displaystyle{\mathcal{L}}_{\text{GRPO}}(\theta)=\mathbb{E}_{\{e^{i}\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert e^{i}\rvert}\times
```

```math
\displaystyle\sum_{t=1}^{\lvert e^{i}\rvert}\Biggl\{\min\Biggl[\frac{\pi_{\theta}(a_{t}^{i}\mid s_{t-1}^{i})}{\pi_{\theta_{\text{old}}}(a_{t}^{i}\mid s_{t-1}^{i})}\,\hat{A}^{i},\,
```

```math
\displaystyle\operatorname{clip}\Biggl(\frac{\pi_{\theta}(a_{t}^{i}\mid s_{t-1}^{i})}{\pi_{\theta_{\text{old}}}(a_{t}^{i}\mid s_{t-1}^{i})},1-\varepsilon,\,1+\varepsilon\Biggr)\hat{A}^{i}\Biggr]
```

```math
\displaystyle-\beta\mathbb{D}_{\mathrm{KL}}\bigl[\pi_{\theta}\|\pi_{\text{ref}}\bigr]\Biggr\}.
```

Figure 1: Performance comparison between PPO and GRPO. Learning curves
for Galactica-125M trained on GSM8K. The V+C experiments use GSM8K-aug
as the unsupervised dataset.

### A.3 Comparison of PPO and GRPO

Under the experimental setup described in Section 5, we observe that
GRPO achieves a larger performance improvement from semi-supervised
training compared to PPO, as illustrated by the learning curves in
Figure 1. The final accuracies, summarized in Table 10, demonstrate that
GRPO more effectively leverages the unlabeled dataset (GSM8K-aug) in the
V+C setup, yielding superior final performance across all evaluated
benchmarks.

| Setup | Algorithm | GSM8K | GSM-Plus | GSM-Symbolic |       |      | SVAMP |
| ----- | --------- | ----- | -------- | ------------ | ----- | ---- | ----- |
|       |           |       |          | main         | p1    | p2   |       |
| V     | PPO       | 30.48 | 18.38    | 30.96        | 8.42  | 0.96 | 31.30 |
| V+C   | PPO       | 35.10 | 22.14    | 34.08        | 12.76 | 3.32 | 40.70 |
| V     | GRPO      | 30.33 | 18.00    | 26.30        | 7.98  | 1.36 | 28.50 |
| V+C   | GRPO      | 37.30 | 23.71    | 35.36        | 11.18 | 2.20 | 45.20 |

Table 10: Final accuracies obtained by PPO and GRPO for Galactica-125M.
The V+C experiments use GSM8K-aug as the unsupervised dataset.

## Appendix B Prompts and Templates

In this section, we present all prompts used for training and
evaluation.

### B.1 Task Templates for Student Model

#### PoT for GSM problems

The SFT model completes the code based on the problem description
provided in the docstring.

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-1.svg)

#### CoT for AIME problems

We pass the following message directly to the model and extract the
generated content within `\\boxed{}` for verification.

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-2.svg)

### B.2 Online Judge Reward Generation

To compute the reward, we extract the judge LLM’s specific logits for
the “Yes” and “No” tokens from the output distribution at position
$`\Box`$.

#### Evaluating Generated PoT on GSM Problems

We instruct the judge LLM to evaluate candidate trajectories (both
reasoning and final answer) using a prompt that requires a binary
response. Note that the Python code is not executed, but rather
statically evaluated by the LLM.

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-3.svg)

#### Evaluating Generated CoT on GSM Problems

We use the following prompt. The final answer string is included within
the “response” block.

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-4.svg)

#### Evaluating Generated CoT on AIME Problems

We use the following prompt. The final answer string is included within
the “response” block.

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-5.svg)

### B.3 Offline Pseudo-label Generation

#### GSM8K-aug Base 1

The following prompt corresponds to Base 1 and is closely comparable to
our method. Here, we instruct the teacher to generate the answer at
position $`\square`$, which we decode and use as the pseudo-label.

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-6.svg)

### B.4 GPT-5.6 Generated RaR

The generated Rubrics-as-Rewards (RaR) criteria for GSM and AIME are
presented in Table 11 and Table 12, respectively.

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-7.svg)

Table 11: GPT-5.6 Generated RaR criteria for GSM.

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-8.svg)

Table 12: GPT-5.6 Generated RaR criteria for AIME.

## Appendix C More Details on Different Reward Formulations

We use one NVIDIA RTX Pro 6000 GPU for every experiment. We sample 200
problems and solutions from Galactica-125M over GSM8K and SVAMP using
the same random seed, respectively. For AIME, we sample solutions from
Qwen3-0.6B, and select problems that have at least two correct and one
incorrect answers. (We need two correct so that we have a reference
answer for Likert score.) In total, there are 78 such AIME problems.

In Table 13 and 14, we compare the time it takes to obtain different
reward signals and compare them in Section 4. Our CCR method is the
fastest as it requires single-token inference. PRR, while also fast,
requires pretrained classifier model and may not be generalized to
substantially different tasks.

|                |       |       |
| -------------- | ----- | ----- |
| Judge          | GSM8K | SVAMP |
| Likert         |       |       |
| Qwen3-8B       | 360   | 204   |
| Llama3.1-8B    | 1098  | 587   |
| Gemma3-12B     | 760   | 456   |
| RaR            |       |       |
| Qwen3-8B       | 152   | 196   |
| Llama3.1-8B    | 444   | 483   |
| Gemma3-12B     | 351   | 337   |
| PRR            |       |       |
| Galactica-6.7B | 56    | 40    |
| LL             |       |       |
| Qwen3-8B       | 93    | 67    |
| Llama3.1-8B    | 89    | 62    |
| Gemma3-12B     | 129   | 101   |
| CCR            |       |       |
| Qwen3-8B       | 49    | 41    |
| Llama3.1-8B    | 53    | 38    |
| Gemma3-12B     | 82    | 68    |

Table 13: Runtime (seconds) on Grade School-level Math.

| Judge Candidate | CCR | LL  | RaR | Likert |
| --------------- | --- | --- | --- | ------ |
| Qwen3-8B        | 16  | 23  | 68  | 82     |
| Qwen3.5-9B      | 34  | 48  | 117 | 153    |
| Gemma3-12B      | 29  | 36  | 119 | 162    |

Table 14: Runtime (seconds) on AIME.

## Appendix D Implementation Details for Grade School Math

Experiments for small LMs (Galactica-125M and CodeGen-350M) are
performed on 4 NVIDIA RTX Pro 6000 GPUs.

### D.1 Data Processing

#### Sampling.

We created 100 completions/responses from 200 randomly selected test
questions, using the SFT warm-up model and same seed for each test set.
We select one correct completion and one incorrect completion and use
the LLM to score the correct completion and the incorrect completion.

#### Training Data.

The grade school-level math reasoning datasets (GSM8K and SVAMP) are
directly obtained from the ReFT repository (Trung et al. 2024). The
augmented version of GSM8K (GSM8K-aug) was originally created by Deng et
al. (2024), and we utilize the processed version available on Hugging
Face at whynlp/gsm8k-aug. For our ablation study regarding the amount of
augmented data, we randomly sample 5% (19,273 examples) and 25% (96,365
examples) of the full GSM8K-aug dataset.

#### Pseudo-label Base 1.

We include a baseline where the judge LM is used to generate
pseudo-labels for GSM8K-aug, enabling us to train the RLVR baseline on
this dataset (see Appendix B for the specific prompts and templates
used). This generation process yielded 384,441 valid pseudo-labeled
samples with 49.33% of the completed answers matching the original
labels of GSM8K-aug, while 1,020 invalid samples were discarded.

### D.2 Models

All models are loaded from Hugging Face. The student models are
facebook/galactica-125m and codegen-350M-mono. The CCR judge model is
Qwen/Qwen3-8B. The pretrained reranking model is
galactica-6.7b-ReFT-Rerank-GSM8k.

### D.3 Hyperparameters

#### Training Steps.

Because we employ two distinct dataloaders for our distillation
experiments (one for labeled data and one for unlabeled data), we track
optimizer update steps rather than epochs. Furthermore, since the
GSM8K-aug dataset is exceptionally large, a single epoch over the
unlabeled data would take at least six days, making epoch counting
impractical. For experiments trained on GSM8K, we trained Galactica-125M
for 100,000 optimizer steps and CodeGen-350M for 50,000 steps. For the
Galactica-6.7B experiments continuing from a checkpoint, we ran 15,000
optimizer steps. For all experiments on SVAMP, training proceeded for
50,000 steps.

#### Supervised Fine-Tuning (SFT).

To warm-up a model, we train it for 10 epochs with a global batch size
of 24. For complete SFT runs, we train the model for 40 epochs, also
with a global batch size of 24. A learning rate of $`2\times 10^{-5}`$
is used in both scenarios.

#### RL-Based Distillation.

All small models are initialized from their warm-up checkpoints and
trained using a global batch size of 24. The number of PPO epochs is set
to 2. The value function coefficient $`\alpha`$ is set to 5 for
Galactica-125M and 0.1 for CodeGen-350M. The KL coefficient $`\beta`$ is
set to 0.01 across all Program-of-Thought (PoT) tasks.

#### CCR Reward.

We set $`\tau=1`$ and $`a=0.35`$.

#### GRPO.

We use a global batch size of 16 and sample 8 hypotheses ($`G=8`$) per
question to calculate advantages. All other settings remain identical to
those used in PPO.

#### Large Model Experiment.

We use a global batch size of 16 and sample 100 hypotheses per question
for majority voting.

#### Overall Objective.

While the reward functions could theoretically be combined into a single
PPO loss, we maintain three separate losses since we draw from two
different dataloaders (labeled and unlabeled). To ensure fair
comparisons across methods, we record progress strictly in optimizer
update steps rather than data-specific metrics.

## Appendix E Additional Results on GSM

VR

CCR

PRR

Figure 2: All three types of our active training reward on
Galactica-125M in a semi-supervised setup.

Figure 3: Evaluation Accuracy of Galactica-125M on GSM8K during RL
training.

The three curves in Figure 2 demonstrate that all three reward types
increase steadily during training before eventually plateauing. These
curves correspond to our semi-supervised setup utilizing Verifiable
Reward (VR), Continuous CoT Reward (CCR), and Pretrained Reranker (PRR)
on the combined GSM8K and GSM8K-aug datasets.

Furthermore, Figure 3 shows that semi-supervised RL, when combined with
diverse reward signals (CCR and PRR), yields the highest overall
performance.

Table 15 presents an ablation study on the loss coefficients
$`\lambda`$, $`\mu`$, and $`\rho`$. It contains additional results for
using the GSM8K as unsupervised data for combining VR and CRR rewards.

|                                 |         |          |       |          |              |       |      |       |
| ------------------------------- | ------- | -------- | ----- | -------- | ------------ | ----- | ---- | ----- |
| Coefficients                    |         |          | GSM8K | GSM-Plus | GSM-Symbolic |       |      | SVAMP |
| $`\lambda`$                     | $`\mu`$ | $`\rho`$ |       |          | main         | p1    | p2   |       |
| Reported by Trung et al. (2024) |         |          |       |          |              |       |      |       |
| SFT Complete                    |         |          | 23.70 | —        | —            | —     | —    | —     |
| 1.0                             | –       | –        | 29.80 | —        | —            | —     | —    | —     |
| Our reproduction                |         |          |       |          |              |       |      |       |
| SFT Warm-up                     |         |          | 21.83 | 12.49    | 20.24        | 4.88  | 1.80 | 29.50 |
| SFT Complete                    |         |          | 25.25 | 15.14    | 22.88        | 7.40  | 3.64 | 32.30 |
| 1.0                             | 0.0     | 0.0      | 30.48 | 18.38    | 30.96        | 8.42  | 0.48 | 31.30 |
| GSM8K unsup                     |         |          |       |          |              |       |      |       |
| 0.0                             | 1.0     | 0.0      | 29.19 | 17.95    | 26.42        | 11.70 | 4.72 | 37.00 |
| GSM8K sup + GSM8K unsup         |         |          |       |          |              |       |      |       |
| 0.5                             | 0.5     | 0.0      | 30.63 | 18.96    | 28.32        | 9.64  | 1.40 | 35.90 |
| Pseudo-labeled GSM8K-aug sup    |         |          |       |          |              |       |      |       |
| Base 1                          |         |          |       |          |              |       |      |       |
| 1.0                             | 0.0     | 0.0      | 28.89 | 16.78    | 25.00        | 6.48  | 2.16 | 44.60 |
| Base 2                          |         |          |       |          |              |       |      |       |
| 1.0                             | 0.0     | 0.0      | 36.01 | 26.64    | 37.20        | 14.58 | 1.96 | 51.00 |
| GSM8K-aug unsup                 |         |          |       |          |              |       |      |       |
| 0.0                             | 1.0     | 0.0      | 33.13 | 21.28    | 30.38        | 11.36 | 3.32 | 44.70 |
| 0.0                             | 0.0     | 1.0      | 32.98 | 20.92    | 31.12        | 10.36 | 2.60 | 46.30 |
| 0.0                             | 0.5     | 0.5      | 40.71 | 25.33    | 37.74        | 14.62 | 3.00 | 50.70 |
| GSM8K sup + GSM8K-aug unsup     |         |          |       |          |              |       |      |       |
| 0.5                             | 0.5     | 0.0      | 35.10 | 22.14    | 34.08        | 12.76 | 3.32 | 40.70 |
| 0.5                             | 0.0     | 0.5      | 39.27 | 25.04    | 36.86        | 12.84 | 1.44 | 43.30 |
| 0.5                             | 0.25    | 0.25     | 40.11 | 25.26    | 37.50        | 15.70 | 2.24 | 42.60 |
| GSM8K sup + GSM-Plus unsup      |         |          |       |          |              |       |      |       |
| 0.5                             | 0.5     | 0.0      | 41.70 | 28.97    | 35.52        | 12.78 | 1.12 | 37.90 |
| 0.5                             | 0.25    | 0.25     | 44.66 | 30.67    | 38.74        | 10.88 | 0.84 | 32.50 |

Table 15: Galactica-125M trained on GSM8K. Results with coefficients.

## Appendix F Implementation Details for Competition-level Math

Experiments are performed on eight NVIDIA RTX Pro A6000 GPUs.

### F.1 Models and Datasets

The student model is Qwen/Qwen3-0.6B-Thinking, and the CCR judge model
is Qwen/Qwen3.5-9B.

For training, we utilize the di-zhang-fdu/AIME_1983_2024 and
BytedTsinghua-SIA/DAPO-Math-17k datasets from Hugging Face. For
evaluation, we use math-ai/aime24, math-ai/aime25, and math-ai/aime26.

### F.2 Hyperparameters

#### CCR Reward.

We set $`\tau=0.5`$ and $`a=0.1`$.

#### GRPO Settings.

We employ the Group Relative Policy Optimization (GRPO) algorithm via
the TRL and vLLM libraries. Across all experiments, we set the KL
penalty coefficient $`\beta=0`$ and the group size $`G=7`$. In our
semi-supervised setup, we leverage TRL’s ability to assign sampling
probabilities to different reward formulations, utilizing a
$`\lambda=0.1`$ weight for supervised data and a $`\mu=0.9`$ weight for
unsupervised data. The model is trained for approximately one epoch.
Table 16 contains an ablation study on the loss coefficients $`\lambda`$
and $`\rho`$ in a semi-supervised setup.

| $`\lambda`$ | $`\mu`$ | AIME’24 | AIME’25 | AIME’26 |
| ----------- | ------- | ------- | ------- | ------- |
| 0.1         | 0.9     | 16.67   | 20.73   | 15.21   |
| 0.5         | 0.5     | 11.25   | 16.88   | 11.87   |
| 0.9         | 0.1     | 15.42   | 19.06   | 14.90   |

Table 16: Avg@32 results of Qwen3-0.6B-thinking on AIME with
semi-supervised RL.

#### Optimization and System Details.

To accelerate training, we apply LoRA fine-tuning targeting all-linear
modules with rank $`r=8`$ and scaling factor $`\alpha=16`$. We use a
global batch size of 56, distributed across 7 GPUs with 8 gradient
accumulation steps and a local batch size of 1. The learning rate is set
to $`5\times 10^{-5}`$. We dedicate a single GPU exclusively for vLLM
inference. The maximum number of completion tokens is set to 20,000 in
TRL and 21,600 in vLLM.

## Appendix G Model Collapse in Unsupervised Training

Reward hacking has been reported in existing RL literature (Uehara et
al. 2024; Skalse et al. 2025). It is a phenomenon where the nominal (or
proxy) reward values are high (e.g., the aesthetic score of a generated
image, the plausibility of a generated code snippet), while the
generation itself does not make sense. In our experiments, this may
occur without VR. For example, as seen in Figure 5, the hacked response
does not generate any code (or in some cases, generates a number with no
reasoning whatsoever and returns directly), but the reward is much
higher than the incorrect response with reasonable code. As we further
demonstrate in Figure 4, the transition from a stable state to a failure
state happens quickly. Empirically, this phenomenon can be avoided by
using VR together with CCR, showing the strength of our semi-supervised
setup. Although training without ground-truth labels is not the focus of
this work, below we show some strategies to make unsupervised training
more stable.

#### Mitigation Strategies

We present some intuitive mitigation strategies when CCR is the only
available reward signal (fully unsupervised), but more rigorous
treatments are left for future exploration. Occasionally, reward hacking
persists on the unsupervised dataset even when the supervised dataset
yields reasonable signals. This may lead to model collapse or
drastically slow down training. To alleviate this, we (1) explicitly
instruct the small LMs at the end of the docstring: “Please provide full
reasoning before computing the result; do not answer with the result
only.” and (2) apply a mask so that if a generated code snippet is fewer
than 4 lines, we assign a judge reward of 0. This modification is only
applied to the GSM8K sup + GSM-Plus unsup setting using CodeGen in
Table 7. However, these patches may not fully resolve the issue, since
our modifications occasionally led to consistent zero rewards and,
consequently, model collapse. They also cause performance degradation in
runs that do not exhibit reward hacking.

The instance of reward hacking reported in Trung et al. (2024) on the
MathQA-Multiple Choice dataset, where the model reports the correct
answer option despite using incorrect reasoning, is different from our
encounter, where the model provides little to no response without any
reasoning. Nevertheless, both scenarios achieve high rewards with
low-quality reasoning. Future work on new regularization methods is
therefore necessary.

Figure 4: Judge Mean Reward Curve

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-9.svg)

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-10.svg)

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-11.svg)

![](arxiv-2604-02621v2--5aaedcfda30b.figures/inline-12.svg)

Figure 5: Example completions with CCR scores only. Notice how the
hacked response does not generate anything given the declaration and
docstring, yet obtaining a good score.
