---
identifier: arxiv:2402.14016v2
title: Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment
authors:
  - Vyas Raina
  - Adian Liusie
  - Mark Gales
published: "2024-02-21T18:55:20+00:00"
url: https://arxiv.org/abs/2402.14016v2
source: arxiv
doi: null
arxiv_id: 2402.14016v2
categories:
  - cs.CL
---

# Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment

Vyas Raina^(∗) Affiliation: University of Cambridge
Email: [vr313@cam.ac.uk](mailto:)    Adian Liusie^(∗)
Affiliation: University of Cambridge Email: [al826@cam.ac.uk](mailto:)
   Mark Gales Affiliation: University of Cambridge
Email: [mjfg@cam.ac.uk](mailto:)

###### Abstract

Large Language Models (LLMs) are powerful zero-shot assessors used in
real-world situations such as assessing written exams and benchmarking
systems. Despite these critical applications, no existing work has
analyzed the vulnerability of judge-LLMs to adversarial manipulation.
This work presents the first study on the adversarial robustness of
assessment LLMs, where we demonstrate that short universal adversarial
phrases can be concatenated to deceive judge LLMs to predict inflated
scores. Since adversaries may not know or have access to the judge-LLMs,
we propose a simple surrogate attack where a surrogate model is first
attacked, and the learned attack phrase then transferred to unknown
judge-LLMs. We propose a practical algorithm to determine the short
universal attack phrases and demonstrate that when transferred to unseen
models, scores can be drastically inflated such that irrespective of the
assessed text, maximum scores are predicted. It is found that judge-LLMs
are significantly more susceptible to these adversarial attacks when
used for absolute scoring, as opposed to comparative assessment. Our
findings raise concerns on the reliability of LLM-as-a-judge methods,
and emphasize the importance of addressing vulnerabilities in LLM
assessment methods before deployment in high-stakes real-world
scenarios. ¹¹ 1 Code:
[https://github.com/rainavyas/attack-comparative-assessment](https://github.com/rainavyas/attack-comparative-assessment)

^(†)^(†) ^(∗) Equal Contribution.

## 1 Introduction

Large Language Models (LLMs) have shown to be proficient zero-shot
assessors, capable of evaluating texts without requiring any
domain-specific training Zheng et al. (2023); Chen et al. (2023); Zhang
et al. (2023a). Typical zero-shot approaches prompt powerful LLMs to
either generate a single quality score of the assessed text Wang et al.
(2023a); Liu et al. (2023b) or to use pairwise comparisons to determine
which of two texts are better Liusie et al. (2023); Qin et al. (2023).
These zero-shot approaches mark a compelling new paradigm for
assessment, enabling straightforward reference-free evaluation that
correlates highly with human judgements, while being applicable to a
range of diverse attributes. There has consequently been a surge of
leveraging LLM-as-a-judge in many applications, including as benchmarks
for assessing new models Zheng et al. (2023); Zhu et al. (2023b) or as
tools for assessing the written examinations of real candidates.

![](arxiv-2402-14016v2--7ea2cea7fcba.figures/figure-1.webp)

Figure 1: A simple universal adversarial attack phrase can be
concatenated to a candidate response to fool an LLM assessment system
into predicting that it is of higher quality. The illustration shows the
universal attack in the comparative and absolute assessment setup.

Despite the clear advantages of zero-shot LLM assessment methods, the
limitations and robustness of LLM-as-a-judge have been less
well-studied. Previous works have demonstrated potential limitations in
robustness, and the presence of biases such as positional bias Wang
et al. (2023b); Liusie et al. (2023); Zhu et al. (2023b), length bias
Koo et al. (2023) and self-preferential behaviours Zheng et al. (2023);
Liu et al. (2023d). This paper pushes this paradigm further by
investigating whether appending a simple universal phrase to the end of
an assessed text could deceive an LLM into predicting high scores
regardless of the text’s quality. Such approaches not only pose
challenges for model evaluation, where adversaries may manipulate
benchmark metrics, but also raise concerns about academic integrity, as
students may employ similar tactics to cheat and attain higher scores.

This work is the first to propose adversarial attacks Szegedy et al.
(2014) targeting zero-shot LLM assessment. In practical settings, the
adversary may either not have any knowledge of the judge-LLMs, access to
the model weights, or be limited in the number of queries that can be
made to the model (due to costs or suspicion from excessive querying).
Therefore, we learn the attack phrase while using a surrogate model
 (Papernot et al., 2016) and transfer the universal attack phrase to
other judge-LLMs. We demonstrate that universal attack phrases learned
with access only to FlanT5-3B model, a small encoder-decoder
transformer, can transfer to larger decoder-only models and cause
Llama2-7B, Mistral-7B and ChatGPT to return the maximum score,
irrespective of the input text. We find that LLM-scoring (as opposed to
pairwise LLM-comparative assessment) can be particularly vulnerable to
such attacks, and concatenating a universal phrase of just 5 tokens can
trick these systems into providing highly increased assessment scores.
Additionally, we find that comparative assessment is more robust than
LLM-scoring to such adversarial attacks, although the direct attacks on
the surrogate model can yield marginally inflated scores. Finally, as an
initial step towards defending against such attacks, we use the
perplexity score (Jain et al., 2023) as a simple detection approach,
which demonstrates some success. As a whole, our work raises awareness
of the vulnerabilities of zero-shot LLM assessment, and highlights that
if such systems are to be deployed in critical real-world scenarios,
adversarial vulnerabilities should be considered and addressed.

## 2 Related Work

#### Bespoke NLG Evaluation.

For Natural Language Generation tasks such as summarization or
translation, traditional assessment metrics evaluate generated texts
relative to gold standard manual references Lin (2004); Banerjee and
Lavie (2005); Zhang et al. (2019). These methods, however, tend to
correlate weakly with human assessments. Following work designed
automatic evaluation system systems for particular domains and
attributes. Examples include systems for dialogue assessment Mehri and
Eskenazi (2020), question answering systems for summary consistency Wang
et al. (2020); Manakul et al. (2023), boolean answering systems for
general summary assessment Zhong et al. (2022a) or neural frameworks for
machine translation Rei et al. (2020).

#### Zero-Shot Assessment with LLMs.

Although suitable for particular domains, these automatic evaluation
methods cannot be applied to more general and unseen settings. With the
rapidly improving ability of instruction-following LLMs, various works
have proposed zero-shot approaches. These include prompting LLMs to
provide absolute assessment scores Wang et al. (2023a); Liu et al.
(2023b), comparing pairs of texts Liusie et al. (2023); Zheng et al.
(2023) or through leveraging assigned output language model
probabilities Fu et al. (2023), and in some cases demonstrating
state-of-the-art correlations and outperforming performance of bespoke
evaluation methods.

#### Adversarial Attacks on Generative Systems.

Traditionally, NLP attack literature focuses on attacking classification
tasks (Alzantot et al., 2018; Garg and Ramakrishnan, 2020; Li et al.,
2020; Gao et al., 2018; Wang et al., 2019). However, with the emergence
of generative LLMs (Zhao et al., 2023), there has been discussion around
NLG adversarial attacks. A range of approaches seek to jailbreak LLMs,
and circumvent inherent alignment to generate harmful content (Carlini
et al., 2023). Attacks can be categorized as input text perturbation
optimization (Zou et al., 2023; Zhu et al., 2024; Lapid et al., 2023);
automated adversarial prompt learning (Mehrotra et al., 2023; Liu
et al., 2023a; Chao et al., 2023; Jin et al., 2024); human adversarial
prompt learning (Wei et al., 2023; Zeng et al., 2024; Liu et al.,
2023c); or model configuration manipulation (Huang et al., 2024). Beyond
jailbreaking, other works look to extract sensitive data from LLMs (Nasr
et al., 2023; Carlini et al., 2020), provoke misclassification (Zhu
et al., 2023a) or trick translation systems into making a change in
perception (Raina and Gales, 2023; Sadrizadeh et al., 2023). For
assessment, although early research has explored attacking NLP
assessment systems (Raina et al., 2020), there has been no work on
developing attacks for general LLM assessment models such as prompting
LLama and GPT, and we are the first to conduct such a study.

## 3 Zero-shot Assessment with LLMs

As discussed by Zhu et al. (2023b); Liusie et al. (2023), there are two
standard reference-free methods of prompting instruction-tuned LLMs for
quality assessment:

- •
  LLM Comparative Assessment where the system uses pairwise comparisons
  to determine which of two responses are better.
- •
  LLM Scoring where an LLM is asked to assign an absolute score to each
  considered text.

For various assessment methods, we consider rankings tasks where given a
query context $`\mathbf{d}`$ and a set of $`N`$ responses
$`\mathbf{x}_{1:N}`$, the objective is to determine the quality of each
response, $`s_{1:N}`$. An effective LLM judge should predict scores for
each candidate that match the ranking $`r_{1:N}`$ of the text’s true
quality. This section will further discuss the details of both
comparative assessment (Section 3.1) and absolute assessment (Section
3.2).

### 3.1 Comparative Assessment

An LLM prompted for comparative assessment, $`\mathcal{F}`$, can be used
to determine the probability that the first candidate is better than the
second. Given the context $`\mathbf{d}`$ and two candidate responses,
$`\mathbf{x}_{i}`$ and $`\mathbf{x}_{j}`$, to account for positional
bias Liusie et al. (2023); Wang et al. (2023b) one can run comparisons
over both orderings and average the probabilities to predict the
probability that response $`\mathbf{x}_{i}`$ is better than response
$`\mathbf{x}_{j}`$,

```math
p_{ij}=\frac{1}{2}\big(\mathcal{F}(\mathbf{x}_{i},\mathbf{x}_{j},\mathbf{d})+(1-\mathcal{F}(\mathbf{x}_{j},\mathbf{x}_{i},\mathbf{d}))\big) \tag{1}
```

Note that by doing two inference passes of the model, symmetry is
ensured such that $`p_{ij}=1-p_{ji}`$ for all $`i,j\!\in\!\{1,...,N\}`$.
The average comparative probability for each option $`\mathbf{x}_{n}`$
can then be used as the predicted quality score $`\hat{s}_{n}`$,

```math
\hat{s}_{n}=\hat{s}(\mathbf{x}_{n})=\frac{1}{N}\sum_{j=1}^{N}p_{nj}, \tag{2}
```

which can be converted to ranks $`\hat{r}_{1:N}`$, that can be evaluated
against the true ranks $`r_{1:N}`$.

### 3.2 Absolute Scoring Assessment

In LLM absolute scoring, the LLM, $`\mathcal{F}`$, is prompted to
directly predict the assessment score. The prompt is designed to request
the LLM to assess the quality of a text with a score (e.g. between 1-5).
Two variants of scoring can be applied; first where the score is
directly predicted by the LLM,

```math
\hat{s}_{n}=\hat{s}(\mathbf{x}_{n})=\mathcal{F}(\mathbf{x}_{n},\mathbf{d}). \tag{3}
```

Alternatively, following G-Eval Liu et al. (2023b), if the output logits
are accessible one can estimate the expected score through a
fair-average by multiplying each score by its normalized probability,

```math
\hat{s}_{n}=\hat{s}(\mathbf{x}_{n})=\sum_{k=1:K}kP_{\mathcal{F}}(k|\mathbf{x}_{n},\mathbf{d}), \tag{4}
```

where $`K`$ is the maximum score, as indicated in the prompt, and the
probability for each possible score $`k\!\in\!\{1,...,K\}`$ is
normalized to satisfy basic probability rules,
$`\sum_{k}P_{\mathcal{F}}(k|\mathbf{x}_{n},\mathbf{c})=1`$ and
$`P_{\mathcal{F}}(k|\mathbf{x}_{n},\mathbf{c})\geq 0`$, $`\forall n`$.

## 4 Adversarial Assessment Attacks

### 4.1 Attack Threat Model

#### Objective.

For typical adversarial attacks, an adversary aims to minimally modify
the input text $`{\mathbf{x}}\rightarrow{\mathbf{x}+\bm{\delta}}`$ in an
attempt to manipulate the system’s response. The adversarial example
$`{\bm{\delta}}`$ is a small perturbation on the input $`\mathbf{x}`$,
designed to cause a significant change in the output prediction of the
system, $`\mathcal{F}`$,

```math
\mathcal{F}(\mathbf{x}+\bm{\delta})\neq\mathcal{F}(\mathbf{x}), \tag{5}
```

The small perturbation, $`+\bm{\delta}`$, is constrained to have a small
difference in the input text space, measured by a proxy function of
human perception,
$`\mathcal{G}(\mathbf{x},\mathbf{x}+\bm{\delta})\leq\epsilon`$. Our work
considers applying simple concatenative attacks to assessment LLMs,
where a phrase $`\bm{\delta}`$ of length $`L\!\ll\!|\mathbf{x}|`$ is
added to the original text $`\mathbf{x}`$,

```math
\mathbf{x}+\bm{\delta}=x_{1},\ldots,x_{|\mathbf{x}|},\delta_{1},\ldots,\delta_{L} \tag{6}
```

The attack objective is to then maximally improve the rank of the
attacked candidate response with respect to the other candidates. Let
$`\hat{r}^{\prime}_{i}`$ represent the rank of the attacked response,
$`\mathbf{x}_{i}+\bm{\delta}`$, when no other response in
$`\mathbf{x}_{1:N}`$ is perturbed,

```math
\hat{r}^{\prime}_{i}(\bm{\delta})=\texttt{rank}_{i}\left(\hat{s}(\mathbf{x}_{1}),\ldots,\hat{s}(\mathbf{x}_{i}+\bm{\delta}),\ldots,\hat{s}(\mathbf{x}_{N})\right)
```

The adversarial objective is to minimize the predicted rank of candidate
$`i`$ (i.e. the attacked sample) relative to the other unattacked
candidates,

```math
\bm{\delta}^{*}_{i}=\argmin_{\bm{\delta}}(\hat{r}^{\prime}_{i}(\bm{\delta})). \tag{7}
```

#### Universal Attack.

In an assessment setting, it is impractical for adversaries to learn an
adversarial example $`\bm{\delta}^{*}_{i}`$ for each candidate response
$`\mathbf{x}_{i}`$. Much more practical is to use a universal
adversarial example $`\bm{\delta}^{*}`$ that could be applied to any
candidate’s response $`\mathbf{x}_{i}`$ to consistently boost the
predicted assessment rank. Assuming a training set of $`M`$ samples of
contexts and $`N`$ candidate responses per context,
$`\{(\mathbf{d}^{(m)},\mathbf{x}^{(m)}_{1:N})\}_{m=1}^{M}`$, the optimal
universal adversarial example $`\bm{\delta}^{*}`$ is the one that most
improves the expected rank when attacking each candidate in turn,

```math
\displaystyle\bar{r}(\bm{\delta}) \displaystyle=\frac{1}{NM}\sum_{m}\sum_{n}\hat{r}^{\prime(m)}_{n}(\bm{\delta}). \tag{8}
```

```math
\displaystyle\bm{\delta}^{*} \displaystyle=\argmin_{\bm{\delta}}(\bar{r}(\bm{\delta})) \tag{9}
```

where the average is computed over all $`M`$ contexts and $`N`$
candidates.

#### Surrogate Model Transfer Attack.

Traditional adversarial attack methods often assume full access to the
target model, but this setting might be unrealistic when attacking
assessment systems. Hence, we consider the more practical scenario where
the adversary only has full access to a surrogate model that differs
from the actual judge-LLM used by the assessment system. The attack can
be learned on the surrogate model and then transferred to the target
model as initially proposed by Liu et al. (2016); Papernot et al.
(2016). The assumption is that due to possible similarities in training
data, training recipes and model architectures, the attacks may transfer
reasonably to the target model.

### 4.2 Practical Attack Approach

In this work, we use a simple greedy search to learn the universal
attack phrase ²² 2 We also carried out experiments using the Greedy
Coordinate Gradient (GCG) attack (Zou et al., 2023) to learn the
universal attack phrase, but this approach was found to be not as
effective as the greedy search process. Results for GCG experiments are
provided in Appendix E.. For a vocabulary, $`\mathcal{V}`$ the greedy
search finds the most effective adversarial word to append iteratively,

```math
\displaystyle\delta^{*}_{l+1}=\argmin_{\delta\in\mathcal{V}}(\bar{r}(\delta^{*}_{1:l}+\delta)). \tag{10}
```

In practice, it may be computationally too expensive to compute the
average rank (as specified in Equation 8). Therefore, we instead
approximate the search by greedily finding the token that maximises the
expected score when appended to the current sample,

```math
\displaystyle\delta^{*}_{l+1} \displaystyle=\argmax_{\delta}\mathbbm{E}_{\mathbf{x}}[\hat{s}(\mathbf{x}+\delta^{*}_{1:l}+\delta)]
```

The algorithm for the practical greedy search attack on comparative
assessment and absolute assessment systems is given in Algorithm 1.

$`\left\{(\mathbf{c}^{(m)},\mathbf{x}^{(m)}_{1:N})\right\}_{m=1}^{M}`$
$`\triangleright`$ Training Data

$`\mathcal{F}()`$ $`\triangleright`$ Target Model

$`\bm{\delta}^{*}\leftarrow\text{ empty string}`$

for $`l=1:L`$ do

  $`a,b\sim\{1,...,N\}`$ $`\triangleright`$ Select candidate indices

  $`\delta^{*}_{l}\leftarrow\text{ none}`$

  $`q^{*}\leftarrow 0`$ $`\triangleright`$ Initialize best score

  for $`\delta\in\mathcal{V}`$ do

   $`\bm{\delta}\leftarrow\bm{\delta}^{*}+\delta`$ $`\triangleright`$
trial attack phrase

   $`q\leftarrow 0`$

   for $`m=1:M`$ do

     if comparative then

      $`p_{1}\leftarrow\mathcal{F}(\mathbf{x}^{(m)}_{a}+\bm{\delta},\mathbf{x}^{(m)}_{b},\mathbf{c}^{(m)})`$

      $`p_{2}\leftarrow\mathcal{F}(\mathbf{x}^{(m)}_{a},\mathbf{x}^{(m)}_{b}+\bm{\delta},\mathbf{c}^{(m)})`$

      $`q\leftarrow q+p_{1}+(1\!-\!p_{2})`$

     else if scoring then

      $`s\leftarrow\mathcal{F}(\mathbf{x}^{(m)}_{a}+\bm{\delta},\mathbf{c}^{(m)})`$

      $`q\leftarrow q+s`$

     end if

   end for

   if $`q>q^{*}`$ then

     $`q^{*}\leftarrow q`$

     $`\delta^{*}_{l}\leftarrow\delta`$ $`\triangleright`$ Update best
attack word

   end if

  end for

  $`\bm{\delta}^{*}\leftarrow\bm{\delta}^{*}+\delta^{*}_{l}`$
$`\triangleright`$ Update attack phrase

end for

Algorithm 1 Greedy Search Universal Attack for LLM Comparative
Assessment LLM and Scoring

## 5 Experimental Setup

### 5.1 Datasets

We run experiments on two standard language generation evaluation
benchmark datasets. The first dataset used is SummEval Fabbri et al.
(2021), which is a summary evaluation benchmark of 100 passages, with 16
machine-generated summaries per passage. Each summary is evaluated by
human assessors on coherency (COH), consistency (CON), fluency (FLU) and
relevance (REL). These attributes can be combined into an overall score
(OVE), which is the average of all the individual attributes. The second
dataset is TopicalChat Gopalakrishnan et al. (2019), which is a
benchmark for dialogue evaluation. There are 60 dialogue contexts, where
each context has 6 different machine-generated responses. The responses
are assessed by human evaluators on coherency (COH), continuity (CNT),
engagingness (ENG), naturalness (NAT), where again the overall score
(OVE) can be computed as the average of the individual attributes.

### 5.2 LLM Assessment Systems

We consider a range of standard instruction-tuned generative language
models that can be used as judge-LLMs: FlanT5-xl (3B parameters) Chung
et al. (2022), Llama2-7B-chat Touvron et al. (2023), Mistral-7B-chat
Jiang et al. (2023), and GPT3.5 (175B parameters). FlanT5-xl, the
smallest and the only encoder-decoder system, is used as the surrogate
model for learning the universal adversarial attack phrases for both
comparative and absolute assessment. Once the attack phrases are learned
on FlanT5-xl, they are transferred to the other target LLMs to evaluate
their effectiveness. Our prompts for comparative assessment follow the
prompts used in Liusie et al. (2023), where different attributes use
different adjectives in the prompt. For absolute assessment, we follow
the prompts of G-Eval Liu et al. (2023b) and use continuous scores
(Equation 4) by calculating the expected score over a score range (e.g.,
1-5 normalized by their probabilities). Note that the GPT3.5 API does
not provide token probabilities, so for GPT3.5, we use standard prompts
without token probability normalization.

### 5.3 Methodology

Each dataset is split into a development set and a test set following a
20:80 ratio. We use the development set (20% of the passages) to learn
the attack phrase using a simple greedy search to maximize the expected
score of the attacked samples and evaluate using the test set (80% of
the passages). Furthermore, we only use two of the candidate texts to
learn the attacks (i.e., 2 of 16 for SummEval and 2 of 6 for
TopicalChat), and therefore perform the search over a modest total of 40
summaries for SummEval and 24 responses for TopicalChat.

For each dataset and attribute, we perform a separate universal
concatenation attack using the notation (TASK ASSESSMENT ATTRIBUTE) to
indicate the task (SummEval, TopicalChat), the assessment method
(comparative, scoring), and the evaluation attribute (overall,
consistency, continuity) for each learned universal attack phrase ³³ 3
The learned universal attack phrases for each configuration are given in
Appendix A.. E.g., SUMM-COMP-OVE denotes the phrase learned for
comparative assessment when attacking the SummEval overall score.

We learn a single universal attack phrase on the surrogate model,
FlanT5-xl, for all experiments in the main paper. Once the universal
attack phrases are learned on the surrogate model, the attack is further
assessed when transferred to the other target models: Mistral-7B,
Llama2-7B, and GPT3.5. The vocabulary for the greedy attack is sourced
from the NLTK python package ⁴⁴ 4 English words corpus is sourced from:
nltk.corpus.

### 5.4 Attack Evaluation

To assess the success of an attack phrase, and for comparing the
performance between comparative and absolute, we calculate the average
rank of each candidate after an attack is applied (Equation 8). An
unsuccessful attack will yield a rank near the average rank, while a
very strong attack will provide an average rank of 1 (where each
attacked candidate is assumed to be the best of all unattacked
candidates of the context).

## 6 Results

(a) SummEval

(b) TopicalChat

Figure 2: Universal attack evaluation (average rank of attacked
summary/response) for surrogate FlanT5-xl.

### 6.1 Assessment Performance

|             |            |      |      |      |      |
| ----------- | ---------- | ---- | ---- | ---- | ---- |
| Assessment  | Model      | OVE  | COH  | FLU  | CON  |
| Comparative | FlanT5-xl  | 54.6 | 51.2 | 32.5 | 47.1 |
|             | Llama2-7b  | 31.4 | 28.2 | 23.0 | 27.5 |
|             | Mistral-7b | 25.1 | 27.6 | 21.1 | 27.1 |
| Absolute    | FlanT5-xl  | 24.6 | 27.0 | 16.6 | 37.7 |
|             | Llama2-7b  | 25.0 | 28.2 | 23.0 | 29.4 |
|             | Mistral-7b | 10.2 | 14.3 | 10.5 | 7.1  |
|             | GPT3.5     | 52.5 | 45.1 | 38.0 | 43.2 |

Table 1: Zero-shot performance (Spearman correlation coefficient) on
SummEval. Due to cost GPT3.5 was not evaluated for comparative
assessment.

|             |            |      |      |       |      |
| ----------- | ---------- | ---- | ---- | ----- | ---- |
| Assessment  | Model      | OVE  | COH  | CNT   | ENG  |
| Comparative | FlanT5-xl  | 38.8 | 47.8 | 43.5  | 34.9 |
|             | Llama2-7b  | 34.5 | 35.2 | 37.1  | 32.0 |
|             | Mistral-7b | 38.6 | 33.1 | 36.1  | 33.3 |
| Absolute    | FlanT5-xl  | 36.2 | 31.4 | 43.2  | 34.9 |
|             | Llama2-7b  | 37.1 | 28.7 | 20.0  | 32.9 |
|             | Mistral-7b | 51.7 | 32.2 | 37.10 | 33.5 |
|             | GPT3.5     | 56.2 | 54.7 | 57.7  | 49.1 |

Table 2: Performance (Spearman correlation coefficient) on TopicalChat.
Due to cost GPT3.5 was not evaluated for comparative assessment.

Tables 1 and 2 present the assessment ability of each LLM when applied
to comparative and absolute assessment for SummEval and TopicalChat.
Consistent with literature, comparative assessment performs better than
absolute assessment systems for most systems and attributes. However,
comparative assessment uses $`N\!\cdot\!(N\!-\!1)`$ to compare all pairs
of responses (Equation 2), whilst only $`N`$ inferences are required for
absolute assessment. Smaller LLMs (FlanT5-xl, Llama2-7b and Mistral-7b)
demonstrate reasonable performance on SummEval and TopicalChat, but
larger models (GPT3.5) perform much better, and when applying absolute
scoring can outperform smaller systems using comparative assessment.

### 6.2 Attack on Surrogate Model

Section 5.3 details the attack approach to learn the universal attack
phrases for the surrogate model. Figure 2 illustrates the impact of the
universal adversarial on SummEval and TopicalChat, where FlanT5-xl is
used as the surrogate LLM assessment system. For Summeval, the overall
score (OVE) and consistency (CON) is attacked while for Topical-Chat the
overall score (OVE) and continuity (CNT) is attacked. The attributes CON
and CNT were selected due to the similar performance for these
attributes in the absolute and comparative settings (seen in Tables 1
and 2).

The success of the adversarial attacks is measured by the average ranks
of the text after an attack. Figure 2 demonstrates that both comparative
assessment and absolute assessment systems have some vulnerability to
adversarial attacks, as the average rank decreases, and continues to
decrease as more words are added to the attack phrase. However, absolute
scoring systems are significantly more susceptible to universal
adversarial attacks, and with just four universal attack words, the
absolute scoring system will consistently provide a rank of 1 to nearly
all input texts. Table 3 provides the raw scores for comparative and
absolute assessment, where we see that for absolute assessment, a
universal attack phrase of 4 words will yield assessment scores on
average near the maximum score of 5. The specific universal attack
phrases learnt for each task are given in Appendix A.

| Phrase         | No Attack | Attack |
| -------------- | --------- | ------ |
| SUMM COMP OVE  | 50.00     | 51.34  |
| SUMM COMP CON  | 50.00     | 57.10  |
| TOPIC COMP OVE | 50.00     | 53.94  |
| TOPIC COMP CNT | 50.00     | 54.06  |
| SUMM ABS OVE   | 3.73      | 4.74   |
| SUMM ABS CON   | 3.88      | 4.35   |
| TOPIC ABS OVE  | 2.93      | 4.63   |
| TOPIC ABS CNT  | 3.02      | 4.32   |

Table 3: Scores for 4-word universal attacks on FlanT5-xl. Note that
scores for comparative and absolute assessment are not comparable.

(a) SummEval

(b) TopicalChat

Figure 3: Transferability of universal attack phrases from surrogate
FlanT5-xl to target models.

The relative robustness of comparative assessment systems over absolute
assessment systems can perhaps be explained intuitively. In an absolute
assessment setting, an adversary exploits an input space which is not
well understood by the model and identifies a region that spuriously
encourages the model to predict a high score. However, in comparative
assessment, the model is forced to compare the quality of the attacked
text to another (unattacked) text, meaning the attack phrase learnt has
to be invariant to the text used for comparison. This makes it more
challenging to find an effective universal attack phrase. Further
explanations for the relative robustness of comparative assessment
systems are explored in Appendix B.

### 6.3 Transferability of the Surrogate Attack

Figure 2 demonstrated that absolute assessment systems are highly
vulnerable to a simple universal attack phrase concatenated to an input
text. To evaluate the effectiveness of these attack phrases on more
powerful target models, we explicitly transfer the attacks learned on
the FlanT5-xl surrogate model to other models such as Llama2, Mistral
and GPT3.5. We focus on transferring the absolute scoring attacks, as
comparative assessments were found to be relatively robust for the
surrogate FlanT5-xl model. Figure 3 shows the results of transferring
the attack phrases to these models, highlighting several key
findings: 1) There can be a high level of attack transferability for
absolute scoring. For TopicalChat, the attacks generalize very well to
nearly all systems, with all systems being very susceptible to attacks
when assessing continuity. 2) When more powerful models assess the
overall (OVE) quality, the transferability is less effective, suggesting
that assessing more general, abstract qualities can be more robust.
Interestingly, powerful large models (GPT3.5) are more susceptible when
attacked by shorter phrases, possibly because longer phrases may begin
to overfit the properties of the surrogate model. 3) The attack
transfers with mixed success for SummEval, which may highlight that the
complexity of the dataset can influence attack transferability.

(a) SummEval

(b) TopicalChat

Figure 4: Precision-Recall curve when applying perplexity as a detection
defence

### 6.4 Attack Detection

In this section, we perform an initial investigation into possible
defences that could be applied to detect if an adversary is exploiting a
system. Defences can take two forms: adversarial training (Goodfellow
et al., 2015) where the LLM is re-trained with adversarial examples, or
adversarial attack detection where a separate module is designed to
identify adversarial inputs. Although recent LLM adversarial training
approaches have been proposed (Zhou et al., 2024; Zhang et al., 2023b),
re-training is computationally expensive and can harm model performance,
hence detection is preferred. Recent detection approaches for NLG
adversarial attacks tend to focus on attacks that circumvent LLM safety
filters, e.g., generating malicious content by jailbreaking (Liu et al.,
2023c; Zou et al., 2023; Jin et al., 2024). Robey et al. (2023) propose
SmoothLLM, where multiple versions of the perturbed input are passed to
an LLM and the outputs aggregated. Such defences are inappropriate for
LLM-as-a-judge setups, as though the perturbations are designed to cause
no semantic change, they can result in changes in other attributes, such
as fluency and style, which will impact the LLM assessment. Similarly,
Jain et al. (2023); Kumar et al. (2024) propose defence approaches that
involve some form of paraphrasing or filtering of the input sequence,
which again interferes with the LLM-as-a-judge scores.

A simple and valid defence approach for LLM-as-a-judge is to use
perplexity to detect adversarial examples (Jain et al., 2023; Raina
et al., 2020). The perplexity is a measure of how unnatural a model,
$`\theta`$ finds a sentence $`\mathbf{x}`$,

```math
\texttt{perp}=-\frac{1}{|\mathbf{x}|}\log(P_{\theta}(\mathbf{x})). \tag{11}
```

We use the base Mistral-7B model to compute perplexity. Adversarially
attacked samples are expected to be less natural and have higher
perplexity. Therefore, we can evaluate the detection performance using
precision and recall. We select a specific threshold, $`\beta`$ to
classify an input sample $`\mathbf{x}`$ as clean or adversarial, where
if $`\texttt{perp}>\beta`$ the sample would be classified as
adversarial. The precision, recall and F1 is then

```math
\texttt{P}=\frac{\texttt{TP}}{\texttt{TP+FP}}\quad\texttt{R}=\frac{\texttt{TP}}{\texttt{TP+FN}}\quad\texttt{F1}=2\cdot\frac{\texttt{P}\cdot\texttt{R}}{\texttt{P}+\texttt{R}},
```

where FP, TP and FN are standard counts for False-Positive,
True-Positive and False-Negative respectively. The F1 can be used as a
single-value summary of detection performance.

To assess detection, we evaluate on the test split of each dataset,
augmented with the universal attack phrase concatenated to each text,
such that there is balance between clean and adversarial examples.
Figure 4 presents precision-recall (p-r) curves for perplexity detection
as the threshold $`\beta`$ is swept, for the different universal
adversarial phrases. Table 4 gives the best F1 scores from the p-r
curves. For SummEval all the F1 scores are near 0.7 or significantly
above, whilst for TopicalChat the performance is generally even better.
This demonstrates that perplexity is fairly effective in disentangling
clean and adversarial samples for attacks on LLM-as-a-judge. However,
Zhou et al. (2024) argue that defence approaches such as perplexity
detection can be circumvented by adaptive adversarial attacks. Hence,
though perplexity gives a promising starting point as a defence
strategy, future work will explore other more sophisticated detection
approaches. Nevertheless, it can also be concluded from the findings in
this work that an effective defence against the most threatening
adversarial attacks on LLM-as-a-judge is to use comparative assessment
over absolute scoring, despite an increased computational cost.

| Attack      | precision | recall | F1    |
| ----------- | --------- | ------ | ----- |
| Summ-CON-2  | 0.635     | 0.794  | 0.706 |
| Summ-CON-4  | 0.679     | 0.819  | 0.742 |
| Summ-OVE-2  | 0.539     | 0.988  | 69.6  |
| Summ-OVE-4  | 64.7      | 81.3   | 72.0  |
| Topic-CNT-2 | 66.2      | 84.4   | 81.7  |
| Topic-CNT-4 | 74.8      | 79.5   | 77.1  |
| Topic-OVE-2 | 75.2      | 78.8   | 76.9  |
| Topic-OVE-4 | 78.5      | 85.1   | 81.7  |

Table 4: Best F1 (%) (precision, recall) for adversarial sample
detection using perplexity. Attack phrases of length 2 words and 4 words
considered.

## 7 Conclusions

This is the first work to examine the adversarial robustness of
zero-shot LLM assessment methods against universal adversarial attacks,
and reveal significant vulnerabilities in LLM absolute scoring and mild
vulnerabilities in LLM comparative assessment. We demonstrate that the
same short 4-word universal adversarial can be appended to any input
text to deceive LLM assessment system into predicting inflated scores.
Notably, LLM-scoring attacks developed with a smaller surrogate
LLM-scoring system can be effectively transferred to larger LLMs such as
ChatGPT. We also provide an initial investigation into simple detection
approaches, and show that perplexity can be a promising tool for
identifying adversarially manipulated inputs. Further work can explore
adaptive attacks and more sophisticated defence approaches to minimize
the risk of misuse. On the whole, this paper raises awareness around the
susceptibility of LLM-as-a-judge NLG assessment systems to universal and
transferable adversarial attacks.

## 8 Limitations

This paper investigates the vulnerability of LLM-as-a-judge methods in
settings where malicious entities may wish to trick systems into
returning inflated assessment scores. As the first work on the
adversarial robustness of LLM assessment, we used simple attacks
(concatenation attack found through a greedy search) which led to simple
defences (perplexity). Future work can investigate methods of achieving
more subtle attacks, which may require more complex defences to detect.
Further, this work focuses on attacking zero-shot assessment methods,
however, it is possible to use LLM assessment in few-shot settings,
which may be more robust and render attacks less effective. Future work
can explore this direction, and also investigate designing prompts that
are more robust to attacks.

## 9 Risks & Ethics

This work reports on the topic of adversarial attacks, where it’s shown
that a universal adversarial attack can fool NLG assessment systems into
inflating scores of assessed texts. The methods and attacks proposed in
this paper do not encourage any harmful content generation and the aim
of the work is to raise awareness of the risk of adversarial
manipulation for zero-shot NLG assessment. It is possible that
highlighting these susceptibilities may inform adversaries of this
vulnerability, however, we hope that raising awareness of these risks
will encourage the community to further study the robustness of
zero-shot LLM assessment methods and reduce the risk of future misuse.

## References

- Alzantot et al. (2018) Moustafa Alzantot, Yash Sharma, Ahmed Elgohary,
  Bo-Jhang Ho, Mani Srivastava, and Kai-Wei Chang. 2018. [Generating
  natural language adversarial
  examples](https://doi.org/10.18653/v1/D18-1316). pages 2890–2896.
- Banerjee and Lavie (2005) Satanjeev Banerjee and Alon Lavie. 2005.
  [METEOR: An automatic metric for MT evaluation with improved
  correlation with human judgments](https://aclanthology.org/W05-0909).
  In _Proceedings of the ACL Workshop on Intrinsic and Extrinsic
  Evaluation Measures for Machine Translation and/or Summarization_,
  pages 65–72, Ann Arbor, Michigan. Association for Computational
  Linguistics.
- Carlini et al. (2023) Nicholas Carlini, Milad Nasr, Christopher A.
  Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei
  Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, and Ludwig
  Schmidt. 2023. [Are aligned neural networks adversarially
  aligned?](http://arxiv.org/abs/2306.15447)
- Carlini et al. (2020) Nicholas Carlini, Florian Tramèr, Eric Wallace,
  Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts,
  Tom B. Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin
  Raffel. 2020. [Extracting training data from large language
  models](http://arxiv.org/abs/2012.07805). _CoRR_, abs/2012.07805.
- Chao et al. (2023) Patrick Chao, Alexander Robey, Edgar Dobriban,
  Hamed Hassani, George J. Pappas, and Eric Wong. 2023. [Jailbreaking
  black box large language models in twenty
  queries](http://arxiv.org/abs/2310.08419).
- Chen et al. (2023) Yi Chen, Rui Wang, Haiyun Jiang, Shuming Shi, and
  Ruifeng Xu. 2023. [Exploring the use of large language models for
  reference-free text quality evaluation: An empirical
  study](https://aclanthology.org/2023.findings-ijcnlp.32). In _Findings
  of the Association for Computational Linguistics: IJCNLP-AACL 2023
  (Findings)_, pages 361–374, Nusa Dua, Bali. Association for
  Computational Linguistics.
- Chung et al. (2022) Hyung Won Chung, Le Hou, Shayne Longpre, Barret
  Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa
  Dehghani, Siddhartha Brahma, et al. 2022. Scaling
  instruction-finetuned language models. _arXiv preprint
  arXiv:2210.11416_.
- Fabbri et al. (2021) Alexander R Fabbri, Wojciech Kryściński, Bryan
  McCann, Caiming Xiong, Richard Socher, and Dragomir Radev. 2021.
  Summeval: Re-evaluating summarization evaluation. _Transactions of the
  Association for Computational Linguistics_, 9:391–409.
- Fu et al. (2023) Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei
  Liu. 2023. Gptscore: Evaluate as you desire. _arXiv preprint
  arXiv:2302.04166_.
- Gao et al. (2018) Ji Gao, Jack Lanchantin, Mary Lou Soffa, and Yanjun
  Qi. 2018. [Black-box generation of adversarial text sequences to evade
  deep learning classifiers](http://arxiv.org/abs/1801.04354). _CoRR_,
  abs/1801.04354.
- Garg and Ramakrishnan (2020) Siddhant Garg and Goutham
  Ramakrishnan. 2020. [BAE: BERT-based adversarial examples for text
  classification](https://doi.org/10.18653/v1/2020.emnlp-main.498). In
  _Proceedings of the 2020 Conference on Empirical Methods in Natural
  Language Processing (EMNLP)_, pages 6174–6181, Online. Association for
  Computational Linguistics.
- Goodfellow et al. (2015) Ian J. Goodfellow, Jonathon Shlens, and
  Christian Szegedy. 2015. [Explaining and harnessing adversarial
  examples](http://arxiv.org/abs/1412.6572).
- Gopalakrishnan et al. (2019) Karthik Gopalakrishnan, Behnam
  Hedayatnia, Qinlang Chen, Anna Gottardi, Sanjeev Kwatra, Anu
  Venkatesh, Raefer Gabriel, and Dilek Hakkani-Tür. 2019. [Topical-Chat:
  Towards Knowledge-Grounded Open-Domain
  Conversations](https://doi.org/10.21437/Interspeech.2019-3079). In
  _Proc. Interspeech 2019_, pages 1891–1895.
- Huang et al. (2024) Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai
  Li, and Danqi Chen. 2024. [Catastrophic jailbreak of open-source LLMs
  via exploiting
  generation](https://openreview.net/forum?id=r42tSSCHPh). In _The
  Twelfth International Conference on Learning Representations_.
- Jain et al. (2023) Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami
  Somepalli, John Kirchenbauer, Ping yeh Chiang, Micah Goldblum,
  Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023. [Baseline
  defenses for adversarial attacks against aligned language
  models](http://arxiv.org/abs/2309.00614).
- Jiang et al. (2023) Albert Q Jiang, Alexandre Sablayrolles, Arthur
  Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas,
  Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier,
  et al. 2023. Mistral 7b. _arXiv preprint arXiv:2310.06825_.
- Jin et al. (2024) Haibo Jin, Ruoxi Chen, Andy Zhou, Jinyin Chen, Yang
  Zhang, and Haohan Wang. 2024. [Guard: Role-playing to generate
  natural-language jailbreakings to test guideline adherence of large
  language models](http://arxiv.org/abs/2402.03299).
- Koo et al. (2023) Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park,
  Zae Myung Kim, and Dongyeop Kang. 2023. [Benchmarking cognitive biases
  in large language models as
  evaluators](http://arxiv.org/abs/2309.17012).
- Kumar et al. (2024) Aounon Kumar, Chirag Agarwal, Suraj Srinivas,
  Aaron Jiaxun Li, Soheil Feizi, and Himabindu Lakkaraju. 2024.
  [Certifying llm safety against adversarial
  prompting](http://arxiv.org/abs/2309.02705).
- Lapid et al. (2023) Raz Lapid, Ron Langberg, and Moshe Sipper. 2023.
  [Open sesame! universal black box jailbreaking of large language
  models](http://arxiv.org/abs/2309.01446).
- Li et al. (2020) Linyang Li, Ruotian Ma, Qipeng Guo, Xiangyang Xue,
  and Xipeng Qiu. 2020. [BERT-ATTACK: Adversarial attack against BERT
  using BERT](https://doi.org/10.18653/v1/2020.emnlp-main.500). pages
  6193–6202.
- Lin (2004) Chin-Yew Lin. 2004. Rouge: A package for automatic
  evaluation of summaries. In _Text summarization branches out_, pages
  74–81.
- Liu et al. (2023a) Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao.
  2023a. [Autodan: Generating stealthy jailbreak prompts on aligned
  large language models](http://arxiv.org/abs/2310.04451).
- Liu et al. (2023b) Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
  Ruochen Xu, and Chenguang Zhu. 2023b. [G-eval: NLG evaluation using
  gpt-4 with better human
  alignment](https://doi.org/10.18653/v1/2023.emnlp-main.153). In
  _Proceedings of the 2023 Conference on Empirical Methods in Natural
  Language Processing_, pages 2511–2522, Singapore. Association for
  Computational Linguistics.
- Liu et al. (2016) Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn
  Song. 2016. [Delving into transferable adversarial examples and
  black-box attacks](http://arxiv.org/abs/1611.02770). _CoRR_,
  abs/1611.02770.
- Liu et al. (2023c) Yi Liu, Gelei Deng, Zhengzi Xu, Yuekang Li, Yaowen
  Zheng, Ying Zhang, Lida Zhao, Tianwei Zhang, and Yang Liu. 2023c.
  [Jailbreaking chatgpt via prompt engineering: An empirical
  study](http://arxiv.org/abs/2305.13860).
- Liu et al. (2023d) Yiqi Liu, Nafise Sadat Moosavi, and Chenghua Lin.
  2023d. [Llms as narcissistic evaluators: When ego inflates evaluation
  scores](http://arxiv.org/abs/2311.09766).
- Liusie et al. (2023) Adian Liusie, Potsawee Manakul, and Mark JF
  Gales. 2023. Zero-shot nlg evaluation through pairware comparisons
  with llms. _arXiv preprint arXiv:2307.07889_.
- Manakul et al. (2023) Potsawee Manakul, Adian Liusie, and Mark JF
  Gales. 2023. Mqag: Multiple-choice question answering and generation
  for assessing information consistency in summarization. _arXiv
  preprint arXiv:2301.12307_.
- Mehri and Eskenazi (2020) Shikib Mehri and Maxine Eskenazi. 2020.
  [Unsupervised evaluation of interactive dialog with
  DialoGPT](https://doi.org/10.18653/v1/2020.sigdial-1.28). In
  _Proceedings of the 21th Annual Meeting of the Special Interest Group
  on Discourse and Dialogue_, pages 225–235, 1st virtual meeting.
  Association for Computational Linguistics.
- Mehrotra et al. (2023) Anay Mehrotra, Manolis Zampetakis, Paul
  Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin
  Karbasi. 2023. [Tree of attacks: Jailbreaking black-box llms
  automatically](http://arxiv.org/abs/2312.02119).
- Nasr et al. (2023) Milad Nasr, Nicholas Carlini, Jonathan Hayase,
  Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A.
  Choquette-Choo, Eric Wallace, Florian Tramèr, and Katherine Lee. 2023.
  [Scalable extraction of training data from (production) language
  models](http://arxiv.org/abs/2311.17035).
- Papernot et al. (2016) Nicolas Papernot, Patrick D. McDaniel, Ian J.
  Goodfellow, Somesh Jha, Z. Berkay Celik, and Ananthram Swami. 2016.
  [Practical black-box attacks against deep learning systems using
  adversarial examples](http://arxiv.org/abs/1602.02697). _CoRR_,
  abs/1602.02697.
- Qin et al. (2023) Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang,
  Junru Wu, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui
  Wang, and Michael Bendersky. 2023. [Large language models are
  effective text rankers with pairwise ranking
  prompting](http://arxiv.org/abs/2306.17563).
- Raina and Gales (2023) Vyas Raina and Mark Gales. 2023. [Sentiment
  perception adversarial attacks on neural machine translation
  systems](http://arxiv.org/abs/2305.01437).
- Raina et al. (2020) Vyas Raina, Mark J.F. Gales, and Kate M.
  Knill. 2020. [Universal Adversarial Attacks on Spoken Language
  Assessment Systems](https://doi.org/10.21437/Interspeech.2020-1890).
  In _Proc. Interspeech 2020_, pages 3855–3859.
- Rei et al. (2020) Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon
  Lavie. 2020. Comet: A neural framework for mt evaluation. In
  _Proceedings of the 2020 Conference on Empirical Methods in Natural
  Language Processing (EMNLP)_, pages 2685–2702.
- Robey et al. (2023) Alexander Robey, Eric Wong, Hamed Hassani, and
  George J. Pappas. 2023. [Smoothllm: Defending large language models
  against jailbreaking attacks](http://arxiv.org/abs/2310.03684).
- Sadrizadeh et al. (2023) Sahar Sadrizadeh, Ljiljana Dolamic, and
  Pascal Frossard. 2023. [A classification-guided approach for
  adversarial attacks against neural machine
  translation](http://arxiv.org/abs/2308.15246).
- Szegedy et al. (2014) Christian Szegedy, Wojciech Zaremba, Ilya
  Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob
  Fergus. 2014. [Intriguing properties of neural
  networks](http://arxiv.org/abs/1312.6199).
- Touvron et al. (2023) Hugo Touvron, Louis Martin, Kevin Stone, Peter
  Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya
  Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open
  foundation and fine-tuned chat models. _arXiv preprint
  arXiv:2307.09288_.
- Wang et al. (2020) Alex Wang, Kyunghyun Cho, and Mike Lewis. 2020.
  Asking and answering questions to evaluate the factual consistency of
  summaries. In _Proceedings of the 58th Annual Meeting of the
  Association for Computational Linguistics_, pages 5008–5020.
- Wang et al. (2023a) Jiaan Wang, Yunlong Liang, Fandong Meng, Haoxiang
  Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. 2023a. Is chatgpt
  a good nlg evaluator? a preliminary study. _arXiv preprint
  arXiv:2303.04048_.
- Wang et al. (2023b) Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei
  Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui.
  2023b. [Large language models are not fair
  evaluators](http://arxiv.org/abs/2305.17926).
- Wang et al. (2019) Xiaosen Wang, Hao Jin, and Kun He. 2019. [Natural
  language adversarial attacks and defenses in word
  level](http://arxiv.org/abs/1909.06723). _CoRR_, abs/1909.06723.
- Wei et al. (2023) Alexander Wei, Nika Haghtalab, and Jacob
  Steinhardt. 2023. [Jailbroken: How does llm safety training
  fail?](http://arxiv.org/abs/2307.02483)
- Zeng et al. (2024) Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang,
  Ruoxi Jia, and Weiyan Shi. 2024. [How johnny can persuade llms to
  jailbreak them: Rethinking persuasion to challenge ai safety by
  humanizing llms](http://arxiv.org/abs/2401.06373).
- Zhang et al. (2019) Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q
  Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text
  generation with bert. In _International Conference on Learning
  Representations_.
- Zhang et al. (2023a) Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv,
  Tingwen Liu, Fei Huang, Hongbo Xu, and Yongbin Li. 2023a. [Wider and
  deeper llm networks are fairer llm
  evaluators](http://arxiv.org/abs/2308.01862).
- Zhang et al. (2023b) Zhexin Zhang, Junxiao Yang, Pei Ke, and Minlie
  Huang. 2023b. [Defending large language models against jailbreaking
  attacks through goal prioritization](http://arxiv.org/abs/2311.09096).
- Zhao et al. (2023) Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
  Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang,
  Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao
  Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu,
  Jian-Yun Nie, and Ji-Rong Wen. 2023. [A survey of large language
  models](http://arxiv.org/abs/2303.18223).
- Zheng et al. (2023) Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
  Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li,
  Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and
  chatbot arena. _arXiv preprint arXiv:2306.05685_.
- Zhong et al. (2022a) Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu
  Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and Jiawei Han. 2022a.
  Towards a unified multi-dimensional evaluator for text generation.
  _arXiv preprint arXiv:2210.07197_.
- Zhong et al. (2022b) Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu
  Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and Jiawei Han. 2022b.
  [Towards a unified multi-dimensional evaluator for text
  generation](https://doi.org/10.18653/v1/2022.emnlp-main.131). In
  _Proceedings of the 2022 Conference on Empirical Methods in Natural
  Language Processing_, pages 2023–2038, Abu Dhabi, United Arab
  Emirates. Association for Computational Linguistics.
- Zhou et al. (2024) Andy Zhou, Bo Li, and Haohan Wang. 2024. [Robust
  prompt optimization for defending language models against jailbreaking
  attacks](http://arxiv.org/abs/2401.17263).
- Zhu et al. (2023a) Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen
  Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang,
  Neil Zhenqiang Gong, and Xing Xie. 2023a. [Promptbench: Towards
  evaluating the robustness of large language models on adversarial
  prompts](http://arxiv.org/abs/2306.04528).
- Zhu et al. (2023b) Lianghui Zhu, Xinggang Wang, and Xinlong Wang.
  2023b. [Judgelm: Fine-tuned large language models are scalable
  judges](http://arxiv.org/abs/2310.17631).
- Zhu et al. (2024) Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe
  Barrow, Furong Huang, and Tong Sun. 2024. [AutoDAN: Automatic and
  interpretable adversarial attacks on large language
  models](https://openreview.net/forum?id=ZuZujQ9LJV).
- Zou et al. (2023) Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr,
  J. Zico Kolter, and Matt Fredrikson. 2023. [Universal and transferable
  adversarial attacks on aligned language
  models](http://arxiv.org/abs/2307.15043).

## Appendix A Universal Adversarial Phrases

In the main paper, results are presented for a range of universal attack
phrases, learnt in different configurations. Further configurations are
considered in different sections of the Appendix. For all of these
attack phrases, the specific words constituting each phrase are
presented in Table 5.

|                     |                                                        |
| ------------------- | ------------------------------------------------------ |
| SUMM COMP OVE       | E answer E grammatically                               |
| SUMM COMP CON       | uncontradictory Ay supplementary answer                |
| SUMM ABS OVE        | outstandingly superexcellently outstandingly summable  |
| SUMM ABS CON        | uncontradictedly undisputably congruity impeccable     |
| TOPIC COMP OVE      | informative ending answer E                            |
| TOPIC COMP CNT      | interester extemporaneous informative answer           |
| TOPIC ABS OVE       | informative supercomplete impeccable ovated            |
| TOPIC ABS CNT       | continuous superexcellently conformant uncontradictory |
| SUMM COMP-asymA OVE | E applicableness E E                                   |
| SUMM COMP-asymB OVE | grammatically sound emendable correctly                |
| SUMM UNI OVE        | whoa boggle righto hah                                 |
| SUMM UNI COH        | read inustion newsprint introductorily                 |
| SUMM UNI CON        | compendent at id id                                    |
| SUMM UNI FLU        | Feuillants cavort extortionately ashore                |

Table 5: Universal Attack Phrases. Length 1 to length 4 words

## Appendix B Analysis of Relative Robustness of Comparative Assessment

It is observed that comparative assessment is more robust than absolute
assessment. Arguably this could be due to an implicit prompt ensemble
with different output objectives in comparative assessment. In absolute
assessment, the adversary has to find a phrase that always pushes the
predicted token to the maximal score 5, irrespective of the input test.
For comparative assessment, to evaluate the probability summary $`i`$ is
better than $`j`$ to ensure symmetry, we do two passes through the
system. To attack system $`i`$, for the first pass, the adversary has to
ensure the attack phrase increases the probability of token A (the
prompt asks the system to select which text input, A or B, is better,
where A corresponds to the text in position 1 and B corresponds to the
text in position 2) being predicted. For the second pass the adversary
has to decrease the predicted probability of token A (as attacked
summary is in position 2). This means the objective of the adversary in
the different passes is dependent on the prompt ordering of summaries,
as well as the objectives being the complete opposite in the two passes
(competing objectives). This means the universal attack phrase has to
recognise automatically whether it is in position 1 or in position 2 and
respectively increase or decrease the output probability of generating
token A. This is a lot more challenging and could explain the robustness
of comparative assessment. How do we assess this hypothesis:

- •
  We perform an ablation where the comparative assessment system does
  asymmetric evaluation such that the probability system $`i`$ is better
  than $`j`$ is measured asymmetrically, with the attacked text always
  in position 1, such that the adversarial attack only has to maximize
  the probability of token A. It is expected that the asymmetric
  comparative assessment system is less robust.
- •
  We re-apply the greedy search algorithm with this asymmetric setup.
- •
  We evaluate the efficacy of the attack phrase in the asymmetric
  setting.
- •
  We repeat the above experiments with the attack only in position 2
  (objective then being to minimize the probability of token B). We term
  the universal attack phrases asymA and asymB.

The results are presented in Table 6 and Table 7. It seems that even in
this asymmetric setting the robustness performance is only slightly (if
that) worse than that of the symmetric evaluation setting in the main
paper. This suggests that perhaps there is a separate aspect of
comparative assessment approach that contributes significantly to the
robustness. Further analysis will be required to better understand
exactly which aspects of comparative assessment are giving the greatest
robustness.

| \#words | s-s   | s-u   | u-s   | u-u   | all   | $`\bar{r}`$ |
| ------- | ----- | ----- | ----- | ----- | ----- | ----------- |
| None    | 45.43 | 41.07 | 37.70 | 42.07 | 41.54 | 8.50        |
| 1       | 51.12 | 51.80 | 46.68 | 50.23 | 50.03 | 6.17        |
| 2       | 34.96 | 38.09 | 34.32 | 37.54 | 37.21 | 9.80        |
| 3       | 48.23 | 49.04 | 44.60 | 47.10 | 47.06 | 6.81        |

Table 6: Direct attack on FlanT5-xl. Evaluating attack phrase SUMM
COMP-asymA OVE

| \#words | s-s   | s-u   | u-s   | u-u   | all   | $`\bar{r}`$ |
| ------- | ----- | ----- | ----- | ----- | ----- | ----------- |
| None    | 54.57 | 62.30 | 58.93 | 57.93 | 58.46 | 8.50        |
| 1       | 51.91 | 60.80 | 52.80 | 54.36 | 54.86 | 9.52        |
| 2       | 57.84 | 65.04 | 56.58 | 58.38 | 58.90 | 8.16        |
| 3       | 57.89 | 63.78 | 56.29 | 57.20 | 57.83 | 8.54        |
| 4       | 64.70 | 68.95 | 60.53 | 62.00 | 62.64 | 7.06        |

Table 7: Direct attack on FlanT5-xl. Evaluating attack phrase SUMM
COMP-asymB OVE

## Appendix C Transferability of the Comparative Assessment Attack

Figure 2 shows that when the surrogate model (FlanT5-xl) is run as
comparative assessment it is only mildly susceptible to the universal
adversarial attack. Hence, Section 6.3 in the paper reports only the
transferability of the attack on the absolute assessment systems to the
target larger models (Mistral, Llama2 and ChatGPT). For completeness, in
this section we provide the impact of transferring the attacks for
comparative assessment. The transferability plots are given in Figure 5.
As would be expected, the mild attacks learnt for the surrogate model
FlanT5-xl are only are able to maintain at best a mild impact for the
target models.

(a) SummEval

(b) TopicalChat

Figure 5: Transferability of universal attack phrases from FlanT5-xl to
other models for comparative assessment.

## Appendix D Direct Attack on Target Model

The main paper proposes a practical method to attack LLM-as-a-Judge
system that use large LLMs, via a surrogate model (FlanT5-xl in this
work). For comparison, this section presents the results for performing
a direct attack on Llama2-7B (a target larger model). The resulst are
presented for absolute assessment in Figure 6. As would be expected from
the bounds of the transfer attacks, the direct attack is equally (and
more) successful in deceiving the LLM absolute scoring systems into
giving the attacked text the highest ranking score.

(a) SummEval

(b) TopicalChat

Figure 6: Universal Attack Evaluation (average rank of attacked
summary/response) for Llama2-7B.

## Appendix E Greedy Coordinate Gradient (GCG) Universal Attack

In the main paper we present an iterative greedy search for a universal
concatenative attack phrase. Here, we contrast our approach against the
Greedy Coordinate Gradient (GCG) adversarial attack approach used by Zou
et al. (2023). In our GCG experiments we adopt the default
hyperparameter settings from the paper for the universal GCG algorithm.
The GCG attack is a whitebox approach that exploits embedding gradients
to identify which tokens to substitute from the concatenated phrase.
Table 8 shows the impact of incorporating GCG with initialization from
the existing learnt attack phrases for absolute assessment and the
comparative assessment on overall assessment. From these results it
appears that GCG has a negligible impact on the adversarial attack
efficacy, and can in many cases degrade the attack (worse average
rank) - this is perhaps expected for the best / well optimized attack
phrases.

| Initialisation | No GCG ($`\bar{r}`$) | With GCG ($`\bar{r}`$) |
| -------------- | -------------------- | ---------------------- |
| SUMM COMP OVE  | 7.96                 | 7.88                   |
| SUMM ABS OVE   | 1.03                 | 2.42                   |
| TOPIC COMP OVE | 3.16                 | 3.18                   |
| TOPIC ABS OVE  | 1.07                 | 3.56                   |

Table 8: Impact of universal GCG adversarial attack on existing
universal attacks

## Appendix F Interpretable Attack Results

The main paper presents the impact of the adversarial attack phrases for
comparative and absolute assessment systems on the average rank as
defined in Equation 8. However, it is more interpretable to understand
the the impact on the probability, $`p_{ij}`$ (Equation 1) of an
attacked system being better than other systems for comparative
assessment and the impact on the average predicted score (Equation 3)
for absolute assessment. Tables 9-12 give the interpretable breakdown of
each attack for comparative assessment and Tables 13-28 give the
equivalent interpretable breakdown for absolute assessment.

| \#words | s-s   | s-u   | u-s   | u-u   | $`\bar{p}_{ij}`$ | $`\bar{r}`$ |
| ------- | ----- | ----- | ----- | ----- | ---------------- | ----------- |
| None    | 50.00 | 51.68 | 48.32 | 50.00 | 50.00            | 8.50        |
| 1       | 50.59 | 55.97 | 50.48 | 52.73 | 52.80            | 7.48        |
| 2       | 41.22 | 49.73 | 43.90 | 46.49 | 46.48            | 9.75        |
| 3       | 51.27 | 58.55 | 51.84 | 54.33 | 54.48            | 6.97        |
| 4       | 50.01 | 55.88 | 47.49 | 51.27 | 51.34            | 7.96        |

Table 9: Direct Attack on FlanT5-xl. Evaluating attack phrase SUMM COMP
OVE. SummEval. 16 candidates, with 2 seen candidates (s) and remaining
unseen candidates (u).

| \#words | s-s   | s-u   | u-s   | u-u   | $`\bar{p}_{ij}`$ | $`\bar{r}`$ |
| ------- | ----- | ----- | ----- | ----- | ---------------- | ----------- |
| None    | 50.00 | 53.26 | 46.74 | 50.00 | 50.00            | 8.50        |
| 1       | 51.65 | 56.44 | 48.62 | 52.04 | 52.14            | 7.79        |
| 2       | 52.55 | 57.70 | 48.99 | 52.42 | 52.62            | 7.62        |
| 3       | 51.95 | 56.88 | 48.38 | 51.64 | 51.86            | 7.93        |
| 4       | 56.64 | 62.47 | 53.49 | 56.85 | 57.10            | 6.32        |

Table 10: Direct Attack on FlanT5-xl. Evaluating attack phrase SUMM COMP
CON. SummEval. 16 candidates, with 2 seen candidates (s) and remaining
unseen candidates (u).

| \#words | s-s   | s-u   | u-s   | u-u   | $`\bar{p}_{ij}`$ | $`\bar{r}`$ |
| ------- | ----- | ----- | ----- | ----- | ---------------- | ----------- |
| None    | 50.00 | 44.70 | 55.30 | 50.00 | 50.00            | 3.50        |
| 1       | 51.25 | 46.37 | 56.93 | 50.13 | 50.93            | 3.37        |
| 2       | 55.00 | 48.11 | 58.88 | 52.77 | 53.34            | 3.18        |
| 3       | 56.19 | 49.61 | 60.14 | 53.95 | 54.61            | 3.06        |
| 4       | 55.18 | 48.62 | 59.84 | 53.33 | 53.94            | 3.16        |

Table 11: Direct Attack on FlanT5-xl. Evaluating attack phrase TOPIC
COMP OVE. TopicalChat. 6 candidates, with 2 seen candidates (s) and
remaining unseen candidates (u).

| \#words | s-s   | s-u   | u-s   | u-u   | $`\bar{p}_{ij}`$ | $`\bar{r}`$ |
| ------- | ----- | ----- | ----- | ----- | ---------------- | ----------- |
| None    | 50.00 | 44.27 | 55.73 | 50.00 | 50.00            | 3.50        |
| 1       | 47.72 | 44.11 | 56.19 | 48.33 | 49.07            | 3.55        |
| 2       | 49.81 | 44.52 | 56.39 | 49.04 | 49.76            | 3.48        |
| 3       | 53.18 | 47.88 | 58.90 | 52.02 | 52.76            | 3.18        |
| 4       | 54.88 | 48.87 | 60.07 | 53.45 | 54.06            | 3.12        |

Table 12: Direct Attack on FlanT5-xl. Evaluating attack phrase TOPIC
COMP CNT. TopicalChat. 6 candidates, with 2 seen candidate types (s) and
remaining unseen candidates (u).

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 3.61 | 3.76 | 3.79 | 3.74 | 3.74 | 3.76 | 3.79 | 3.76 | 3.65 | 3.79 | 3.78 | 3.77 | 3.62 | 3.77 | 3.67 | 3.78 | 3.73 | 8.50        |
| 1       | 3.96 | 4.24 | 4.26 | 4.19 | 4.16 | 4.21 | 4.19 | 4.17 | 3.90 | 4.20 | 4.26 | 4.27 | 3.99 | 4.21 | 4.10 | 4.24 | 4.16 | 2.08        |
| 2       | 4.27 | 4.49 | 4.49 | 4.47 | 4.44 | 4.48 | 4.48 | 4.41 | 4.31 | 4.44 | 4.48 | 4.51 | 4.47 | 4.47 | 4.38 | 4.49 | 4.44 | 1.18        |
| 3       | 4.47 | 4.62 | 4.63 | 4.62 | 4.60 | 4.63 | 4.61 | 4.59 | 4.46 | 4.61 | 4.62 | 4.64 | 4.65 | 4.62 | 4.56 | 4.61 | 4.60 | 1.07        |
| 4       | 4.70 | 4.76 | 4.76 | 4.75 | 4.74 | 4.76 | 4.75 | 4.73 | 4.62 | 4.74 | 4.76 | 4.77 | 4.75 | 4.75 | 4.73 | 4.75 | 4.74 | 1.03        |

Table 13: Direct Attack on FlanT5-xl. Evaluating attack phrase SUMM ABS
OVE. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 3.61 | 3.90 | 3.94 | 3.88 | 3.90 | 3.93 | 4.00 | 3.92 | 3.74 | 3.95 | 3.95 | 3.96 | 3.77 | 3.93 | 3.74 | 3.91 | 3.88 | 8.50        |
| 1       | 3.83 | 4.22 | 4.26 | 4.18 | 4.19 | 4.23 | 4.19 | 4.15 | 3.77 | 4.17 | 4.27 | 4.29 | 3.98 | 4.22 | 3.99 | 4.21 | 4.13 | 3.51        |
| 2       | 3.93 | 4.27 | 4.31 | 4.25 | 4.25 | 4.29 | 4.30 | 4.23 | 3.92 | 4.25 | 4.32 | 4.35 | 4.25 | 4.27 | 4.09 | 4.28 | 4.22 | 2.49        |
| 3       | 4.10 | 4.37 | 4.38 | 4.36 | 4.35 | 4.39 | 4.41 | 4.37 | 4.25 | 4.39 | 4.40 | 4.42 | 4.44 | 4.38 | 4.24 | 4.37 | 4.35 | 1.71        |
| 4       | 4.10 | 4.37 | 4.38 | 4.36 | 4.35 | 4.39 | 4.41 | 4.37 | 4.25 | 4.39 | 4.40 | 4.42 | 4.44 | 4.38 | 4.24 | 4.37 | 4.35 | 1.71        |

Table 14: Direct Attack on FlanT5-xl. Evaluating attack phrase SUMM ABS
CON. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 3.00 | 3.81 | 3.89 | 3.75 | 3.75 | 3.84 | 3.88 | 4.00 | 3.52 | 3.96 | 3.86 | 3.99 | 4.00 | 3.84 | 3.52 | 3.52 | 3.76 | 8.50        |
| 1       | 3.16 | 3.80 | 3.90 | 3.73 | 3.73 | 3.89 | 3.99 | 4.00 | 3.54 | 3.99 | 3.91 | 4.06 | 3.98 | 3.80 | 3.56 | 3.52 | 3.78 | 8.32        |
| 2       | 2.80 | 3.48 | 3.59 | 3.19 | 3.39 | 3.41 | 3.46 | 3.86 | 3.01 | 3.74 | 3.45 | 3.52 | 3.95 | 3.35 | 2.99 | 3.16 | 3.40 | 10.47       |
| 3       | 2.80 | 3.54 | 3.60 | 3.24 | 3.49 | 3.45 | 3.61 | 3.92 | 2.90 | 3.74 | 3.59 | 3.64 | 3.99 | 3.39 | 3.08 | 3.21 | 3.45 | 10.23       |
| 4       | 3.01 | 3.64 | 3.71 | 3.40 | 3.51 | 3.49 | 3.61 | 3.98 | 2.58 | 3.90 | 3.61 | 3.66 | 3.90 | 3.50 | 3.31 | 3.50 | 3.52 | 9.48        |

Table 15: Transfer Attack on GPT3.5. Evaluating attack phrase SUMM ABS
OVE. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 3.67 | 4.05 | 4.15 | 4.00 | 4.00 | 4.04 | 4.19 | 4.05 | 3.89 | 4.05 | 4.12 | 4.26 | 4.04 | 4.01 | 3.92 | 3.92 | 4.02 | 8.50        |
| 1       | 3.70 | 4.20 | 4.24 | 4.04 | 4.09 | 4.26 | 4.44 | 4.09 | 3.91 | 4.09 | 4.30 | 4.61 | 4.28 | 4.11 | 3.94 | 3.94 | 4.14 | 7.63        |

Table 16: Transfer Attack on GPT3.5. Evaluating attack phrase SUMM ABS
CON. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 2.08 | 1.86 | 1.95 | 1.83 | 1.86 | 1.82 | 1.87 | 2.07 | 1.76 | 1.99 | 1.87 | 1.86 | 2.04 | 1.86 | 1.95 | 2.09 | 1.92 | 8.50        |
| 1       | 2.02 | 1.89 | 2.01 | 1.85 | 1.90 | 1.88 | 1.99 | 1.98 | 1.74 | 1.96 | 1.95 | 1.93 | 1.98 | 1.87 | 1.85 | 2.07 | 1.93 | 8.41        |
| 2       | 1.75 | 1.69 | 1.80 | 1.63 | 1.70 | 1.68 | 1.79 | 1.72 | 1.63 | 1.70 | 1.71 | 1.76 | 1.79 | 1.68 | 1.63 | 1.77 | 1.71 | 12.38       |
| 3       | 1.73 | 1.68 | 1.76 | 1.65 | 1.69 | 1.67 | 1.75 | 1.69 | 1.61 | 1.70 | 1.69 | 1.71 | 1.81 | 1.67 | 1.65 | 1.75 | 1.70 | 12.83       |
| 4       | 1.87 | 1.79 | 1.94 | 1.76 | 1.81 | 1.75 | 1.92 | 1.85 | 1.65 | 1.86 | 1.81 | 1.86 | 1.98 | 1.79 | 1.74 | 1.92 | 1.83 | 10.46       |

Table 17: Transfer Attack on Mistral-7B. Evaluating attack phrase SUMM
ABS OVE. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 1.64 | 1.42 | 1.45 | 1.46 | 1.44 | 1.41 | 1.40 | 1.54 | 1.50 | 1.51 | 1.43 | 1.37 | 1.47 | 1.44 | 1.54 | 1.57 | 1.47 | 8.50        |
| 1       | 1.59 | 1.44 | 1.42 | 1.48 | 1.45 | 1.44 | 1.40 | 1.53 | 1.49 | 1.50 | 1.42 | 1.39 | 1.44 | 1.46 | 1.53 | 1.52 | 1.47 | 8.46        |
| 2       | 1.62 | 1.45 | 1.41 | 1.50 | 1.46 | 1.46 | 1.39 | 1.54 | 1.55 | 1.51 | 1.42 | 1.38 | 1.46 | 1.49 | 1.56 | 1.54 | 1.48 | 8.02        |
| 3       | 1.52 | 1.38 | 1.34 | 1.41 | 1.39 | 1.38 | 1.33 | 1.47 | 1.52 | 1.45 | 1.34 | 1.31 | 1.38 | 1.41 | 1.48 | 1.45 | 1.41 | 10.98       |
| 4       | 1.56 | 1.40 | 1.36 | 1.44 | 1.42 | 1.40 | 1.34 | 1.50 | 1.56 | 1.49 | 1.37 | 1.33 | 1.38 | 1.44 | 1.52 | 1.49 | 1.44 | 10.07       |

Table 18: Transfer Attack on Mistral-7B. Evaluating attack phrase SUMM
ABS CON. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 3.58 | 3.74 | 3.87 | 3.65 | 3.72 | 3.78 | 3.94 | 3.73 | 3.88 | 3.69 | 3.80 | 3.93 | 3.72 | 3.70 | 3.52 | 3.61 | 3.74 | 8.50        |
| 1       | 3.66 | 3.76 | 3.87 | 3.68 | 3.72 | 3.76 | 3.85 | 3.77 | 4.02 | 3.74 | 3.79 | 3.86 | 3.78 | 3.69 | 3.56 | 3.67 | 3.76 | 8.31        |
| 2       | 4.23 | 4.28 | 4.45 | 4.26 | 4.25 | 4.24 | 4.33 | 4.30 | 4.29 | 4.28 | 4.31 | 4.33 | 4.21 | 4.21 | 4.15 | 4.24 | 4.27 | 3.36        |
| 3       | 4.20 | 4.23 | 4.42 | 4.17 | 4.21 | 4.19 | 4.35 | 4.28 | 4.37 | 4.26 | 4.24 | 4.31 | 4.19 | 4.18 | 4.08 | 4.24 | 4.24 | 3.52        |
| 4       | 4.43 | 4.44 | 4.58 | 4.42 | 4.40 | 4.39 | 4.46 | 4.50 | 4.41 | 4.49 | 4.45 | 4.43 | 4.33 | 4.42 | 4.35 | 4.48 | 4.44 | 2.30        |

Table 19: Transfer Attack on Llama-7B. Evaluating attack phrase SUMM ABS
OVE. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 2.39 | 2.38 | 2.38 | 2.36 | 2.37 | 2.39 | 2.38 | 2.38 | 2.27 | 2.36 | 2.38 | 2.38 | 2.36 | 2.38 | 2.37 | 2.39 | 2.37 | 8.50        |
| 1       | 2.38 | 2.39 | 2.37 | 2.38 | 2.39 | 2.39 | 2.37 | 2.38 | 2.31 | 2.37 | 2.38 | 2.37 | 2.39 | 2.38 | 2.38 | 2.40 | 2.38 | 8.16        |
| 2       | 2.38 | 2.39 | 2.38 | 2.38 | 2.39 | 2.38 | 2.36 | 2.38 | 2.31 | 2.38 | 2.37 | 2.36 | 2.40 | 2.39 | 2.38 | 2.40 | 2.38 | 8.16        |
| 3       | 2.39 | 2.39 | 2.37 | 2.39 | 2.39 | 2.38 | 2.36 | 2.39 | 2.36 | 2.38 | 2.37 | 2.36 | 2.43 | 2.39 | 2.40 | 2.39 | 2.38 | 7.81        |
| 4       | 2.40 | 2.39 | 2.37 | 2.39 | 2.39 | 2.38 | 2.36 | 2.38 | 2.34 | 2.38 | 2.38 | 2.36 | 2.41 | 2.40 | 2.40 | 2.39 | 2.38 | 7.82        |

Table 20: Transfer Attack on Llama-7B. Evaluating attack phrase SUMM ABS
CON. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 2.98 | 2.88 | 2.88 | 2.88 | 2.83 | 3.15 | 2.93 | 3.50        |
| 1       | 3.59 | 3.55 | 3.59 | 3.54 | 3.55 | 3.85 | 3.61 | 1.54        |
| 2       | 4.11 | 4.13 | 4.11 | 4.00 | 4.03 | 4.35 | 4.12 | 1.22        |
| 3       | 4.44 | 4.45 | 4.40 | 4.33 | 4.36 | 4.57 | 4.42 | 1.09        |
| 4       | 4.63 | 4.63 | 4.61 | 4.60 | 4.61 | 4.67 | 4.63 | 1.07        |

Table 21: Direct Attack on FlanT5-xl. Evaluating attack phrase TOPIC ABS
OVE. TopicalChat. 6 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 3.38 | 2.54 | 2.90 | 2.94 | 2.67 | 3.73 | 3.02 | 3.50        |
| 1       | 4.92 | 5.00 | 4.85 | 4.88 | 4.88 | 4.60 | 4.85 | 1.21        |
| 2       | 4.58 | 4.71 | 4.90 | 4.69 | 4.75 | 3.96 | 4.60 | 1.53        |
| 3       | 4.50 | 4.77 | 4.75 | 4.71 | 4.48 | 3.96 | 4.53 | 1.61        |
| 4       | 4.35 | 4.69 | 4.67 | 4.69 | 4.44 | 3.06 | 4.32 | 1.86        |

Table 22: Direct Attack on FlanT5-xl. Evaluating attack phrase TOPIC ABS
CNT. TopicalChat. 6 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 2.98 | 2.08 | 2.42 | 2.56 | 2.21 | 3.19 | 2.57 | 3.50        |
| 1       | 3.38 | 2.88 | 3.19 | 3.23 | 2.90 | 3.29 | 3.14 | 2.64        |
| 2       | 3.23 | 2.88 | 3.23 | 3.44 | 2.79 | 3.21 | 3.13 | 2.74        |
| 3       | 3.69 | 3.44 | 3.94 | 3.94 | 3.33 | 3.35 | 3.61 | 2.28        |
| 4       | 2.40 | 2.46 | 2.56 | 2.60 | 1.83 | 2.29 | 2.36 | 3.79        |

Table 23: Transfer Attack on GPT3.5. Evaluating attack phrase TOPIC ABS
OVE. TopicalChat. 6 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 3.38 | 2.54 | 2.90 | 2.94 | 2.67 | 3.73 | 3.02 | 3.50        |
| 1       | 4.92 | 5.00 | 4.85 | 4.88 | 4.88 | 4.60 | 4.85 | 1.21        |
| 2       | 4.58 | 4.71 | 4.90 | 4.69 | 4.75 | 3.96 | 4.60 | 1.53        |
| 3       | 4.50 | 4.77 | 4.75 | 4.71 | 4.48 | 3.96 | 4.53 | 1.61        |
| 4       | 4.35 | 4.69 | 4.67 | 4.69 | 4.44 | 3.06 | 4.32 | 1.86        |

Table 24: Transfer Attack on GPT3.5. Evaluating attack phrase TOPIC ABS
CNT. TopicalChat. 6 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 1.63 | 1.50 | 1.52 | 1.51 | 1.51 | 1.72 | 1.57 | 3.50        |
| 1       | 1.59 | 1.57 | 1.59 | 1.58 | 1.58 | 1.70 | 1.60 | 3.11        |
| 2       | 1.62 | 1.58 | 1.60 | 1.58 | 1.58 | 1.73 | 1.61 | 2.98        |
| 3       | 1.59 | 1.57 | 1.59 | 1.58 | 1.58 | 1.70 | 1.60 | 3.11        |
| 4       | 1.60 | 1.57 | 1.61 | 1.59 | 1.58 | 1.73 | 1.61 | 2.98        |

Table 25: Transfer Attack on Mistral-7B. Evaluating attack phrase TOPIC
ABS OVE. TopicalChat. 6 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 2.15 | 1.85 | 1.97 | 2.03 | 1.81 | 2.25 | 2.01 | 3.50        |
| 1       | 3.33 | 3.30 | 3.32 | 3.27 | 3.24 | 3.36 | 3.30 | 1.23        |
| 2       | 3.02 | 3.09 | 3.17 | 3.11 | 3.12 | 3.25 | 3.13 | 1.33        |
| 3       | 3.11 | 3.10 | 3.16 | 3.19 | 3.15 | 3.44 | 3.19 | 1.26        |
| 4       | 3.23 | 3.29 | 3.34 | 3.28 | 3.28 | 3.19 | 3.27 | 1.22        |

Table 26: Transfer Attack on Mistral-7B. Evaluating attack phrase TOPIC
ABS CNT. TopicalChat. 6 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 2.33 | 2.27 | 2.31 | 2.29 | 2.27 | 2.46 | 2.32 | 3.50        |
| 1       | 2.57 | 2.66 | 2.65 | 2.64 | 2.67 | 2.56 | 2.62 | 1.57        |
| 2       | 3.28 | 3.46 | 3.48 | 3.47 | 3.48 | 3.02 | 3.37 | 1.04        |
| 3       | 3.36 | 3.47 | 3.49 | 3.46 | 3.48 | 3.15 | 3.40 | 1.03        |
| 4       | 3.03 | 3.13 | 3.15 | 3.12 | 3.12 | 2.97 | 3.09 | 1.09        |

Table 27: Transfer Attack on Llama-7B. Evaluating attack phrase TOPIC
ABS OVE. TopicalChat. 6 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 2.60 | 2.58 | 2.61 | 2.62 | 2.59 | 2.61 | 2.60 | 3.50        |
| 1       | 3.28 | 3.35 | 3.35 | 3.34 | 3.34 | 3.23 | 3.31 | 1.02        |
| 2       | 3.20 | 3.35 | 3.40 | 3.36 | 3.34 | 3.06 | 3.28 | 1.08        |
| 3       | 3.31 | 3.50 | 3.52 | 3.47 | 3.46 | 3.19 | 3.41 | 1.03        |
| 4       | 3.11 | 3.40 | 3.40 | 3.36 | 3.33 | 3.01 | 3.27 | 1.17        |

Table 28: Transfer Attack on Llama-7B. Evaluating attack phrase TOPIC
ABS CNT. TopicalChat. 6 candidates.

## Appendix G LLM Prompts

Figure 7 shows the prompts used for absolute scoring via G-EVAL, while
Figure 8 shows the prompt template used for comparative assessment.

Figure 7: G-Eval prompt for assessing consistency in Summeval taken from
[https://github.com/nlpyang/geval](https://github.com/nlpyang/geval).
When adapted to TopicalChat, the word ’summary’ is replaced with
’dialogue’ and further minor details are changed for specific attributes

Figure 8: Comparative assessment prompts based on the simple ones used
in Liusie et al. (2023). displayed is a prompt for coherency assessment,
however different adjectives can be used for different attributes.

## Appendix H Attacking Bespoke Assessment Systems

The focus of the paper is on adversarially attacking zero-shot NLG
assessment systems. However, one practical defence could be to use a
bespoke NLG assessment system that is finetuned to a specific domain.
Zhong et al. (2022b) propose such a bespoke system, Unieval that has
been finetuned for summary assessment evaluation for each attribute on
SummEval. The Unieval system predicts a quality score from 1-5 for each
attribute of assessment. Here we explore attacking each attribute of
Unieval in turn for the SummEval dataset. Interestingly Unieval appears
significantly more robust to these form of adversarial attacks than the
zero-shot NLG systems in the main paper. However, it can be observed
that there is some vulnerability in the Unieval when assessed on the
fluency attribute.

## Appendix I Licensing

All datasets used are publicly available. Our implementation utilizes
the PyTorch 1.12 framework, an open-source library. We obtained a
license from Meta to employ the Llama-7B model via HuggingFace.
Additionally, our research is conducted per the licensing agreements of
the Mistral-7B, GPT-3.5, and GPT-4 models. We ran our experiments on
A100 Nvidia GPU and via OpenAI API.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 0.55 | 0.82 | 0.80 | 0.83 | 0.82 | 0.86 | 0.84 | 0.88 | 0.61 | 0.87 | 0.80 | 0.90 | 0.95 | 0.84 | 0.76 | 0.71 | 0.80 | 8.50        |
| 1       | 0.55 | 0.73 | 0.73 | 0.73 | 0.72 | 0.74 | 0.73 | 0.79 | 0.44 | 0.79 | 0.72 | 0.79 | 0.71 | 0.73 | 0.70 | 0.68 | 0.70 | 12.29       |
| 2       | 0.57 | 0.76 | 0.76 | 0.75 | 0.75 | 0.77 | 0.76 | 0.82 | 0.48 | 0.81 | 0.75 | 0.82 | 0.73 | 0.76 | 0.72 | 0.70 | 0.73 | 11.78       |
| 3       | 0.57 | 0.75 | 0.76 | 0.75 | 0.75 | 0.77 | 0.77 | 0.81 | 0.49 | 0.80 | 0.75 | 0.83 | 0.74 | 0.76 | 0.71 | 0.69 | 0.73 | 11.80       |
| 4       | 0.57 | 0.75 | 0.76 | 0.74 | 0.74 | 0.76 | 0.77 | 0.81 | 0.50 | 0.80 | 0.75 | 0.82 | 0.72 | 0.75 | 0.71 | 0.69 | 0.73 | 11.90       |

Table 29: Direct Attack on Unieval. Evaluating attack phrase SUMM UNI
OVE. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 0.38 | 0.79 | 0.70 | 0.83 | 0.81 | 0.89 | 0.86 | 0.96 | 0.51 | 0.95 | 0.68 | 0.97 | 0.97 | 0.85 | 0.74 | 0.58 | 0.78 | 8.50        |
| 1       | 0.34 | 0.61 | 0.61 | 0.57 | 0.60 | 0.64 | 0.74 | 0.76 | 0.21 | 0.74 | 0.58 | 0.79 | 0.35 | 0.62 | 0.57 | 0.50 | 0.58 | 12.46       |
| 2       | 0.38 | 0.70 | 0.66 | 0.70 | 0.72 | 0.77 | 0.80 | 0.86 | 0.29 | 0.85 | 0.64 | 0.86 | 0.60 | 0.74 | 0.69 | 0.55 | 0.67 | 11.77       |
| 3       | 0.35 | 0.61 | 0.61 | 0.57 | 0.61 | 0.65 | 0.73 | 0.75 | 0.24 | 0.74 | 0.57 | 0.76 | 0.41 | 0.62 | 0.60 | 0.50 | 0.58 | 12.51       |
| 4       | 0.37 | 0.63 | 0.64 | 0.60 | 0.64 | 0.68 | 0.76 | 0.77 | 0.27 | 0.76 | 0.60 | 0.79 | 0.44 | 0.64 | 0.62 | 0.53 | 0.61 | 12.35       |

Table 30: Direct Attack on Unieval. Evaluating attack phrase SUMM UNI
COH. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 0.73 | 0.93 | 0.94 | 0.93 | 0.92 | 0.94 | 0.94 | 0.91 | 0.58 | 0.91 | 0.94 | 0.95 | 0.94 | 0.93 | 0.86 | 0.90 | 0.89 | 8.50        |
| 1       | 0.77 | 0.94 | 0.94 | 0.94 | 0.92 | 0.93 | 0.93 | 0.92 | 0.57 | 0.92 | 0.94 | 0.95 | 0.94 | 0.93 | 0.88 | 0.91 | 0.90 | 8.93        |
| 2       | 0.77 | 0.94 | 0.95 | 0.94 | 0.92 | 0.94 | 0.91 | 0.92 | 0.55 | 0.92 | 0.95 | 0.95 | 0.94 | 0.94 | 0.88 | 0.92 | 0.90 | 7.79        |
| 3       | 0.77 | 0.94 | 0.94 | 0.94 | 0.92 | 0.94 | 0.89 | 0.92 | 0.57 | 0.92 | 0.95 | 0.95 | 0.94 | 0.94 | 0.88 | 0.91 | 0.90 | 8.27        |
| 4       | 0.77 | 0.93 | 0.94 | 0.93 | 0.91 | 0.93 | 0.90 | 0.92 | 0.58 | 0.92 | 0.94 | 0.95 | 0.94 | 0.93 | 0.88 | 0.91 | 0.89 | 9.75        |

Table 31: Direct Attack on Unieval. Evaluating attack phrase SUMM UNI
CON. SummEval. 16 candidates.

| \#words | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | avg  | $`\bar{r}`$ |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| None    | 0.55 | 0.75 | 0.76 | 0.74 | 0.72 | 0.74 | 0.72 | 0.77 | 0.74 | 0.76 | 0.77 | 0.79 | 0.93 | 0.74 | 0.67 | 0.64 | 0.74 | 8.50        |
| 1       | 0.45 | 0.55 | 0.57 | 0.53 | 0.53 | 0.54 | 0.53 | 0.59 | 0.40 | 0.57 | 0.58 | 0.60 | 0.71 | 0.55 | 0.51 | 0.53 | 0.55 | 13.21       |
| 2       | 0.62 | 0.80 | 0.80 | 0.80 | 0.76 | 0.78 | 0.71 | 0.81 | 0.64 | 0.80 | 0.81 | 0.83 | 0.92 | 0.79 | 0.74 | 0.70 | 0.77 | 7.42        |
| 3       | 0.63 | 0.80 | 0.81 | 0.80 | 0.77 | 0.79 | 0.70 | 0.81 | 0.60 | 0.81 | 0.82 | 0.84 | 0.93 | 0.80 | 0.75 | 0.70 | 0.77 | 7.25        |
| 4       | 0.63 | 0.80 | 0.81 | 0.80 | 0.77 | 0.79 | 0.70 | 0.81 | 0.60 | 0.81 | 0.82 | 0.84 | 0.93 | 0.80 | 0.75 | 0.70 | 0.77 | 7.26        |

Table 32: Direct Attack on Unieval. Evaluating attack phrase SUMM UNI
FLU. SummEval. 16 candidates.
