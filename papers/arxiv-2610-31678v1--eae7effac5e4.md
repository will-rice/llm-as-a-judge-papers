---
identifier: arxiv:2610.31678v1
title: "Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation"
authors:
  - Ancil Crayton
published: "2026-10-01T03:02:54+00:00"
url: https://arxiv.org/abs/2610.31678v1
source: arxiv
doi: null
arxiv_id: 2610.31678v1
categories:
  - cs.AI
  - cs.CL
  - cs.HC
---

\workshoptitle

Trustworthy AI for Good (AI4GOOD)

# Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation

Ancil Crayton Affiliation: Booz Allen Hamilton
Email: [crayton.ancil@gmail.com](mailto:)

###### Abstract

Many LLM studies compare how models respond to different follow-up
prompts. For example, one prompt may neutrally ask a model to
reconsider, while another insists that an invented film really exists.
The difference is used to measure the effect of conversational pressure.
An automatic scorer then labels whether each response makes an
unsupported factual claim. Researchers commonly validate the scorer with
one overall accuracy or $`F_{1}`$ score. However, high overall
performance can hide prompt-specific mistakes: if the scorer makes more
errors after one prompt, the measured difference can be too large, too
small, or point in the wrong direction. An audit containing mostly
difficult responses can also misrepresent performance in the full study.
We derive how such errors alter a prompt comparison and examine the
problem in UUP-Bench using six models, 180 invented-entity questions,
and six prompt conditions. In a balanced audit of 288 responses, overall
$`F_{1}`$ was $`.916`$. Yet, depending on the prompt, the scorer flagged
2.6%–33.3% of responses that the primary LLM rater judged not to contain
an unsupported claim; these prompt-specific estimates were uncertain. In
a second audit, the scorer and two separate LLM raters evaluated the
same 300 model–question cases under each prompt. An apparent 9.3-point
increase from asserting certainty rather than neutrally asking the model
to reconsider fell to 2.3 points under both raters. The primary-rater
interval included zero, so it did not establish an increase. Two other
prompt comparisons kept their direction under both raters, while an
attempted prompt-specific correction remained too uncertain for a firm
conclusion. Researchers should therefore validate the exact difference
they plan to report, not just the scorer as a whole: audit both prompts
on matching cases and include scorer and rater uncertainty in the final
result.

## 1 Introduction

Consider an evaluation that asks a model about a film that does not
exist. After the model expresses uncertainty, one prompt asks it to
reconsider; another insists that the film exists and requests a direct
answer. The study compares how often the model makes an unsupported
claim after each prompt. If the automatic scorer treats hypothetical
answers differently in the two conditions, the measured difference
combines model behavior with scoring error. Pooling the conditions can
still produce a strong $`F_{1}`$ because errors in one are hidden by
correct labels in the other.

We call each prompt condition an _arm_ and the difference between two
arms a _contrast_. Interactive evaluations use contrasts to support
claims about pressure, persuasion, or robustness. The claim concerns the
difference, not the scorer’s average accuracy. A scorer can therefore
look reliable overall while overstating, understating, or changing the
direction of the result (Figure 1).

Sampling and routing add further risks. A hard-case audit can reveal
failures without estimating accuracy in the study population. In a
hybrid pipeline, rules score some responses while ambiguous ones go to a
fallback LLM, or routed LLM judge; prompt wording can change both its
errors and how often it is used. We examine these problems in UUP-Bench
using six models, 180 questions about constructed nonexistent entities,
and six prompt conditions. A 550-response stress audit finds failures, a
balanced 288-response audit targets an equal-model population, and a
300-key paired audit compares automatic labels with two independent LLM
raters on identical model–item pairs.

We derive how each arm’s false positives and false negatives enter the
reported contrast, test same-key comparisons, and show why sparse
calibration can leave a correction too uncertain to use. The resulting
protocol retains, qualifies, rejects, or quarantines a comparison based
on its construct, target population, audit sample, paired evidence, and
uncertainty. These analyses are exploratory diagnostics, not a scorer
leaderboard.

Figure 1: Why high overall scorer accuracy can hide a distorted
comparison (schematic). Here $`Y`$ is the label from the primary LLM
rater (Mistral Large 3 2512), and $`\widetilde{Y}`$ is the
automatic-pipeline label, for treatment arm $`a`$ and comparator $`c`$.
Pooled $`F_{1}`$ can look strong while the arms have different signed
errors, $`b_{j}{=}E[\widetilde{Y}-Y\mid A{=}j]`$, as the unequal false
positives illustrate. The contrast error is
$`\widetilde{\Delta}_{a,c}-\Delta_{a,c}{=}b_{a}-b_{c}`$. All counts and
rates are illustrative, not study estimates.

## 2 How scoring errors distort a prompt comparison

### 2.1 Exact accounting of comparison bias

The concrete problem comes before the notation. The automatic pipeline
can add more false positives, or miss more positives, in one arm than in
the other. The reported arm difference then combines the outcome
contrast with a difference in scoring error.

Let $`A\in\{a,c\}`$ identify the treatment and comparator. Let
$`Y\in\{0,1\}`$ be the primary-rater label and $`\widetilde{Y}`$ the
automatic-pipeline label. All probabilities refer to a declared target
population $`\mathcal{P}`$ over models, items, prompts, and turns.
Define

```math
\displaystyle p_{j} \displaystyle=P_{\mathcal{P}}(Y=1\mid A=j), \displaystyle\widetilde{p}_{j} \displaystyle=P_{\mathcal{P}}(\widetilde{Y}=1\mid A=j),
```

```math
\displaystyle\alpha_{j} \displaystyle=P(\widetilde{Y}=1\mid Y=0,A=j), \displaystyle\beta_{j} \displaystyle=P(\widetilde{Y}=0\mid Y=1,A=j). \tag{1}
```

Conditioning on $`Y`$ gives

```math
\widetilde{p}_{j}=(1-\beta_{j})p_{j}+\alpha_{j}(1-p_{j}). \tag{2}
```

Therefore, for $`\Delta_{a,c}=p_{a}-p_{c}`$ and
$`\widetilde{\Delta}_{a,c}=\widetilde{p}_{a}-\widetilde{p}_{c}`$,

```math
\boxed{\widetilde{\Delta}_{a,c}-\Delta_{a,c}=b_{a}-b_{c}},\qquad b_{j}=\alpha_{j}(1-p_{j})-\beta_{j}p_{j}. \tag{3}
```

Here $`b_{j}`$ is signed arm bias: false positives raise it, while false
negatives lower it. Pooled error constrains a contrast only when it also
implies $`b_{a}\approx b_{c}`$ for the target arms. Even equal
sensitivity and specificity can produce unequal bias when prevalence
differs. Conversely, an automatic scorer can err often in both arms
while preserving their difference. Neither pooled $`F_{1}`$ nor two
per-arm $`F_{1}`$ values therefore validate a contrast by themselves.

### 2.2 Why the audit sample and fallback judge matter

Evaluation metrics inherit the way the audit was sampled. Let $`S=1`$
mean that a row is included. For sensitivity $`\mathrm{Se}_{a,S}`$,
specificity $`\mathrm{Sp}_{a,S}`$, and audit prevalence $`\pi_{a,S}`$,

```math
\mathrm{PPV}_{a,S}=\frac{\mathrm{Se}_{a,S}\pi_{a,S}}{\mathrm{Se}_{a,S}\pi_{a,S}+(1-\mathrm{Sp}_{a,S})(1-\pi_{a,S})}. \tag{4}
```

Oversampling predicted positives, difficult responses, selected arms, or
routed cases changes $`\pi_{a,S}`$. It may also change errors within
label groups. A hard-case audit can reveal failures, but it cannot
estimate target precision without a justified sampling model and
weights. Keeping the scorer version fixed does not make its measured
performance _transportable_, meaning applicable to a different target
population.

Hybrid evaluation adds another mixture. Rules score some rows, while
ambiguous rows go to the routed LLM judge. Let $`R=1`$ denote that
route, $`q_{a}=P(R=1\mid A=a)`$, and
$`e_{ar}=E[\widetilde{Y}-Y\mid A=a,R=r]`$. The second index records the
route: $`e_{a1}`$ is error among routed-judge cases and $`e_{a0}`$ is
error among heuristic-only cases. Averaging over the two routes gives

```math
b_{a}=q_{a}e_{a1}+(1-q_{a})e_{a0}. \tag{5}
```

Either route-specific error or the routing share can change arm bias. A
pooled hybrid score hides both. This breakdown is descriptive. Harder
cases enter the routed LLM judge, so route-specific behavior does not
show the causal effect of using that judge.

We therefore attach validity to

```math
\mathcal{V}=(Y,\ a\!:\!c,\ \mathcal{P},\ \mathcal{S}),
```

where $`\mathcal{S}`$ is the sampling and weighting plan. A complete
record also fixes the automatic pipeline and LLM-rater process. Four
separate gates ask whether the arms measure the same construct, whether
errors were observed by arm, whether the audit transports to
$`\mathcal{P}`$, and whether same-key ratings preserve the contrast.
Changing arm wording, the label schema, the routed LLM judge, the
router, the model mixture, sampling, or the rater process can create a
new validation problem.

#### Relation to prior work.

LLM judges can agree with independent ratings and still have task- and
model-dependent bias (Liu et al., 2023; Zheng et al., 2023).
Meta-evaluation usually pools a benchmark and summarizes that agreement.
We ask whether signed error preserves one declared prompt comparison.
The outcome must also be explicit: factual support, abstention, and
uncertainty are different labels (Min et al., 2023; Madhusudhan et al.,
2025; Yona et al., 2024). Sensitivity/specificity correction and
prediction-powered inference can use LLM-rater labels (Rogan and Gladen,
1978; Angelopoulos et al., 2023). Neither repairs a changed construct or
an audit that does not apply to the target cases.

#### Concurrent companion paper.

An anonymized companion studies which model–item pairs continue after a
Soft cue (Anonymous, 2026). It shares the panel but asks a different
question and uses a different empirical analysis. Here we declare the
target population first, then ask whether the automatic pipeline
preserves the difference between two prompt conditions.

## 3 Case study and audit design

UUP-Bench evaluates Claude Sonnet 5, Gemma 3 4B, Llama 4 Maverick,
GPT-5.6 Luna, Qwen3-32B, and Qwen3-8B on 180 constructed
nonexistent-entity items (nonreferents) drawn from cities, films, and
scientists. We screened names with lexical checks, knowledge bases, and
manual review. “Clean” means the retained screened layer; it does not
mean perfect undetectability. The six audited arms are Spontaneous, Soft
cue, reconsider Neutral, instructional-license Existence, Confidence,
and Authority. The positive outcome is an entity-specific unsupported
assertion.

Our design assigns four distinct roles. The _automatic pipeline_ applies
deterministic and heuristic rules. It sends ambiguous responses to the
_routed LLM judge_, DeepSeek V4 Pro. A separate audit process uses the
_primary LLM rater (Mistral Large 3 2512)_ and the _secondary LLM rater
(MiniMax M3)_ at $`T{=}0`$ with the written unsupported-assertion
rubric. The raters see response text and item metadata, but not
hypothesis names or automatic labels. The primary LLM rater supplies the
comparison labels throughout. Independently collected secondary-rater
labels measure disagreement, and we do not adjudicate them into
consensus. Neither rater family is otherwise used as an evaluated
endpoint or automatic judge, so the audits provide unused-family
cross-model evidence. Neither rater is human, and neither supplies
ground truth.

The first audit has 550 rated responses and deliberately includes extra
pressure-arm and routed-LLM-judge cases. Across it, TP/FP/FN/TN are
$`111/52/58/329`$ ($`P=.681`$, $`R=.657`$, $`F_{1}=.669`$). The routed
LLM judge handles $`264/550=48.0\%`$. The heuristic-only slice has
$`F_{1}=.762`$ ($`n=286`$), versus $`.545`$ for the routed slice
($`n=264`$). These values describe a stress sample. Harder cases enter
the routed LLM judge, so the difference does not show that routing
causes errors.

The representative audit samples 48 clean responses per arm and eight
per model, for $`n=288`$. It estimates behavior for an equal-model
target, not provider traffic, future endpoints, or every prompt mixture.
Both LLM raters label a 96-row subset from Neutral, Existence,
Confidence, and Authority. They agree on $`94/96=.979`$, with Cohen’s
$`\kappa=.958`$. Spontaneous and Soft cue have no secondary-rater labels
in this representative audit. The 288-row audit and all analyses based
on it are exploratory.

### 3.1 Audit sampling changes the reported accuracy

Precision changes by different amounts when the audit sample changes
(Figure 2). Neutral rises from $`.667`$ ($`n=81`$ hard rows) to
$`23/24=.958`$ on 48 representative rows. Authority rises from $`.633`$
($`n=111`$) to $`28/29=.966`$, while Spontaneous rises from $`.684`$ on
24 hard rows to $`27/34=.794`$. Wilson intervals are wide and often
overlap. These summaries describe different sampling plans. They neither
test temporal drift nor establish a plan-by-arm interaction.

Figure 2: Arm precision changes when the audit sample changes. Points
show $`\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})`$ against the primary LLM
rater (Mistral Large 3 2512). Bars are 95% Wilson intervals over
predicted positives. The representative audit has 48 rows per arm and
balances six models. The enriched hard audit contributes 478 plotted
rows out of 550 total. From hard to representative sampling, Neutral
rises .667$`\rightarrow`$.958, Authority rises .633$`\rightarrow`$.966,
and Spontaneous rises .684$`\rightarrow`$.794. The wide, often
overlapping intervals do not establish a plan-by-arm interaction.

### 3.2 Why errors must be measured by prompt condition

The balanced audit makes the pooled-versus-arm problem concrete.
TP/FP/FN/TN are $`120/18/4/146`$, agreement is $`266/288=.924`$, and
pooled $`F_{1}=.916`$. Yet observed arm false-positive rates range from
$`.026`$ to $`.333`$ (Table 1). Existence precision is $`15/18=.833`$
($`[.608,.942]`$), while Authority precision is $`28/29=.966`$
($`[.828,.994]`$). Their FPRs are $`3/32=.094`$ and $`1/20=.050`$. A
Fisher test does not find this particular difference significant
($`p=1.000`$).

| Arm         | TP/FP/FN/TN | Precision \[95% CI\] | FPR \[95% CI\]     | $`\widehat{b}_{j}`$ |
| ----------- | ----------- | -------------------- | ------------------ | ------------------- |
| Spontaneous | 27/7/0/14   | .794 \[.632,.897\]   | .333 \[.172,.546\] | +.146               |
| Soft cue    | 7/1/3/37    | .875 \[.529,.978\]   | .026 \[.005,.135\] | $`-.042`$           |
| Neutral     | 23/1/0/24   | .958 \[.798,.993\]   | .040 \[.007,.195\] | +.021               |
| Existence   | 15/3/1/29   | .833 \[.608,.942\]   | .094 \[.032,.242\] | +.042               |
| Confidence  | 20/5/0/23   | .800 \[.609,.911\]   | .179 \[.079,.356\] | +.104               |
| Authority   | 28/1/0/19   | .966 \[.828,.994\]   | .050 \[.009,.236\] | +.021               |

Table 1: Arm-level automatic-pipeline accuracy in the balanced
representative audit ($`48`$ rows per arm; eight per model). Confusion
counts compare automatic unsupported-assertion labels with labels from
the primary LLM rater (Mistral Large 3 2512). Precision and FPR use 95%
Wilson intervals; signed arm bias is
$`\widehat{b}_{j}=(\mathrm{FP}-\mathrm{FN})/48`$. Although pooled
$`F_{1}=.916`$, FPR spans .026–.333, bias spans $`-.042`$–$`+.146`$, and
precision spans .794 (Spontaneous) to .966 (Authority).

The FPR analysis uses the 164 rows that the primary LLM rater labels
negative. An exploratory linear-probability model includes fixed effects
for arm and model and clusters its CR1 covariance by item. It does not
reject equal arm FPRs ($`\chi^{2}_{5}=9.10`$, $`p=.1052`$; 102 item
clusters). Of 15 pairwise comparisons, Soft cue minus Spontaneous has
the largest model-adjusted difference, $`-.325`$ (Holm $`p=.1572`$).
This imprecise estimate is relative to the primary LLM rater in the
audited six-model panel; it neither identifies a cause nor establishes
pairwise heterogeneity. Under an independent-proportion approximation,
the observed raw extreme needs 20 primary-rater-negative rows per arm. A
10-point difference around $`.20`$–$`.30`$ needs 292, before extra costs
from model clustering.

The corresponding unpaired bias diagnostics,
$`(\mathrm{FP}-\mathrm{FN})/48`$, range from $`-.042`$ to $`+.146`$. The
difference in bias is $`+.083`$ for Confidence$`-`$Neutral and $`-.021`$
for Authority$`-`$Existence. Soft cue has three false negatives among
ten primary-rater positives, and Existence has one among sixteen, so
sensitivity is imprecise. These values diagnose the audit slice; they
are not corrected panel effects.

Existence also fails the construct gate. Its instruction permits
hypothetical completion. That changes which answers comply with the
prompt instead of merely adding scoring noise to Neutral. In a clean
unpaired slice, Authority$`-`$Existence is $`+22.9`$ points under the
automatic pipeline and $`+25.0`$ points under primary-rater labels. The
hard slice gives $`-19.0`$ and $`-5.0`$ points. These diagnostics have
no contrast interval and are not effect estimates. A separate paired
audit of Authority$`-`$Existence (registered H3) using primary-rater
labels is positive ($`+.150`$, item-clustered 95% CI $`[+.061,+.279]`$,
$`n=100`$ pairs). H3 remains quarantined because primary-rater labels
cannot turn an instructional-license comparator into a neutral pressure
contrast.

Claude Sonnet 5 acts as an alternative automatic judge, not as the
secondary LLM rater (Table 6). It rescores the same 288 rows. It agrees
with the automatic pipeline on $`259/288=.899`$ and with the primary LLM
rater on $`273/288=.948`$; TP/FP/FN/TN are $`119/10/5/154`$
($`F_{1}=.941`$). It repeats the relevant ordering: Neutral precision is
$`23/23=1.000`$, and Existence precision is $`16/19=.842`$. This
evidence is correlated, not independent. Claude shares the rows,
primary-rater labels, and framing, and Claude Sonnet 5 is also an
evaluated panel model. The result only shows that the ordering is not
unique to one automatic implementation. It adds no independent LLM-rater
process or cross-family transport guarantee.

## 4 Does error change the comparison on the same cases?

Per-arm metrics explain error. The same-key contrast check asks the more
direct question: does that error change the contrast on identical keys?
A seeded, model-balanced audit contains 300 complete clean item–model
keys, with 50 per model. The same keys appear in four continuation arms
after the same Soft-cue response. Neutral asks for reconsideration.
Certainty-only states confidence in the model’s knowledge. Confidence
also states that the entity exists and requests an unhedged answer.
Existence$`\times`$Directness tells the model to assume existence and
answer directly. Both LLM raters independently label all 1,200
responses. Item-clustered bootstrap intervals preserve pairing, and the
second rater directly tests cross-family sensitivity on the target rows.

| Contrast                      | Auto-pipeline RD | Primary-rater RD \[95% CI\]       | Decision                     |
| ----------------------------- | ---------------- | --------------------------------- | ---------------------------- |
| Cert-only $`-`$ Neutral       | +.093            | $`+.023`$ \[$`-.059`$,$`+.053`$\] | attenuated; interval spans 0 |
| Conf $`-`$ Exist$`\times`$Dir | $`-.163`$        | $`-.167`$ \[$`-.207`$,$`-.081`$\] | sign retained                |
| Conf $`-`$ Neutral            | +.113            | $`+.070`$ \[$`+.001`$,$`+.112`$\] | sign retained                |

Table 2: Same-key automatic-pipeline and primary-rater contrast
decisions ($`300`$ paired keys per row; 50 per model). Risk difference
(RD) is the treatment rate minus the comparator rate for unsupported
assertions. Primary-rater intervals use an item-clustered bootstrap. The
primary LLM rater is Mistral Large 3 2512. Certainty-only$`-`$Neutral
attenuates +.093$`\rightarrow+.023`$ with an interval spanning zero.
Confidence$`-`$Neutral remains positive (+.113/+.070), and
Confidence$`-`$Existence$`\times`$Directness remains negative
($`-.163/-.167`$). Existence$`\times`$Directness also changes the
comparator construct.

The gate withholds the raw Certainty-only$`-`$Neutral headline:
automatic $`+.093`$ attenuates to primary-rater $`+.023`$
($`[-.059,+.053]`$), and the secondary-rater estimate is also $`+.023`$.
Confidence$`-`$Neutral keeps its positive sign but shrinks to $`+.070`$
($`[+.001,+.112]`$); the secondary estimate is $`+.107`$.
Confidence$`-`$Existence$`\times`$Directness keeps a negative sign, from
$`-.163`$ automatic to $`-.167`$ under primary-rater labels
($`[-.207,-.081]`$) and $`-.190`$ under secondary-rater labels. No
contrast reverses sign. These decisions apply only to the labeled
factorial contrasts. They do not repair H3’s changed comparator.

### 4.1 Correction does not remove calibration uncertainty

After the construct and transport checks, Confidence–Neutral permits a
separate-arm diagnostic. We apply a sensitivity-and-specificity
prevalence correction (Rogan–Gladen) separately to each arm:

```math
\widehat{p}_{j}^{\,RG}=\frac{\widetilde{p}_{j}+\widehat{\mathrm{Sp}}_{j}-1}{\widehat{\mathrm{Se}}_{j}+\widehat{\mathrm{Sp}}_{j}-1}.
```

It changes automatic RD from $`+.113`$ to $`+.032`$, near the
primary-rater value of $`+.070`$ (Table 3). A 200,000-draw Beta
posterior calculation with a Jeffreys prior carries uncertainty from the
target rate, sensitivity, and specificity into the result. It includes
zero-FN cells instead of treating an estimated sensitivity of one as
known. The interval is $`[-.146,+.197]`$, with $`P(\Delta>0)=.685`$.
Forty-eight calibration rows per arm cannot support a precise corrected
headline. Prediction-powered inference adds no information because the
primary LLM rater labels all 300 target keys. Using the separate 288-row
audit as its correction term would require an unsupported transport
assumption.

| Arm/contrast             | Auto   | Primary | Audit Se/Sp | Corrected \[95%\]        |
| ------------------------ | ------ | ------- | ----------- | ------------------------ |
| Neutral                  | 0.397  | 0.400   | 1.000/0.960 | 0.372 \[0.261,0.449\]    |
| Confidence               | 0.510  | 0.470   | 1.000/0.821 | 0.403 \[0.240,0.523\]    |
| Confidence $`-`$ Neutral | +0.113 | +0.070  | —           | +0.032 \[-0.146,+0.197\] |

Table 3: Arm-aware correction for the 300-key Confidence$`-`$Neutral
target. “Auto” gives automatic-pipeline positive rates, and “Primary
rater” gives rates from the primary LLM rater (Mistral Large 3 2512).
Sensitivity and specificity from 48 representative rows per arm enter an
arm-specific Rogan–Gladen correction. Automatic RD +.113 becomes +.032,
near the primary-rater RD of +.070, but the exploratory 95% interval is
\[$`-.146,+.197`$\]. The calculation only approximates uncertainty from
sparse calibration and dependence.

LLM-rater uncertainty remains separate from automatic-scorer
uncertainty. Across all 1,200 factorial labels, the primary and
secondary LLM raters disagree on $`57/1200=4.75\%`$; arm-specific
disagreement ranges from $`1.7\%`$ to $`6.7\%`$. Their contrast
estimates are close: primary/secondary RDs are $`+.023/+.023`$,
$`+.070/+.107`$, and $`-.167/-.190`$ (Table 8). This does not identify
label error or make either rater preferable. Seven single-label changes
among 600 arm labels can move Certainty-only$`-`$Neutral to zero.
Confidence$`-`$Neutral requires 21, and
Confidence$`-`$Existence$`\times`$Directness requires 50 (Table 7).
Under illustrative five-point per-arm prevalence-error budgets, the RD
bounds are $`[-.077,+.123]`$, $`[-.030,+.170]`$, and $`[-.267,-.067]`$,
respectively. The budget marks breakpoints rather than turning
disagreement into a ground-truth error rate. Only the third sign
survives.

## 5 How to decide whether a comparison is supported

Apply the protocol separately to every intended headline.

#### 1. Specify the measurement record.

Before seeing effects, define the positive label and unit. Reproduce the
treatment and comparator prompts. State the pairing key, target models,
items, domains, turns, weights, automatic-pipeline and router versions,
audit design, and LLM-rater process. Prespecify a meaningful tolerance
$`\epsilon`$ for automatic–primary-rater contrast bias. State whether
success means preserving the sign or showing equivalence.

#### 2. Separate target sampling from failure discovery.

Draw a probability sample, or an explicitly balanced sample, from both
arms. Record inclusion probabilities or target weights. Preserve
identical treatment–control keys. A separate hard sample can find
failures, but applying its metrics to the target requires a justified
model.

#### 3. Construct an LLM-rater process.

Blind LLM raters to hypotheses, expected directions, and automatic
labels. Have both raters independently label a declared subset. Report
counts, agreement, $`\kappa`$, and the adjudication policy. Rater
disagreement is outcome uncertainty, not a nuisance to omit.

#### 4. Report errors by arm and route.

Publish TP/FP/FN/TN, intervals, prevalence, missingness, and routing
shares for each arm. For a hybrid scorer, also report behavior within
the rule and routed-LLM-judge routes and the quantities in Equation 5.
Pooled $`F_{1}`$ cannot replace this disclosure.

#### 5. Test the comparison on identical cases.

On the target keys and weights, estimate
$`\widehat{\Delta}^{\rm auto}`$, $`\widehat{\Delta}^{\rm ref}`$, and
$`\widehat{d}=\widehat{\Delta}^{\rm auto}-\widehat{\Delta}^{\rm ref}`$
with clustering that matches the design. An unpaired slice is only
diagnostic. Here “ref” denotes the primary-rater contrast.

#### 6. Apply four gates and a disposition.

_Construct_ asks whether both prompts implement one intended comparison.
_Observation_ asks whether the audit measured errors by arm.
_Applicability (transport)_ asks whether the audit applies to
$`\mathcal{P}`$. _Same-key_ asks whether the prespecified contrast rule
passes on identical keys. _Retain_ when all gates pass. _Qualify_ when
the stronger rule fails but a prespecified weaker sign rule passes.
_Reject_ when a sufficiently reliable same-key audit contradicts the raw
headline. _Quarantine_ when the construct, LLM-rater process, or audit
cannot answer the claim. Correction is eligible only after the first
three gates and must carry calibration uncertainty forward. It cannot
repair a changed comparator.

#### 7. Version and revalidate.

Changes to the prompt, label, routed LLM judge, router, model panel,
target weights, sampling, or LLM-rater process create a new record.
Preserve exploratory status and earlier versions. Do not relabel
diagnostics as prospective confirmation.

## 6 Discussion and conclusion

The case study shows why trust cannot be assigned to an evaluator in the
abstract. Pooled $`F_{1}=.916`$ coexists with prompt-specific
false-positive rates whose differences remain too imprecise to dismiss.
The stress and balanced audits describe different groups of cases, and
the alternative automatic judge repeats one ordering without providing
an independent check. On identical cases, one automatic comparison
shrinks to an interval that includes zero, while two keep their
direction under both independent LLM raters. The prompt-specific
correction moves toward the primary-rater estimate but remains too
uncertain to replace it.

Validation should therefore attach to a defined outcome, two prompts, a
target set of cases, and an audit sample, together with versioned
automatic and LLM-rater processes. Error counts by prompt help explain a
problem; same-case ratings test whether the reported difference
survives. Neither can rescue prompts that measure different constructs
or an audit that does not apply to the target population. The protocol
uses explicit decisions instead of a universal $`F_{1}`$ threshold.

#### What the case study supports.

The representative audit supports a local statement: in an equal-model
mixture of these six arms, observed FPRs span $`.026`$–$`.333`$, but the
omnibus does not reject equal arm FPRs and modest variation remains
unresolved. The hard audit finds failures under its documented
enrichment and routing; it does not estimate a population error rate.
The alternative automatic judge shows that the Neutral–Existence
ordering is not unique to the automatic pipeline on the same rows.
Shared data, framing, and primary-rater labels make this corroboration
rather than independent validation.

The factorial audit addresses the reported contrasts more directly. On
the same 300 keys, Certainty-only$`-`$Neutral attenuates to an interval
spanning zero, Confidence$`-`$Neutral weakens but stays positive under
both raters, and Confidence$`-`$Existence$`\times`$Directness stays
negative under the illustrative rater-error budget. These outcomes
justify contrast-specific retain, qualify, or quarantine decisions. They
do not identify a globally trustworthy scorer or an error budget for
future prompts. Arm correction cannot replace the fully rated estimate
because calibration uncertainty is wide and sample dependence is only
approximated.

The practical lesson is to match evidence to the decision. For a
benchmark prevalence claim, estimate error in the benchmark population.
For a difference between prompts, rate matched cases and test whether
scoring error changes that difference. If the prompts do not measure the
same construct, the LLM ratings are unstable, or the audit does not
apply to the target cases, quarantine the result instead of turning
pooled agreement into confidence. This discipline keeps automatic
evaluation useful while making each score’s limits explicit.

## Limitations

The representative audit has 48 responses per arm and eight per model.
Precision, FPR, and especially sensitivity intervals are wide. Its
Wilson intervals are row-level descriptive summaries. The exploratory
omnibus has 102 primary-rater-negative item clusters, treats six models
as fixed, and lacks precision for modest arm differences. Factorial
intervals are also exploratory.

The audit labels are uncertain LLM ratings, not human judgments or
ground truth. Both rater families are unused among the evaluated
endpoints and automatic judges, but cross-family agreement cannot
establish correctness. The representative dual subset excludes
Spontaneous and Soft cue, disagreements are not adjudicated, and both
raters label all 1,200 factorial responses. Sensitivity bounds expose
but do not resolve this uncertainty. The “representative” target is only
an equal-model mixture of clean nonreferents in six arms. It is not
provider traffic, future models, or arbitrary prompts. Hard-sample and
route summaries support neither causal nor population claims.

The alternative automatic judge is one correlated frontier model on the
same rows. Rogan–Gladen assumes that arm calibration transports to the
target and handles overlap only approximately. Its interval rules it out
as a replacement headline. The protocol supplies no universal
$`\epsilon`$, and this retrospective study cannot demonstrate
prospective preregistration.

## Data and code availability

Reproduction materials are available at
https://anonymous.4open.science/r/scorer-validity-53E8/. They include
data, code, prompts, metadata, and seeds; no proprietary scorer is
claimed.

## Ethical considerations

Overstated automatic evaluation can mislead benchmarks and model
comparisons. Arm counts and uncertainty reduce that risk. UUP-Bench uses
generated nonreferents, no private conversations, and collision-screened
names. Hosted LLM ratings are neither human labels nor ground truth; no
human subjects took part. Hosted-call costs were not logged
exhaustively, and endpoints are not redistributed.

## References

- Angelopoulos et al. (2023) A. N. Angelopoulos, S. Bates, C.
  Fannjiang, M. I. Jordan, and T. Zrnic Prediction-powered inference.
  Science 382 (6671), pp. 669–674. External Links:
  [Document](https://dx.doi.org/10.1126/science.adi6000) Cited by: §2.2.
- Anonymous (2026) Anonymous Elicited is not spontaneous: how follow-up
  gates change interactive abstention evaluation. Note: Under review
  Cited by: §2.2.
- Hong et al. (2025) J. Hong, G. Byun, S. Kim, and K. Shu Measuring
  sycophancy of language models in multi-turn dialogues. In Findings of
  the Association for Computational Linguistics: EMNLP 2025, Suzhou,
  China, pp. 2239–2259. External Links:
  [Document](https://dx.doi.org/10.18653/v1/2025.findings-emnlp.121),
  [Link](https://aclanthology.org/2025.findings-emnlp.121/) Cited by:
  Appendix G.
- Liu et al. (2023) Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu
  G-eval: NLG evaluation using GPT-4 with better human alignment. In
  Proceedings of the 2023 Conference on Empirical Methods in Natural
  Language Processing, pp. 2511–2522. External Links:
  [Document](https://dx.doi.org/10.18653/v1/2023.emnlp-main.153),
  [Link](https://aclanthology.org/2023.emnlp-main.153/) Cited by: §2.2.
- Madhusudhan et al. (2025) N. Madhusudhan, S. T. Madhusudhan, V. Yadav,
  and M. Hashemi Do LLMs know when to NOT answer? investigating
  abstention abilities of large language models. In Proceedings of the
  31st International Conference on Computational Linguistics, Abu Dhabi,
  UAE, pp. 9329–9345. External Links:
  [Link](https://aclanthology.org/2025.coling-main.627/) Cited by: §2.2.
- Min et al. (2023) S. Min, K. Krishna, X. Lyu, M. Lewis, W. Yih, P.
  Koh, M. Iyyer, L. Zettlemoyer, and H. Hajishirzi FActScore:
  fine-grained atomic evaluation of factual precision in long form text
  generation. In Proceedings of the 2023 Conference on Empirical Methods
  in Natural Language Processing, Singapore, pp. 12076–12100. External
  Links: [Document](https://dx.doi.org/10.18653/v1/2023.emnlp-main.741),
  [Link](https://aclanthology.org/2023.emnlp-main.741/) Cited by: §2.2.
- Rogan and Gladen (1978) W. J. Rogan and B. C. Gladen Estimating
  prevalence from the results of a screening test. American Journal of
  Epidemiology 107 (1), pp. 71–76. External Links:
  [Document](https://dx.doi.org/10.1093/oxfordjournals.aje.a112510)
  Cited by: §2.2.
- Yao et al. (2025) X. Yao, S. Ghosh, G. Lee, W. H. Logian, L. N.
  Cui, E. Podoshev, S. Srivastava, M. Li, A. Sandoval, S. O’Brien, M.
  Saxon, S. Dev, and K. Zhu GASLIGHTBENCH: quantifying LLM
  susceptibility to social prompting. In NeurIPS 2025 Workshop on
  Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities,
  and Scaling, Note: Equal contribution among Yao, Ghosh, Lee, Logian,
  and Cui External Links:
  [Link](https://openreview.net/forum?id=0BYRYwGCbK) Cited by: Appendix
  G.
- Yona et al. (2024) G. Yona, R. Aharoni, and M. Geva Can large language
  models faithfully express their intrinsic uncertainty in words?. In
  Proceedings of the 2024 Conference on Empirical Methods in Natural
  Language Processing, Miami, Florida, USA, pp. 7752–7764. External
  Links: [Document](https://dx.doi.org/10.18653/v1/2024.emnlp-main.443),
  [Link](https://aclanthology.org/2024.emnlp-main.443/) Cited by: §2.2.
- Zheng et al. (2023) L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z.
  Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E.
  Gonzalez, and I. Stoica Judging LLM-as-a-judge with MT-bench and
  chatbot arena. In Advances in Neural Information Processing Systems,
  External Links: [Link](https://openreview.net/forum?id=uccHPGDlao)
  Cited by: §2.2.

## Appendix A Bias and routing derivations

Conditioning on the primary-rater label yields

```math
E[\widetilde{Y}\mid A=j]=(1-\beta_{j})p_{j}+\alpha_{j}(1-p_{j}).
```

Subtracting $`p_{j}`$ gives
$`b_{j}=\alpha_{j}(1-p_{j})-\beta_{j}p_{j}`$. Subtracting the comparator
bias gives Equation 3. Equal $`F_{1}`$ values need not imply equal
$`b_{j}`$: $`F_{1}`$ omits true negatives, while signed bias depends on
prevalence and both error directions.

Likewise,

```math
E[\widetilde{Y}-Y\mid A=a]=\sum_{r\in\{0,1\}}P(R=r\mid A=a)E[\widetilde{Y}-Y\mid A=a,R=r],
```

which gives Equation 5. A pooled hybrid metric can change with routing
shares even when behavior within each route is fixed.

## Appendix B Sampling and sensitivity diagnostics

| Arm         | Hard $`N`$ | Hard precision \[CI\] | Rep. $`N`$ | Rep. precision \[CI\] |
| ----------- | ---------- | --------------------- | ---------- | --------------------- |
| Spontaneous | 24         | .684 \[.460,.846\]    | 48         | .794 \[.632,.897\]    |
| Soft cue    | 74         | .800 \[.584,.919\]    | 48         | .875 \[.529,.978\]    |
| Neutral     | 81         | .667 \[.454,.828\]    | 48         | .958 \[.798,.993\]    |
| Existence   | 63         | .759 \[.579,.878\]    | 48         | .833 \[.608,.942\]    |
| Confidence  | 125        | .625 \[.453,.771\]    | 48         | .800 \[.609,.911\]    |
| Authority   | 111        | .633 \[.455,.781\]    | 48         | .966 \[.828,.994\]    |

Table 4: Arm precision under hard-enriched versus representative
sampling. $`N`$ counts rows labeled by the primary LLM rater (Mistral
Large 3 2512). Precision and 95% Wilson intervals use automatic
predicted-positive denominators. From hard to representative sampling,
Neutral rises .667$`\rightarrow`$.958, Authority rises
.633$`\rightarrow`$.966, and Spontaneous rises .684$`\rightarrow`$.794.
These shifts depend on the sampling plan; they are not a tested
interaction.

The displayed six arms account for 478 of the 550 hard labels; 72
control-layer rows are omitted. We do not inverse-weight the expanded
sample because it was tiered by pressure and routed-LLM-judge status to
find failures.

| Diagnostic                | Result and interpretation                                                        |
| ------------------------- | -------------------------------------------------------------------------------- |
| Arm omnibus               | $`\chi^{2}_{5}=9.10`$, $`p=.1052`$; model fixed effects, clustered by item.      |
| Strongest pair            | Soft cue $`-`$ Spontaneous: $`-.325`$ (Holm $`p=.1572`$); exploratory, 15 pairs. |
| Observed-extreme planning | 20 negatives/arm; independent approximation.                                     |
| 10-point planning         | 292 negatives/arm; independent approximation.                                    |

Table 5: Exploratory arm-FPR heterogeneity and audit-size planning. The
omnibus and 15 pairwise tests use representative rows labeled negative
by the primary LLM rater (Mistral Large 3 2512), model fixed effects,
and item-clustered CR1 covariance. Arm heterogeneity is not resolved
($`p=.1052`$); Soft cue$`-`$Spontaneous is $`-.325`$ (Holm $`p=.1572`$).
Independent planning needs 20 negatives/arm for that raw extreme but 292
for a 10-point difference; these are approximations, not power
guarantees.

## Appendix C Alternative automatic judge details

| Arm         | Pipeline TP/auto$`+`$ ($`P`$) | Claude auto TP/auto$`+`$ ($`P`$) | Claude–primary agreement |
| ----------- | ----------------------------- | -------------------------------- | ------------------------ |
| Spontaneous | 27/34 (.794)                  | 25/26 (.962)                     | .938                     |
| Soft cue    | 7/8 (.875)                    | 9/10 (.900)                      | .958                     |
| Neutral     | 23/24 (.958)                  | 23/23 (1.000)                    | 1.000                    |
| Existence   | 15/18 (.833)                  | 16/19 (.842)                     | .938                     |
| Confidence  | 20/25 (.800)                  | 18/21 (.857)                     | .896                     |
| Authority   | 28/29 (.966)                  | 28/30 (.933)                     | .958                     |

Table 6: Automatic pipeline versus the Claude Sonnet 5 alternative
automatic judge on the same 288 representative rows. Scorer columns give
true positives over predicted positives, with precision in parentheses,
against the primary LLM rater (Mistral Large 3 2512). Agreement is also
with that rater. Claude gives Existence precision .842 and Neutral
precision 1.000, but it shares rows, framing, and primary-rater labels
and scores some Claude outputs. This is correlated automatic-scoring
evidence, not an independent rating study.

## Appendix D LLM-rater sensitivity details

| Contrast                                       | Primary RD / flips (% labels) | RD bound, $`\epsilon=.05`$ |
| ---------------------------------------------- | ----------------------------- | -------------------------- |
| Certainty-only $`-`$ Neutral                   | $`+.023`$ / 7 (1.2)           | \[$`-.077`$,$`+.123`$\]    |
| Confidence $`-`$ Existence$`\times`$Directness | $`-.167`$ / 50 (8.3)          | \[$`-.267`$,$`-.067`$\]    |
| Confidence $`-`$ Neutral                       | $`+.070`$ / 21 (3.5)          | \[$`-.030`$,$`+.170`$\]    |

Table 7: Sensitivity of same-key primary-rater contrasts to label
changes. The primary LLM rater is Mistral Large 3 2512. “Flips” is the
minimum number of adversarial changes among 600 arm labels that erases
the primary-rater RD sign. Bounds allow $`\pm.05`$ prevalence error per
arm. Certainty-only$`-`$Neutral is erased by seven flips, whereas only
Confidence$`-`$Existence$`\times`$Directness retains its sign under the
five-point budget \[$`-.267,-.067`$\]. The budget is illustrative and is
close to, but not estimated from, the observed 4.75% cross-rater
disagreement.

| Contrast                                       | Primary / secondary RD | Difference |
| ---------------------------------------------- | ---------------------- | ---------- |
| Certainty-only $`-`$ Neutral                   | $`+.023`$ / $`+.023`$  | $`+.000`$  |
| Confidence $`-`$ Existence$`\times`$Directness | $`-.167`$ / $`-.190`$  | $`-.023`$  |
| Confidence $`-`$ Neutral                       | $`+.070`$ / $`+.107`$  | $`+.037`$  |

Table 8: Full factorial sensitivity across unused-family LLM raters.
Mistral Large 3 2512 and MiniMax M3 independently label all 1,200
responses on the same 300 keys. Their RDs differ by at most 3.7 points
across the three contrasts; neither rater supplies ground truth.

## Appendix E Routing details

| Route            | $`N`$ (share) | TP/FP/FN/TN   | $`P`$ | $`R`$ | $`F_{1}`$ |
| ---------------- | ------------- | ------------- | ----- | ----- | --------- |
| Heuristic-only   | 286 (52.0%)   | 72/28/17/169  | .720  | .809  | .762      |
| Routed LLM judge | 264 (48.0%)   | 39/24/41/160  | .619  | .488  | .545      |
| Pooled           | 550 (100%)    | 111/52/58/329 | .681  | .657  | .669      |

Table 9: Route composition and primary-rater-relative performance in the
550-row hard audit. The primary LLM rater is Mistral Large 3 2512, and
the routed LLM judge is DeepSeek V4 Pro. Rows report sample share,
confusion counts, precision, recall, and $`F_{1}`$ for heuristic-only
and routed-LLM-judge cases. Routed cases are 48.0% with $`F_{1}=.545`$,
versus .762 for heuristic-only cases; pooled $`F_{1}=.669`$. Harder
responses are more likely to be routed, so this difference is
descriptive and not causal.

## Appendix F Existence error taxonomy

The automatic pipeline has three Existence false positives, all
hypothetical abstentions or requests for context. The alternative
automatic judge also has three, all hypothetical or speculative
completions. We assigned categories after reading the response text. The
analysis is descriptive, used one coder, and does not provide another
LLM rating.

| Exclusive response category             | Pipeline FP | Claude FP |
| --------------------------------------- | ----------- | --------- |
| Hypothetical/speculative completion     | 0           | 3         |
| Hypothetical abstention / needs context | 3           | 0         |
| Referent substitution                   | 0           | 0         |
| Total Existence false positives         | 3           | 3         |

Table 10: Exploratory taxonomy of primary-rater-relative Existence false
positives on identical representative rows. The primary LLM rater is
Mistral Large 3 2512. Exclusive categories compare the automatic
pipeline with the Claude Sonnet 5 alternative automatic judge. All three
pipeline false positives are hypothetical abstentions/context requests;
all three alternative-judge false positives are hypothetical/speculative
completions. Categories were single-coded post hoc.

## Appendix G Secondary transfer cases

These cases illustrate the protocol; they do not evaluate any official
benchmark scorer. Both concern our non-original, task-specific
implementations. Neither decision extends to all published systems.

#### GaslightBench.

Our incorrectness heuristic has TP/FP/FN/TN $`=77/132/15/76`$ on
$`n=300`$ primary-rater judgments: precision $`.368`$, recall $`.837`$,
and $`F_{1}=.512`$. Arm $`F_{1}`$ ranges from $`.113`$ on Neutral T2 to
$`.927`$ on Existence T2. Dual-rater $`\kappa=.573`$ does not repair the
heuristic’s disagreement with the primary LLM rater. We quarantine
automatic effects rather than replace them with a primary-rater
headline. Without same-key automatic–primary-rater intervals, the audit
cannot reject a particular contrast.

#### SYCON.

Our non-original automatic acceptance judge has TP/FP/FN/TN
$`=332/21/320/227`$ on $`n=900`$ turn judgments ($`F_{1}=.661`$; dual
$`\kappa=.635`$). Neutral recall is $`65/183=.355`$, versus
$`170/246=.691`$ for Existence. In 60 audited topic–model trajectories,
primary-rater ever-accept is Existence $`60/60=1.000`$ (Wilson
$`[.940,1.000]`$), pressure $`55/60=.917`$ ($`[.819,.964]`$), and
Neutral $`49/60=.817`$ ($`[.701,.894]`$). We retain only this
descriptive ordering. Existence was not the published no-pressure
control, the intervals are marginal rather than paired, and we claim no
reversal of the published conclusion \[Hong et al., 2025\]. We likewise
cite the Gaslight task only as the source setting \[Yao et al., 2025\].

| Case               | $`N`$ | TP/FP/FN/TN    | $`F_{1}`$ | Decision              |
| ------------------ | ----- | -------------- | --------- | --------------------- |
| Gaslight heuristic | 300   | 77/132/15/76   | .512      | quarantine effects    |
| SYCON              | 900   | 332/21/320/227 | .661      | qualify ordering only |

Table 11: Transfer-audit dispositions for two non-original scorers.
$`N`$ counts turn judgments labeled by the primary LLM rater (Mistral
Large 3 2512). Confusion counts and $`F_{1}`$ compare each automatic
implementation with those labels. Gaslight has $`F_{1}=.512`$ and is
quarantined. The SYCON automatic judge has $`F_{1}=.661`$ and supports
only a qualified descriptive ordering. Neither decision applies beyond
the analyzed label and sample.

## Appendix H Suggested contrast card

For each intended headline, archive: (1) outcome, positive class,
rubric, and unit; (2) exact prompts, turn, and pairing key; (3) target
models, items, domains, and weights; (4) automatic-pipeline,
routed-LLM-judge, and routing versions; (5) sampling probabilities or
balancing cells, seed, and analysis status; (6) per-arm and per-route
confusion counts and intervals; (7) same-key automatic and primary-rater
effects; (8) dual-rater counts, agreement, $`\kappa`$, and adjudication;
and (9) the final decision and prohibited extrapolations.

| Candidate claim                | Disposition and binding validity problem                                                                                                             |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| H3 Auth.–Exist.                | Quarantined. The comparator licenses hypothetical completion, so it changes the construct despite representative precision 15/18=.833 \[.608,.942\]. |
| Our Gaslight heuristic effects | Quarantined. Poor overall audit; no same-key automatic–primary-rater contrast intervals.                                                             |
| SYCON reversal                 | No reversal claim. Non-original automatic judge; Neutral recall .355 ($`n=300`$ judgments).                                                          |

Table 12: Claims withheld by the contrast-level validity gate.
Authority$`-`$Existence is quarantined because its prompt permits
hypothetical completion, despite Existence precision .833. Gaslight
lacks same-key intervals. We claim no SYCON reversal because its
automatic judge is non-original and Neutral recall is .355. Reviving
these claims requires a construct-compatible comparator and/or a
target-matched audit, not pooled $`F_{1}`$.
