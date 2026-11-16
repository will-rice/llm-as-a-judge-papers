# Core LLM-as-a-judge papers

The reference set that defines this repository's scope: methods that use
large language models to evaluate outputs, judge models trained for that, and
studies of how reliable and biased those judges are.

## Sources

- Gu et al., *A Survey on LLM-as-a-Judge* (arXiv:2411.15594). Its §2
  (in-context learning, model selection, post-processing), §3 (improvement
  strategies) and §4 (evaluation of LLM-as-a-judge: agreement with humans,
  bias, adversarial robustness) are the scope. Methods named there are
  listed with the section that first cites them. Base models, datasets,
  safety benchmarks and image-captioning metrics it also cites are left out.
- Added: ChatEval, Prometheus 2, J1, Agent-as-a-Judge, the human-alternative
  study of Chiang and Lee, CoBBLEr, self-preference, length-controlled
  AlpacaEval, judgment-distribution inference and judging bias in reasoning
  models, plus the three surveys of the field.

## Use in the topic rule

`core_papers.json` holds every paper below. The topic rule's similarity gate
accepts a paper only if it is close to at least one of them, and the tests
require every one to pass the keyword rule and to clear the similarity
threshold against the others.

| Paper | arXiv | Year | Source | Title |
|---|---|---|---|---|
| GPTScore | [2302.04166](https://arxiv.org/abs/2302.04166) | 2023 | Gu et al. §3 IMPROVEMENT STRATEGY | GPTScore: Evaluate as You Desire |
| G-Eval | [2303.16634](https://arxiv.org/abs/2303.16634) | 2023 | Gu et al. §3 IMPROVEMENT STRATEGY | G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment |
| LLMs vs human evaluation | [2305.01937](https://arxiv.org/abs/2305.01937) | 2023 | added | Can Large Language Models Be an Alternative to Human Evaluations? |
| LLM-Eval | [2305.13711](https://arxiv.org/abs/2305.13711) | 2023 | Gu et al. §3 IMPROVEMENT STRATEGY | LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models |
| InstructScore | [2305.14282](https://arxiv.org/abs/2305.14282) | 2023 | Gu et al. §3 IMPROVEMENT STRATEGY | INSTRUCTSCORE: Explainable Text Generation Evaluation with Finegrained Feedback |
| FairEval | [2305.17926](https://arxiv.org/abs/2305.17926) | 2023 | Gu et al. §2.3 Post-processing | Large Language Models are not Fair Evaluators |
| LM-as-an-Examiner | [2306.04181](https://arxiv.org/abs/2306.04181) | 2023 | Gu et al. §2 BACKGROUND AND METHOD | Benchmarking Foundation Models with Language-Model-as-an-Examiner |
| PandaLM | [2306.05087](https://arxiv.org/abs/2306.05087) | 2023 | Gu et al. §2.1 In-Context Learning | PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization |
| MT-Bench judge | [2306.05685](https://arxiv.org/abs/2306.05685) | 2023 | Gu et al. §2.1 In-Context Learning | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena |
| ChatEval | [2308.07201](https://arxiv.org/abs/2308.07201) | 2023 | added | ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate |
| SocREval | [2310.00074](https://arxiv.org/abs/2310.00074) | 2023 | Gu et al. §3 IMPROVEMENT STRATEGY | SocREval: Large Language Models with the Socratic Method for Reference-Free Reasoning Evaluation |
| CoBBLEr | [2309.17012](https://arxiv.org/abs/2309.17012) | 2023 | added | Benchmarking Cognitive Biases in Large Language Models as Evaluators |
| Auto-J | [2310.05470](https://arxiv.org/abs/2310.05470) | 2023 | Gu et al. §2.1 In-Context Learning | Generative Judge for Evaluating Alignment |
| LLMBar | [2310.07641](https://arxiv.org/abs/2310.07641) | 2023 | Gu et al. §4.1 Agreement with Human Judgments | Evaluating Large Language Models at Evaluating Instruction Following |
| Prometheus | [2310.08491](https://arxiv.org/abs/2310.08491) | 2023 | Gu et al. §2.2 Model Selection | Prometheus: Inducing Fine-grained Evaluation Capability in Language Models |
| BSM | [2310.15123](https://arxiv.org/abs/2310.15123) | 2023 | Gu et al. §3 IMPROVEMENT STRATEGY | Branch-Solve-Merge Improves Large Language Model Evaluation and Generation |
| JudgeLM | [2310.17631](https://arxiv.org/abs/2310.17631) | 2023 | Gu et al. §2.1 In-Context Learning | JudgeLM: Fine-tuned Large Language Models are Scalable Judges |
| CritiqueLLM | [2311.18702](https://arxiv.org/abs/2311.18702) | 2023 | Gu et al. §2 BACKGROUND AND METHOD | CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation |
| Adversarial robustness of judges | [2402.14016](https://arxiv.org/abs/2402.14016) | 2024 | Gu et al. §3 IMPROVEMENT STRATEGY | Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment |
| HD-Eval | [2402.15754](https://arxiv.org/abs/2402.15754) | 2024 | Gu et al. §3 IMPROVEMENT STRATEGY | HD-Eval: Aligning Large Language Model Evaluators Through Hierarchical Criteria Decomposition |
| Length-controlled AlpacaEval | [2404.04475](https://arxiv.org/abs/2404.04475) | 2024 | added | Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators |
| Self-preference | [2404.13076](https://arxiv.org/abs/2404.13076) | 2024 | added | LLM Evaluators Recognize and Favor Their Own Generations |
| Prometheus 2 | [2405.01535](https://arxiv.org/abs/2405.01535) | 2024 | added | Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models |
| Judging the Judges | [2406.12624](https://arxiv.org/abs/2406.12624) | 2024 | Gu et al. §3 IMPROVEMENT STRATEGY | Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges |
| OffsetBias | [2407.06551](https://arxiv.org/abs/2407.06551) | 2024 | Gu et al. §3 IMPROVEMENT STRATEGY | OffsetBias: Leveraging Debiased Data for Tuning Evaluators |
| Trust or Escalate | [2407.18370](https://arxiv.org/abs/2407.18370) | 2024 | Gu et al. §3 IMPROVEMENT STRATEGY | Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement |
| Self-Taught Evaluator | [2408.02666](https://arxiv.org/abs/2408.02666) | 2024 | Gu et al. §1 INTRODUCTION | Self-Taught Evaluators |
| CodeJudge-Eval | [2408.10718](https://arxiv.org/abs/2408.10718) | 2024 | Gu et al. §4.1 Agreement with Human Judgments | CodeJudge-Eval: Can Large Language Models be Good Judges in Code Understanding? |
| DHP | [2408.13704](https://arxiv.org/abs/2408.13704) | 2024 | Gu et al. §3 IMPROVEMENT STRATEGY | DHP Benchmark: Are LLMs Good NLG Evaluators? |
| KUDGE | [2409.11239](https://arxiv.org/abs/2409.11239) | 2024 | Gu et al. §4.1 Agreement with Human Judgments | LLM-as-a-Judge &amp; Reward Model: What They Can and Cannot Do |
| CALM | [2410.02736](https://arxiv.org/abs/2410.02736) | 2024 | Gu et al. §3 IMPROVEMENT STRATEGY | Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge |
| Null-model cheating | [2410.07137](https://arxiv.org/abs/2410.07137) | 2024 | Gu et al. §3 IMPROVEMENT STRATEGY | Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates |
| Agent-as-a-Judge | [2410.10934](https://arxiv.org/abs/2410.10934) | 2024 | added | Agent-as-a-Judge: Evaluate Agents with Agents |
| JudgeBench | [2410.12784](https://arxiv.org/abs/2410.12784) | 2024 | Gu et al. §4.1 Agreement with Human Judgments | JudgeBench: A Benchmark for Evaluating LLM-based Judges |
| Survey (Gu) | [2411.15594](https://arxiv.org/abs/2411.15594) | 2024 | survey | A Survey on LLM-as-a-Judge |
| Survey (Li, generation to judgment) | [2411.16594](https://arxiv.org/abs/2411.16594) | 2024 | survey | From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge |
| Survey (LLMs-as-judges) | [2412.05579](https://arxiv.org/abs/2412.05579) | 2024 | survey | LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods |
| Judgment distribution | [2503.03064](https://arxiv.org/abs/2503.03064) | 2025 | added | Improving LLM-as-a-Judge Inference with the Judgment Distribution |
| LRM judging bias | [2504.09946](https://arxiv.org/abs/2504.09946) | 2025 | added | Assessing Judging Bias in Large Reasoning Models: An Empirical Study |
| J1 | [2505.10320](https://arxiv.org/abs/2505.10320) | 2025 | added | J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning |
| TrustJudge | [2509.21117](https://arxiv.org/abs/2509.21117) | 2025 | Gu et al. §3 IMPROVEMENT STRATEGY | TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them |
