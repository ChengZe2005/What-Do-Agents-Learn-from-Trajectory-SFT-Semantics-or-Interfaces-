# What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces?

Large language models (LLMs) are increasingly evaluated as **interactive agents**. However, standard agent benchmarks can **conflate two different sources of success**:

- **Semantic tool-use**: selecting tools based on their functional meaning and observations.
- **Interface shortcutting**: memorizing *surface* interaction patterns tied to a specific interface (e.g., action/tool names or formats).

This repository implements **PIPE**, an evaluation-time augmentation that *minimally rewrites* environment interfaces while preserving task semantics and backend execution. It also provides **AgentBenchPlus** and **AgentGymPlus**—PIPE-augmented versions of AgentBench and AgentGym—together with **Interface Reliance (IR)**, a diagnostic metric for quantifying preference toward training-time interfaces.

---

## What is PIPE?

**PIPE (Perturb Interface Protocol for Evaluation)** constructs a perturbed counterpart of an environment by rewriting only the **interface specification** (e.g., tool/action identifiers), while keeping:

- task prompts,
- environment dynamics,
- executable tool behavior,
- reward / evaluation logic

**unchanged**.

We include two perturbation families (paper-aligned):

- **Perturbation-1 (Synonym)**: replace each tool/action identifier with a **human-readable synonym** (semantics-preserving).
- **Perturbation-2 (Obfuscation)**: replace identifiers with **meaningless tokens** (e.g., `z1`, `z2`, …), removing lexical cues.

---

## Reproducing the paper results

You will reproduce **two** benchmark suites:

### 1) AgentBenchPlus

```bash
cd PIPE/AgentBenchPlus
# Follow the README inside this folder.
```

### 2) AgentGymPlus

```bash
cd PIPE/AgentGymPlus
# Follow the README inside this folder.
```

---

## Citation

A full citation entry will be added in the camera-ready version.
