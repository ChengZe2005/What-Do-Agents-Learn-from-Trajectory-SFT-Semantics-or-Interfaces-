# PIPE / AgentGymPlus: Interface Perturbation & Interface Reliance (IR)

This package provides a **drop-in replacement** for AgentGym's `agentenv/` directory to reproduce the evaluation protocol described in:

**What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces?**

It implements:
- **PIPE (Perturb Interface Protocol for Evaluation)**: minimal interface rewrites that preserve task semantics and backend execution.
- **AgentGymPlus**: AgentGym evaluated under PIPE perturbations.
- **Interface Reliance (IR)**: a counterbalanced dual-interface protocol that quantifies preference for training-time interfaces.

---

## 1. Package contents

```
README.md
agentenv/
```

---

## 2. Install / Replace AgentGym's `agentenv/`

1. Clone AgentGym to `</path/to/AgentGym>`.
2. Backup the original folder:

```bash
mv </path/to/AgentGym>/agentenv </path/to/AgentGym>/agentenv.bak
```

3. Copy this package's `agentenv/` into AgentGym:

```bash
cp -r ./agentenv </path/to/AgentGym>/agentenv
```

After replacement, you should be able to run AgentGym as usual. The main difference is that the evaluator script now supports `--disturb`.

---

## 3. Start the AgentGym environment server

Same as standard AgentGym:
1. Start the environment server first.
2. Make sure it is reachable at the URL you pass via:

`--env_server_base "${env_server_base}"`

(Please refer to AgentGym's official instructions for your specific setup.)

---

## 4. Run evaluation

Run the evaluator script:

```bash
python agentenv/examples/basic/openai_eval.py \
  --api_key "${api_key}" \
  --base_url "${base_url}" \
  --model "${model}" \
  --inference_file "${inference_file}" \
  --output_dir "${output_dir}" \
  --task_name "${task_name}" \
  --max_round "${max_round}" \
  --env_server_base "${env_server_base}" \
  --disturb "${disturb}"
```

---

## 5. `--disturb` modes (paper-aligned)

This flag selects the evaluation interface condition.

### 5.1 PIPE / AgentGymPlus perturbation settings

- `disturb=0` — **Origin**
  - No interface perturbation (standard AgentGym interface).

- `disturb=1` — **Perturbation-1 (Synonym)**
  - Replace each action/tool identifier with a **human-readable synonym** (semantics-preserving).

- `disturb=2` — **Perturbation-2 (Obfuscation)**
  - Replace each action/tool identifier with **meaningless tokens** (`z1`, `z2`, ...), removing lexical cues.

### 5.2 Interface Reliance (IR) settings (dual-interface + counterbalanced ordering)

IR is evaluated in a **dual-interface** setup where each underlying action is exposed through two functionally identical aliases:
- **Original interface**: training-time identifiers
- **Synonym interface**: Perturbation-1 identifiers

To control for prompt-order (position) bias, we run the suite twice with reversed ordering:

- `disturb=3` — **IR (Ori → Syn)**
  - Original interface is listed first; synonym aliases second.

- `disturb=4` — **IR (Syn → Ori)**
  - Synonym interface is listed first; original aliases second.

To reproduce the paper's IR, run **both** `disturb=3` and `disturb=4`, then compute IR using the counterbalanced protocol.

---

## 6. Notes (important for reproducing paper results)

### 6.1 Strict perturbation policy

For `disturb=1` and `disturb=2`, we use the **strict** PIPE setting:
- The agent is **not allowed** to use the original (non-perturbed) identifiers.
- If the agent outputs an original identifier under perturbation, the wrapper returns an **Invalid/Deprecated Action** error.

This prevents an agent from “falling back” to memorized training-time strings and is required for the PIPE robustness diagnostics.

### 6.2 Output

All outputs are written to:
- `--output_dir`

Use different output folders for different `disturb` values to avoid overwriting.

---

## 7. Reproducing the paper results (recommended run plan)

1. **PIPE robustness**: run each task under
   - `disturb=0` (Origin)
   - `disturb=1` (Perturbation-1 / Synonym)
   - `disturb=2` (Perturbation-2 / Obfuscation)

2. **Interface Reliance (IR)**: run each task under
   - `disturb=3` (Ori → Syn)
   - `disturb=4` (Syn → Ori)

Then aggregate the two IR runs following the counterbalanced protocol in the paper.
