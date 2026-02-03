

# AgentBenchPlus 


**AgentBench** evaluates **LLM-as-Agent** across multiple interactive environments, covering OS, DB, KG, and web/household domains (ALFWorld, WebShop, Mind2Web).

**AgentBenchPlus (PIPE)** extends AgentBench to diagnose interface reliance. It minimally rewrites environment interfaces while preserving task semantics and execution behavior, enabling a clean separation between semantic tool-use and interface shortcutting.

Key additions in AgentBenchPlus:
- Protocol-level interface perturbations via `disturb_type` (synonym or meaningless alias replacements).
- Counterbalanced alias evaluation for Interface Reliance (IR).


## Quick Start

To quickly understand how the framework works, you can follow the instructions below to run a simple evaluation.

### Step 1. Clone this repo and run the following command to install the requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2. Verify that you have successfully installed the requirements by running the following command:

```bash
python eval.py \
    --task configs/tasks/example.yaml \
    --agent configs/agents/do_nothing.yaml
```

### Step 3. Run Example Assignment

> *HINT: Example Assigment is composed of `gpt-3.5-turbo` and `ExampleTask` defined in [`src/tasks/example_task.py`](./src/tasks/example_task.py).*

You need to fill your [OPENAI KEY](https://platform.openai.com/account/api-keys) in `configs/assignments/example.yaml` first.

```yaml
Authorization: Bearer <%% PUT-YOUR-OPENAI-KEY-HERE %%>
```

Then run the following command:

```bash
python create_assignment.py \
    --assignment configs/assignments/example.yaml
```

And you can see the target assignment bash script from the output like this:

```yaml
[System] Run the following command to start evaluation:
    bash .assignments/<TIMESTAMP>.sh
```

Finally, run the assignment bash script that displayed in the output to start evaluation. After that, you can check your output in the `outputs` folder.

## PIPE Interface Perturbation

AgentBenchPlus adds protocol-level interface rewrites to diagnose interface reliance while keeping task semantics unchanged. You can enable this with `disturb_type` in task parameters:

- `disturb_type: 0` uses the original interface.
- `disturb_type: 1` replaces interface tokens with synonyms.
- `disturb_type: 2` replaces interface tokens with meaningless aliases.

### Interface Reliance (IR)

To compute IR, run the counterbalanced alias tasks with `disturb_type: 100` and `disturb_type: 101` (original vs. synonym order swapped). After both runs finish, compute:

```
IR = exp((mp_alpha_100 + mp_alpha_101) / 2)
```

## Tutorial

For more detailed instructions and advanced usage, please refer to our [tutorial](./docs/tutorial.md).

