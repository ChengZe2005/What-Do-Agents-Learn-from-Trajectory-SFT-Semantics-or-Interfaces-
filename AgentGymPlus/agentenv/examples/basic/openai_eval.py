import json
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import transformers
from tqdm import tqdm

from agentenv.controller import (
    APIAgent,
    Evaluator,
)
from agentenv.envs import (
    AcademiaTask,
    AlfWorldTask,
    BabyAITask,
    MazeTask,
    MovieTask,
    SearchQATask,
    SciworldTask,
    SheetTask,
    SqlGymTask,
    TextCraftTask,
    TodoTask,
    WeatherTask,
    WebarenaTask,
    WebshopTask,
    WordleTask,
)

import os
import random


                                                                        
_DISTURB34_ENVS = {"alfworld", "babyai", "sciworld", "weather", "wordle", "maze"}

try:
                                                                                   
    from agentenv.disturb import DisturbManager
except Exception:                    
    DisturbManager = None


def _disturb34_enabled(task_name: str, disturb: int) -> bool:
    """Return True iff this run is in disturb-3 or disturb-4 for a supported env."""
    try:
        d = int(disturb)
    except Exception:
        return False
    return d in (3, 4) and (task_name or "").lower() in _DISTURB34_ENVS


def _extract_action_from_assistant_content(content: str) -> Optional[str]:
    """Extract the action command string from an assistant message.

    Supports formats:
      - Action:\n<command>
      - Action: <command>
    Returns the extracted <command> or None if not found.
    """
    if not isinstance(content, str) or not content:
        return None

    lines = content.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Action:"):
            tail = s[len("Action:") :].strip()
            if tail:
                return tail
                                                              
            for j in range(i + 1, len(lines)):
                nxt = lines[j].strip()
                if nxt:
                    return nxt
            return None
    return None


def _recover_disturb34_counts_from_conversation(
    task_name: str, conversation: List[Dict[str, Any]]
) -> Tuple[int, int]:
    """Best-effort recovery of disturb-3/4 counters from saved conversations.

    This is only used when:
      - args.disturb in {3,4}
      - The saved json doesn't already contain disturb0_calls/disturb1_calls
      - Or the experience object doesn't expose these fields (older agentenv)

    Note: This is a heuristic. When the counters are available from agentenv,
    we always prefer those.
    """
    task = (task_name or "").lower()

                                                         
    if task == "wordle":
        d0 = 0
        d1 = 0
        for msg in conversation or []:
            if (msg or {}).get("role") != "assistant":
                continue
            content = (msg or {}).get("content", "")
            if not isinstance(content, str):
                continue
            for line in content.splitlines():
                s = line.strip()
                if s.startswith("Action:"):
                    d0 += 1
                elif s.startswith("Guess:"):
                    d1 += 1
        return d0, d1

                                                                
    if DisturbManager is None:
        return 0, 0

                                                                       
    dm = DisturbManager(env_key=task, disturb=3)
    if hasattr(dm, "reset_call_counts"):
        dm.reset_call_counts()

    for msg in conversation or []:
        if (msg or {}).get("role") != "assistant":
            continue
        content = (msg or {}).get("content", "")
        action = _extract_action_from_assistant_content(content)
        if not action:
            continue
                                                                      
        try:
            dm.translate_prefix_command(action)
        except Exception:
                                                           
            pass

    return int(getattr(dm, "disturb0_calls", 0)), int(getattr(dm, "disturb1_calls", 0))


@dataclass
class EvalArguments:
    api_key: str
    base_url: str
    model: str
    inference_file: str = field(metadata={"help": "Test dataset."})
    output_dir: str
    max_tokens: int = field(default=4096)
    temperature: float = field(default=1)
    top_p: float = field(default=1)
    task_name: str = field(
        default="webshop", metadata={"help": "Task name for evaluation"}
    )

                         
    max_round: int = field(
        default=6,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

                            
    disturb: int = field(
        default=0,
        metadata={
            "help": "Interface disturbance mode: 0=none, 1=synonym, 2=token (e.g., search->z1), 3=dual (prompt uses disturb0 + note 0->1), 4=dual (prompt uses disturb1 + note 1->0)."
        },
    )

                            
    env_server_base: str = field(default=None)
    data_len: int = field(default=200)
    timeout: int = field(default=2400)


def main(args):

    DATA_PATH = args["inference_file"]

                           
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "babyai": BabyAITask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask,
        "webarena": WebarenaTask,
        "sqlgym": SqlGymTask,
        "maze": MazeTask,
        "wordle": WordleTask,
        "weather": WeatherTask,
        "todo": TodoTask,
        "movie": MovieTask,
        "sheet": SheetTask,
        "academia": AcademiaTask,
        "searchqa": SearchQATask,
    }

                                       
    task_class = task_classes.get(args["task_name"].lower(), None)
    if task_class is None:
        raise ValueError(f"Unsupported task name: {args.task_name}")

                                
    env_args = {
        "env_server_base": args["env_server_base"],
        "data_len": args["data_len"],
        "timeout": args["timeout"],
        "disturb": args.get("disturb", 0),
    }

                    
    evaluator = Evaluator(
        APIAgent(
            api_key=args["api_key"],
            base_url=args["base_url"],
            model=args["model"],
            max_tokens=args["max_tokens"],
            temperature=args["temperature"],
            top_p=args["top_p"],
        ),
        [task_class(client_args=env_args, n_clients=1)],
    )

    with open(DATA_PATH, "r") as file:
        test_data = json.load(file)

    data_idxs = [int(item["item_id"].split("_")[-1]) for item in test_data]
    random.shuffle(data_idxs)

    total_score = 0.0
    total_success = 0.0

                                                                   
                                              
                            
                                                                                             
    disturb34 = _disturb34_enabled(args.get("task_name", ""), args.get("disturb", 0))
    md_alpha = 1.0
    md_log_sum = 0.0
    md_k = 0

    start_time = time.time()
    os.makedirs(args["output_dir"], exist_ok=True)

    for data_idx in tqdm(data_idxs, total=len(data_idxs), desc="[Evaluation Loop]"):
        out_path = os.path.join(args["output_dir"], f"{args['task_name']}_{data_idx}.json")

                                                  
        try:
            with open(out_path, "r") as f:
                item = json.load(f)
                total_score += item["reward"]
                total_success += item["success"]

                if disturb34:
                    d0 = item.get("disturb0_calls", None)
                    d1 = item.get("disturb1_calls", None)
                    if d0 is None or d1 is None:
                        d0, d1 = _recover_disturb34_counts_from_conversation(
                            args["task_name"], item.get("conversations", [])
                        )
                    md_log_sum += math.log((float(d0) + md_alpha) / (float(d1) + md_alpha))
                    md_k += 1

            continue
        except Exception:
            pass

                                     
        for i in range(3):
            try:
                exps = evaluator.eval(
                    max_rounds=args["max_round"],
                    idxs=[data_idx],
                )
                break
            except Exception as e:
                print(e)
                exps = type("AttrDict", (dict,), {"__getattr__": dict.get})(
                    {
                        "score": 0.0,
                        "success": 0,
                        "experiences": [
                            type("AttrDict", (dict,), {"__getattr__": dict.get})(
                                {
                                    "conversation": [
                                        {"role": "assistant", "content": f"[EVAL ERROR] {repr(e)}"}
                                    ],
                                    "reward": 0,
                                }
                            )
                        ],
                    }
                )

        total_score += exps.score
        total_success += exps.success

        cur_experiences = exps.experiences

                                         
        with open(out_path, "w") as f:
            for exp in cur_experiences:
                conversation = exp.conversation
                cur_reward = exp.reward
                cur_success = 1 if exp.reward == 1 else 0
                item_id = f"{args['task_name']}_{data_idx}"

                record: Dict[str, Any] = {
                    "conversations": conversation,
                    "item_id": item_id,
                    "reward": cur_reward,
                    "success": cur_success,
                }

                                                                      
                if disturb34:
                    d0 = getattr(exp, "disturb0_calls", None)
                    d1 = getattr(exp, "disturb1_calls", None)
                    if d0 is None or d1 is None:
                                                              
                        d0, d1 = _recover_disturb34_counts_from_conversation(
                            args["task_name"], conversation
                        )
                    record["disturb0_calls"] = int(d0) if d0 is not None else 0
                    record["disturb1_calls"] = int(d1) if d1 is not None else 0

                    md_log_sum += math.log(
                        (record["disturb0_calls"] + md_alpha)
                        / (record["disturb1_calls"] + md_alpha)
                    )
                    md_k += 1

                json.dump(record, f, ensure_ascii=False, indent=4)

    process_time = time.time() - start_time

    Score = total_score / len(data_idxs)
    Success = total_success / len(data_idxs)
    print("\n\n==== EVALUATION ====\n")
    print(f"Score: {Score}")
    print(f"Success: {Success}")

                                               
    if disturb34 and md_k > 0:
        MD = math.exp(md_log_sum / md_k)
        print(f"MD(alpha={md_alpha}): {MD}")

    print(f"Time: {process_time} seconds")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(EvalArguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = vars(args)
    print(args)
    print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
