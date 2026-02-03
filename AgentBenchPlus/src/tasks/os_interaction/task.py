from src.task import Task, DataPiece, Dataset
from src.agent import Agent, Session
import os
import json
import sys
import time
import re
import math
import random
import datetime
import argparse
import requests
import glob
import importlib
from typing import Callable, List, Dict, Any, Tuple, Optional, Union
import docker.models.containers
import docker
import traceback
import struct
import socket
import src.tasks.os_interaction.disturb as disturb
import copy


class Container:
    def __init__(self, image):
        self.image = image
        self.client = docker.from_env()
        self.container: docker.models.containers.Container = self.client.containers.run(
            image, detach=True, tty=True, stdin_open=True, remove=True,
            labels={"created_by": "os-pipeline"}
        )
        self.exec_id = self.client.api.exec_create(self.container.id, "bash --login", stdin=True, tty=True)["Id"]
        self.sock = self.client.api.exec_start(self.exec_id, socket=True)._sock
        self.sock.settimeout(5)
        # clear buffer
        self.sock.recv(1000)

    def __del__(self):
        try:
            # print("Releasing", self.container.name, self.container.id)
            self.container.stop()
        except:
            pass

    def execute(self, command: str):
        class DummyOutput:
            output: bytes
            exit_code: int

            def __init__(self, code, o):
                self.output = o
                self.exit_code = code

        # print("=== EXECUTING ===\n", command)
        if not isinstance(command, str):
            return DummyOutput(-1, b'')
        self.sock.send(command.encode("utf-8") + b'\n')
        # ignore input line
        data = self.sock.recv(8)
        _, n = struct.unpack('>BxxxL', data)
        _ = self.sock.recv(n)
        output = b''
        while True:
            try:
                data = self.sock.recv(8)
                # print(data)
                if not data:
                    break
                _, n = struct.unpack('>BxxxL', data)
                line = self.sock.recv(n)
                output += line
                if re.search(b"\x1b.+@.+[#|$] ", line):
                    break
            except TimeoutError:
                break
            except socket.timeout:
                break
        # replace the very end \x1b.+@.+[#|$] into nothing (required the suffix)
        # output = re.sub(b"\x1b.+@.+[#|$] $", b"", output)
        return DummyOutput(0, output)

    def execute_independent(self, command, *params):
        # print("=== EXECUTING INDEPENDENT ===\n", command)
        language, command = command
        # if params:
        #     print("== Parameters ==\n", params)
        if language == "bash":
            cmd = ["bash", "-c", command]
            if params:
                cmd.append("--")
                cmd.extend(params)
        elif language == "python":
            cmd = ["python3", "-c", command, *params]
        elif language == "c++":
            self.execute_independent(("bash", f"echo \"{json.dumps(command)}\" > /tmp/main.cpp && "
                                              f"g++ -o /tmp/a.out /tmp/main.cpp"), None)
            cmd = ["/tmp/a.out", *params]
        elif language == "c":
            self.execute_independent(("bash", f"echo \"{json.dumps(command)}\" > /tmp/main.cpp && "
                                              f"gcc -o /tmp/a.out /tmp/main.cpp"), None)
            cmd = ["/tmp/a.out", *params]
        else:
            raise ValueError("Unsupported language")
        return self.container.exec_run(cmd)


class JudgeConfig:
    image: str = None
    init_script: List[Tuple[str, str]] = None
    start: Tuple[str, str] = None
    description: str
    check: list = None
    match: dict = None
    example_script: str = None

    def get_evaluation_type(self):
        if self.check:
            return "check"
        elif self.match:
            return "match"

    def get_evaluation_content(self):
        return self.check or self.match

ONE_SHOT=[
    {
        "role": "user",
        "content": 'tell me how many files are in the directory "/etc"?'
    },
    {
        "role": "agent", "content": """Think: To count the files in /etc, I need to print all the files in it.

Act: bash

```bash
ls /etc
```"""
    },
    {
        "role": "user",
        "content": 'The output of the OS:\ncpi cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols selinux tmpfiles.d [truncated because the output is too long]'
    },
    {
        "role": "agent",
        "content": """Think: The output has been truncated because it is too long, so I need to count files by script directly.

Act: bash

```bash
ls -1 /etc | wc -l
```"""
    },
    {
        "role": "user",
        "content": 'The output of the OS:\n220'
    },
    {
        "role": "agent",
        "content": """Think: Now I get the answer, it is 220.

Act: answer(220)"""
    }
]


class OSInteraction(Task):
    def _load_configs(self, config_path, script_root_dir=".") -> List[JudgeConfig]:
        def load_script(script_obj):
            if script_obj is None:
                return None
            if type(script_obj) is str:
                return "bash", script_obj
            if "language" not in script_obj:
                language = "bash"
            else:
                language = script_obj["language"]
            if "file" in script_obj:
                with open(os.path.join(script_root_dir, script_obj["file"]), encoding="utf-8") as f:
                    return language, f.read()
            elif "code" in script_obj:
                return language, script_obj["code"]
            else:
                raise ValueError("Invalid Script Object")

        # 1. handle input file:
        if config_path.endswith(".json"):
            with open(config_path, encoding="utf-8") as f:
                config_raw = json.load(f)
            if isinstance(config_raw, list):
                pass
            elif isinstance(config_raw, dict):
                config_raw = [config_raw]
            else:
                raise ValueError("Invalid Config File")
        elif config_path.endswith(".jsonl"):
            with open(config_path, encoding="utf-8") as f:
                config_raw = [json.loads(line) for line in f.readlines()]
        else:
            raise ValueError("Invalid Config File")

        # 2. handle configs
        configs: list[JudgeConfig] = []
        for item in config_raw:
            config = JudgeConfig()
            config.description = item["description"]
            if "create" in item:
                config.image = item["create"]["image"] if ("image" in item["create"]) else (self.docker_config["localhost"] + "/default")
                if "init" in item["create"]:
                    if type(item["create"]["init"]) is not list:
                        config.init_script = [load_script(item["create"]["init"])]
                    else:
                        config.init_script = [load_script(script_obj) for script_obj in item["create"]["init"]]
                else:
                    config.init_script = []
            else:
                config.image = self.docker_config["localhost"] + "/default"
            if "start" in item:
                config.start = load_script(item["start"])
            evaluation = item["evaluation"]
            if "match" in evaluation:
                if type(evaluation["match"]) is str:
                    config.match = {
                        "answer": evaluation["match"],
                        "strip": True
                    }
                else:
                    config.match = evaluation["match"]
            elif "check" in evaluation:
                if type(evaluation["check"]) is not list:
                    config.check = [load_script(evaluation["check"])]
                else:
                    config.check = [load_script(script_obj) for script_obj in evaluation["check"]]
            else:
                raise ValueError("check or match must exist.")
            if "check" in evaluation and "example" in evaluation:
                config.example_script = load_script(evaluation["example"])
            configs.append(config)
        return configs

    def __init__(self, **kwargs):
        # TODO: load config here
        self.match_problem: bool = kwargs.pop("match_problem", True)
        self.check_problem: bool = kwargs.pop("check_problem", True)
        self.round_limit: int = kwargs.pop("round_limit", 8)
        self.disturb_type: int = kwargs.pop("disturb_type", 0) # 0: no disturbance
        self.data_config = kwargs.pop("data_config", None)
        if not self.data_config:
            raise ValueError("data_config must be set")
        self.docker_config = kwargs.pop("docker_config", None)
        if not self.docker_config:
            raise ValueError("docker_config must be set")
        configs: List[dict[str, Any]] = []
        matches = []
        for item in self.data_config["files"]:
            path = item["problem_file"]
            for file in glob.glob(path):
                if file.endswith(".json") or file.endswith(".jsonl"):
                    matches.append({
                        "problem_file": file,
                        "script_dir": item["script_dir"]
                    })
        self.data_config["files"] = matches
        for item in self.data_config["files"]:
            problem_file, problem_dir = item["problem_file"], item["script_dir"]
            single_file_configs = self._load_configs(problem_file, problem_dir)
            single_file_check_configs = []
            single_file_match_configs = []
            for idx, config in enumerate(single_file_configs):
                if config.check:
                    single_file_check_configs.append({"file": problem_file, "config": config, "index": idx})
                elif config.match:
                    single_file_match_configs.append({"file": problem_file, "config": config, "index": idx})
            print("Load %s, %d problems, %d check problems, %d match problems." % (problem_file, len(single_file_configs), len(single_file_check_configs), len(single_file_match_configs)))
            if self.match_problem:
                configs.extend(single_file_match_configs)
            if self.check_problem:
                configs.extend(single_file_check_configs)
        self.problem_configs = configs
        self.one_shot: bool = kwargs.pop("one_shot", False)

        self.bash_keyword = 'bash'
        self.finish_keyword = 'finish'
        self.answer_keyword = 'answer'
        if self.disturb_type == 1 or self.disturb_type == 6:
            self.bash_keyword = 'exec'
            self.finish_keyword = 'done'
            self.answer_keyword = 'reply'
        elif self.disturb_type == 2 or self.disturb_type == 7:
            self.bash_keyword = 'z1'
            self.finish_keyword = 'z2'
            self.answer_keyword = 'z3'
        elif self.disturb_type == 100 or self.disturb_type == 101:
            self.bash_keyword = ['bash', 'exec']
            self.finish_keyword = ['finish', 'done']
            self.answer_keyword = ['answer', 'reply']

        
        super().__init__(**kwargs)

    def metric(self, prediction: List[Dict], target: List[None]):
        files = []
        for item in self.data_config["files"]:
            file = item["problem_file"]
            if file.endswith(".json") or file.endswith(".jsonl"):
                if file in self.data_config["ignore"]:
                    continue
                file_configs = [config for config in prediction if (config and config["file"] == file)]
                print("File:", file, "Total:", len(file_configs), "Pass:", len([config for config in file_configs if "result" in config and config["result"]]))
                files.append({
                    "file": file,
                    "total": len(file_configs),
                    "pass": len([config for config in file_configs if "result" in config and config["result"]]),
                    "wrong": len([config for config in file_configs if "result" not in config or not config["result"]]),
                })
                files[-1]["acc"] = files[-1]["pass"] / files[-1]["total"] if files[-1]["total"] else 0
        overall = {
            "total": len([config for config in prediction if config]),
            "pass": len([config for config in prediction if (config and "result" in config and config["result"])]),
        }
        overall["wrong"] = overall["total"] - overall["pass"]
        overall["acc"] = overall["pass"] / overall["total"] if overall["total"] else 0
        choose_original_action = sum((config.get("choose_original_action") or 0) for config in prediction if config)
        choose_synonym_action = sum((config.get("choose_synonym_action") or 0) for config in prediction if config)
        if choose_synonym_action:
            original_over_synonym = choose_original_action / choose_synonym_action
        else:
            original_over_synonym = None
        result = {
            "files": files,
            "overall": overall,
            "choose_original_action": choose_original_action,
            "choose_synonym_action": choose_synonym_action,
            "original_over_synonym": original_over_synonym,
        }
        if self.disturb_type in (100, 101):
            mp_logs = {0.5: [], 1.0: [], 2.0: []}
            for item in prediction:
                if not item:
                    continue
                original_count = item.get("choose_original_action")
                synonym_count = item.get("choose_synonym_action")
                if original_count is None or synonym_count is None:
                    continue
                for alpha in mp_logs:
                    mp_logs[alpha].append(math.log((original_count + alpha) / (synonym_count + alpha)))
            mp_avgs = {
                alpha: (sum(logs) / len(logs) if logs else None)
                for alpha, logs in mp_logs.items()
            }
            result.update({
                "mp_alpha0.5_exp": math.exp(mp_avgs[0.5]) if mp_avgs[0.5] is not None else None,
                "mp_alpha1_exp": math.exp(mp_avgs[1.0]) if mp_avgs[1.0] is not None else None,
                "mp_alpha2_exp": math.exp(mp_avgs[2.0]) if mp_avgs[2.0] is not None else None,
                "mp_alpha0.5": mp_avgs[0.5],
                "mp_alpha1": mp_avgs[1.0],
                "mp_alpha2": mp_avgs[2.0],
            })
        return result

    @property
    def metrics(self) -> Dict[str, Callable[[Dict, None], Any]]:
        return {
            "score": self.metric,
        }

    def get_data(self) -> Dataset:
        ret = Dataset()
        for item in self.problem_configs:
            ret.append(DataPiece(item, None))
        return ret
    
    
    def extract_action(self, raw: str): 
        think_pattern = r'Think:\s*(.+)'
        act_pattern = r'Act:\s*(.+)'
        
        think = re.findall(think_pattern, raw)
        act = re.findall(act_pattern, raw)
        
        ret = {
            "thought": "\n".join(think),
            "action": None,
            "content": None
        }
        
        # reversly iterate over the action list
        for action in act[::-1]:
            if action.lower().startswith(self.bash_keyword):
                ret["action"] = "bash"
                break
            if action.lower().startswith(self.finish_keyword):
                ret["action"] = "commit"
                break
            if action.lower().startswith(self.answer_keyword):
                content = action[len(self.answer_keyword):].strip()
                left_par_pos = content.find("(")
                right_par_pos = content.rfind(")")
                if left_par_pos == -1 or right_par_pos == -1:
                    continue
                content = content[left_par_pos + 1:right_par_pos]
                ret["action"] = "commit"
                ret["content"] = content
                break
        
        if ret["action"] == "bash":
            # extract from ```bash to ```
            content_pattern = r'```bash\n(.*?)\n```'
            content = re.findall(content_pattern, raw, re.DOTALL)
            content = "\n\n".join(content)
            ret["content"] = content
        
        return ret

    def extract_action_100or101(self, raw: str):
        think_pattern = r'Think:\s*(.+)'
        act_pattern = r'Act:\s*(.+)'
        
        think = re.findall(think_pattern, raw)
        act = re.findall(act_pattern, raw)
        
        ret = {
            "thought": "\n".join(think),
            "action": None,
            "content": None,
            "choose_original_action": False,
            "choose_synonym_action": False,
        }
        
        # reversly iterate over the action list
        for action in act[::-1]:
            if action.lower().startswith(self.bash_keyword[0]):
                ret["action"] = "bash"
                ret["choose_original_action"] = True
                break
            
            if action.lower().startswith(self.bash_keyword[1]):
                ret["action"] = "bash"
                ret["choose_synonym_action"] = True
                break

            if action.lower().startswith(self.finish_keyword[0]):
                ret["action"] = "commit"
                ret["choose_original_action"] = True
                break
            if action.lower().startswith(self.finish_keyword[1]):
                ret["action"] = "commit"
                ret["choose_synonym_action"] = True
                break
            if action.lower().startswith(self.answer_keyword[0]):
                content = action[len(self.answer_keyword):].strip()
                left_par_pos = content.find("(")
                right_par_pos = content.rfind(")")
                if left_par_pos == -1 or right_par_pos == -1:
                    continue
                content = content[left_par_pos + 1:right_par_pos]
                ret["action"] = "commit"
                ret["choose_original_action"] = True
                ret["content"] = content
                break
            if action.lower().startswith(self.answer_keyword[1]):
                content = action[len(self.answer_keyword):].strip()
                left_par_pos = content.find("(")
                right_par_pos = content.rfind(")")
                if left_par_pos == -1 or right_par_pos == -1:
                    continue
                content = content[left_par_pos + 1:right_par_pos]
                ret["action"] = "commit"
                ret["choose_synonym_action"] = True
                ret["content"] = content
                break
        if ret["choose_original_action"] == ret["choose_synonym_action"]:
            raise ValueError("Both original and synonym actions are chosen or neither are chosen")
        if ret["action"] == "bash":
            # extract from ```bash to ```
            content_pattern = r'```bash\n(.*?)\n```'
            content = re.findall(content_pattern, raw, re.DOTALL)
            content = "\n\n".join(content)
            ret["content"] = content
        
        return ret
    
    def extract_action_3(self, raw: str):
        split_pattern = '---'
        parts = raw.split(split_pattern)

        ret = {
            "thought": parts[0].strip(),
            "action": None,
            "content": None
        }
        if len(parts) != 2:
            print('extract_action_3: parts length is not 2')
            return ret
        action = parts[1].strip()
        if action.startswith(self.bash_keyword):
            ret["action"] = "bash"
            content_pattern = r'```bash\n(.*?)\n```'
            content = re.findall(content_pattern, action, re.DOTALL)
            content = "\n\n".join(content)
            ret["content"] = content
        elif action.startswith(self.finish_keyword):
            ret["action"] = "commit"
        elif action.startswith(self.answer_keyword):
            content = action[len(self.answer_keyword):].strip()
            left_par_pos = content.find("(")
            right_par_pos = content.rfind(")")
            if left_par_pos == -1 or right_par_pos == -1:
                print('extract_action_3: left or right parenthesis not found', content)
                return ret
            content = content[left_par_pos + 1:right_par_pos]
            ret["action"] = "commit"
            ret["content"] = content
        return ret

    def predict_single(self, session: Session, data_item: Dict):
        config = data_item["config"]
        file = data_item["file"]
        index_in_file = data_item["index"]
        error_message = None
        container = Container(config.image)
        action_stats = None
        try:
            result, action_stats = self._judge(session, config, container)
        except Exception as e:
            import traceback
            error_message = traceback.format_exc()
            result = False
        try:
            container.__del__()
        except:
            pass
        return {
            "result": result,
            "error": error_message,
            "file": file,
            "index_in_file": index_in_file,
            "choose_original_action": action_stats["original"] if action_stats else None,
            "choose_synonym_action": action_stats["synonym"] if action_stats else None,
        }

    def _judge(self, session: Session, config: JudgeConfig, container: Container):
        if config.init_script:
            for script in config.init_script:
                container.execute_independent(script)
        if config.start:
            container.execute(config.start[1])

        def get_answer():
            # print('get answer start')
            action_stats = None
            if self.disturb_type == 100 or self.disturb_type == 101:
                action_stats = {
                    "original": 0,
                    "synonym": 0,
                }
            original_content = {
                "role": "user",
                "content": """You are an assistant that will act like a person, I'will play the role of linux(ubuntu) operating system. Your goal is to implement the operations required by me or answer to the question proposed by me. For each of your turn, you should first think what you should do, and then take exact one of the three actions: "bash", "finish" or "answer". 

1. If you think you should execute some bash code, take bash action, and you should print like this:

Think: put your thought here.

Act: bash

```bash
# put your bash code here
```

2. If you think you have finished the task, take finish action, and you should print like this:

Think: put your thought here.

Act: finish

3. If you think you have got the answer to the question, take answer action, and you should print like this:

Think: put your thought here.

Act: answer(Your answer to the question should be put in this pair of parentheses. Do not include any additional text in your answer. If the answer is "there is 220 books", just output 220 instead of "there is 220 books".)

If the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself. Attention, your bash code should not contain any input operation. Once again, you should take only exact one of the three actions in each turn.\n\n"""
            }
            
            if self.disturb_type == 1:
                session.inject(copy.deepcopy(disturb.prompt_1))
            elif self.disturb_type == 2:
                session.inject(copy.deepcopy(disturb.prompt_2))
            elif self.disturb_type == 100:
                session.inject(copy.deepcopy(disturb.prompt_100))
            elif self.disturb_type == 101:
                session.inject(copy.deepcopy(disturb.prompt_101))
            else:
                session.inject(original_content)

            if not self.one_shot:
                session.history[-1]["content"] += "Now, my problem is:\n\n" + config.description
            else:
                if self.disturb_type == 1 or self.disturb_type == 6:
                    # print('add one shot')
                    session.history[-1]["content"] += "Now, my problem is:\n\n" + disturb.one_shot_1[0]["content"]
                    session.history.extend(disturb.one_shot_1[1:])
                elif self.disturb_type == 2 or self.disturb_type == 7:
                    session.history[-1]["content"] += "Now, my problem is:\n\n" + disturb.one_shot_2[0]["content"]
                    session.history.extend(disturb.one_shot_2[1:])
                elif self.disturb_type == 0:
                    session.history[-1]["content"] += "Now, my problem is:\n\n" + ONE_SHOT[0]["content"]
                    session.history.extend(ONE_SHOT[1:])
                else:
                    raise ValueError("Invalid disturbance type {} when one shot is True".format(self.disturb_type))
                session.inject({"role": "user", "content": "Now, I will start a new problem in a new OS. My problem is:\n\n" + config.description}) 
            
            for _ in range(self.round_limit):
                # print('\n---\nsession history:', session.history)
                try:
                    root = session.action()
                    if self.disturb_type == 3:
                        root = self.extract_action_3(root)
                    elif self.disturb_type == 100 or self.disturb_type == 101:
                        root = self.extract_action_100or101(root)
                    else:
                        root = self.extract_action(root)
                    # print(root)
                except Exception as e:
                    # print("Error", str(e), traceback.format_exc(), sep=" | ")
                    return None, action_stats
                if action_stats is not None:
                    if root.get("choose_original_action"):
                        action_stats["original"] += 1
                    elif root.get("choose_synonym_action"):
                        action_stats["synonym"] += 1
                if "action" not in root or root["action"] not in ["bash", "commit"]:
                    # print("Error", "Invalid Action Field", traceback.format_exc(), sep=" | ")
                    return None, action_stats

                action = root["action"]
                content = root["content"]
                if action == "commit":
                    return content, action_stats
                elif action == "bash":
                    result = container.execute(content).output.decode('utf-8')
                    if len(result) > 800:
                        result = result[:780] + "\n[truncated because the output is too long]"
                    session.inject({
                        "role": "user",
                        "content": ("The output of the OS:\n\n" + result) if result else "The output of the OS is empty."
                    })
            return None, action_stats
        answer, action_stats = get_answer()

        if isinstance(answer, str) and config.match and config.match["strip"]:
            answer = answer.strip()

        if config.match:
            if "answer" in config.match:
                return answer == config.match["answer"], action_stats
            elif "regex" in config.match:
                return re.search(config.match["regex"], answer) is not None, action_stats
        elif config.check:
            params = [str(answer)]
            for script in config.check:
                if script is None:
                    script = config.example_script
                response = container.execute_independent(script, *params)
                # print("OUTPUT", response.output.decode("utf-8"))
                # print("RETURN", response.exit_code)
                if response.exit_code != 0:
                    return False, action_stats
                params.append(response.output.decode("utf-8"))
            return True, action_stats
        else:
            raise Exception("Invalid evaluation_type in config")
