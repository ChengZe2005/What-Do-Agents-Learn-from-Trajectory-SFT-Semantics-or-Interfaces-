import os
import json
import sys
import time
import re
import math
import random
import datetime
import argparse
from turtle import Turtle
import requests
import glob
import importlib
from typing import Callable, List, Dict, Any, Tuple, Optional, Union
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.task import Task, DataPiece, Dataset
from src.agent import Agent, Session
from src.tasks.alfworld.environment import SingleAlfredTWEnv, get_all_game_files
from src.tasks.alfworld.utils import *


class ALFWorld(Task):

    def __init__(self, **kwargs):
        mp.set_start_method('spawn')

        # load data_path 
        self.data_path = kwargs.get("data_path", None)
        if self.data_path is None:
            raise Exception("missing parameter data_path")
        os.environ["ALFWORLD_DATA"] = self.data_path

        # load config for alfworld benchmark
        self.config_path = kwargs.get("config_path", None)
        if self.config_path is None:
            raise Exception("missing parameter config_path")
        self.config = load_config(self.config_path)

        # load prompts
        self.prompts_path = kwargs.get("prompts_path", None)
        if self.prompts_path is None:
            raise Exception("missing parameter prompts_path")
        self.prompts = load_prompts(self.prompts_path)

        # one_shot
        self.one_shot = kwargs.get("one_shot", False)
        # disturb_type
        self.disturb_type = kwargs.get("disturb_type", 0)
        self.action_keyword = "ACTION"
        if self.disturb_type == 1:
            self.action_keyword = "MOVE"
        elif self.disturb_type == 2:
            self.action_keyword = "z2"
        elif self.disturb_type == 100 or self.disturb_type == 101:
            self.action_keyword = ["ACTION", "MOVE"]

        # prepare data_files
        self.data_files = []
        self.split = kwargs.get("split", "dev")
        data_path = os.path.join("data/alfworld", f"{self.split}.json")
        with open(data_path, "r") as f:
            content = json.loads(f.read())
        for _, v in content.items():
            self.data_files.extend(v)
        self.data_files = [os.path.join(self.data_path, file) for file in self.data_files]
        print(f"> successfully loaded {len(self.data_files)} games")

        # other configs
        self.max_step = kwargs.get("max_step", 50)
        self.prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }

        super().__init__(**kwargs)

    def final_score(self, results: List[Dict], targets: List[None]):
        try:
            completions = []
            original_action_count = 0
            synonym_action_count = 0
            mp_logs = {0.5: [], 1: [], 2: []}
            for res in results:
                if res.get("msg", None) is not None:
                    continue
                completions.append(int(res.get("result", 0)))
                if self.disturb_type in [100, 101]:
                    original_count = int(res.get("original_action_count", 0) or 0)
                    synonym_count = int(res.get("synonym_action_count", 0) or 0)
                    original_action_count += original_count
                    synonym_action_count += synonym_count
                    for alpha in mp_logs:
                        ratio = (original_count + alpha) / (synonym_count + alpha)
                        mp_logs[alpha].append(math.log(ratio))
            success = np.sum(completions)
            result = {
                "completion": int(success),
                "total": len(completions),
                "rate": int(success) / len(completions),
                "main": int(success) / len(completions),
            }
            if self.disturb_type in [100, 101]:
                mp_avgs = {
                    alpha: (sum(logs) / len(logs) if logs else 0.0)
                    for alpha, logs in mp_logs.items()
                }
                result.update({
                    "original_action_count": int(original_action_count),
                    "synonym_action_count": int(synonym_action_count),
                    "original_synonym_ratio": (
                        original_action_count / synonym_action_count
                        if synonym_action_count > 0 else None
                    ),
                    "mp_alpha0.5": mp_avgs[0.5],
                    "mp_alpha1": mp_avgs[1],
                    "mp_alpha2": mp_avgs[2],
                    "mp_alpha0.5_exp": math.exp(mp_avgs[0.5]),
                    "mp_alpha1_exp": math.exp(mp_avgs[1]),
                    "mp_alpha2_exp": math.exp(mp_avgs[2]),
                })
            return result
        except Exception as e:
            print(f"eval exception {e}")
            return {"error": 1}

    @property
    def metrics(self) -> Dict[str, Callable[[Dict, None], Any]]:
        return { "score": self.final_score }

    def get_data(self) -> Dataset:
        ret = Dataset()
        for item in self.data_files:
            ret.append(DataPiece(item, None))
        return ret

    def predict_all(self, agent: Agent, inputs: List) -> List:
        print(f"Start Predicting All ...")

        count = self.workers
        if self.worker_limit:
            count = min(self.workers, self.worker_limit)

        executor = ProcessPoolExecutor(max_workers=count)
        results = []
        processes = []
        
        parameters = [(input_, agent, index) for index, input_ in enumerate(inputs)]
        for idx, parameter in enumerate(parameters):
            future = executor.submit(self.call_wrap, parameter)
            processes.append(future)

        with tqdm(total=len(parameters)) as pbar:
            for process in as_completed(processes):
                results.append(process.result())
                pbar.update(1)
        
        return results

    def call_wrap(self, parameters):
        data_item, agent, index = parameters
        session = agent.create_session()
        result = self.predict_single(session, data_item, agent.name)
        self.save_single(index, data_item, result, session)
        return result

    def predict_single(self, session: Session, data_item: str, agent_name: str):
        env = SingleAlfredTWEnv(self.config, data_item)
        env = env.init_env(batch_size=1)
        try:
            result, log_info = self.alfworld_run(session, env)
        except Exception as e:
            result = 0
            log_info = {"msg": f"error {e}"}
        log_info.update({"result": result})
        log_info.update({"agent": agent_name})

        try:
            del env
        except:
            pass
        
        print("finished one game")
        return log_info

    def get_task_instruction(self):
        # return "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. In the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you should choose from two actions: \"THOUGHT\" or \"ACTION\". If you choose \"THOUGHT\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THOUGHT: your thoughts.\n ACTION: your next action\n\"; If you choose \"ACTION\", you should directly output the action in this turn. Your output must strictly follow this format:\"ACTION: your next action\n\". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\n"

        if self.disturb_type == 1:
            return "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: \"THINK\" or \"MOVE\". If you choose \"THINK\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THINK: your thoughts.\n MOVE: your next action\n\"; If you choose \"MOVE\", you should directly output the action in this turn. Your output must strictly follow this format:\"MOVE: your next action\n\". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.\n\n"
        elif self.disturb_type == 2:
            return "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: \"z1\" or \"z2\". If you choose \"z1\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"z1: your thoughts.\n z2: your next action\n\"; If you choose \"z2\", you should directly output the action in this turn. Your output must strictly follow this format:\"z2: your next action\n\". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.\n\n"
        elif self.disturb_type == 100:
            return "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: \"THOUGHT\" or \"ACTION\". Note that \"MOVE\" is an alias for \"ACTION,\" and you can use it as well. If you choose \"THOUGHT\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THOUGHT: your thoughts.\n ACTION: your next action\n\"\nOr you can output: \"THOUGHT: your thoughts.\n MOVE: your next action\n\"\n If you choose \"ACTION\"(or\"MOVE\"), you should directly output the action in this turn. Your output must strictly follow this format:\"ACTION: your next action\n\".\nOr you can output: \"MOVE: your next action\n\"\n After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.\n\n"
        elif self.disturb_type == 101:
            return "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: \"THINK\" or \"MOVE\". Note that \"ACTION\" is an alias for \"MOVE,\" and you can use it as well. If you choose \"THINK\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THINK: your thoughts.\n MOVE: your next action\n\"\nOr you can output:\"THOUGHT: your thoughts.\n ACTION: your next action\n\"\n  If you choose \"MOVE\", you should directly output the action in this turn. Your output must strictly follow this format:\"MOVE: your next action\n\".\nOr you can output:\"ACTION: your next action\n\"\n After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.\n\n"
        return "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: \"THOUGHT\" or \"ACTION\". If you choose \"THOUGHT\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THOUGHT: your thoughts.\n ACTION: your next action\n\"; If you choose \"ACTION\", you should directly output the action in this turn. Your output must strictly follow this format:\"ACTION: your next action\n\". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.\n\n"

    def get_prompt(self, filename: str):
        # return []
        for k,v in self.prefixes.items():
            if filename.startswith(k):
                example = self.prompts[v]
                print('example: \n', example)
                return example
        raise Exception(f"unsupported name: {filename}")
        # return self.prompts["naive_example"]

    def inject_info(self, session: Session, history: List):
        current_role = "user"
        traverse = {"user": "agent", "agent": "user"}
        for his in history:
            session.inject({"role":current_role, "content": his})
            current_role = traverse[current_role]

    def get_available_actions(self, actions):
        actions = "\n".join(actions)
        return " AVAILABLE ACTIONS: " + actions + "\n"

    def alfworld_run(self, session: Session, env: Any):
        # env init
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        # history = self.get_prompt(name)
        # add instruction
        # history[0] = self.get_task_instruction() + "Here is one example.\n" + history[0]
        # self.inject_info(session, history)
        log_info = {}
        log_info["log"] = []
        log_info["original_action_count"] = 0
        log_info["synonym_action_count"] = 0
        session.inject({"role": "user", "content": self.get_task_instruction()})
        session.inject({"role": "agent", "content": "OK. I'll follow your instructions and try my best to solve the task."})

        # 1-shot naive example（可配置关闭）
        if self.one_shot:
            history = self.get_prompt(name)
            history[0] = "Here is one example.\n" + history[0]
            self.inject_info(session, history)

        init_prompt = "Here is your task. " + ob + self.get_available_actions(info.get('admissible_commands', [[]])[0])
        log_info["init_prompt"] = init_prompt
        session.inject({"role": "user", "content": init_prompt})
        # init 
        # for his in session.history:
        #     print(his)
        # print("============ history end ==============")

        # interact
        for i in range(0, self.max_step):
            output = session.action()

            # process action
            admissible_commands = info.get('admissible_commands', [[]])[0]
            action, choose_original_action, choose_synonym_action = process_action(output, admissible_commands, action_keyword=self.action_keyword)
            if choose_original_action:
                log_info["original_action_count"] += 1
            if choose_synonym_action:
                log_info["synonym_action_count"] += 1
            session.history[-2]["content"] = session.history[-2]["content"].split("AVAILABLE ACTIONS")[0] # reduce the prompt length 

            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            session.inject({"role":"user", "content":observation + self.get_available_actions(info.get('admissible_commands', [[]])[0])})

            # save
            payload = {
                "round": i+1,
                "output": output,
                "action": action,
                "admissible_commands": admissible_commands,
                "observation": observation,
                "done": done,
                "choose_original_action": choose_original_action,
                "choose_synonym_action": choose_synonym_action,
            }
            log_info["log"].append(payload)

            # print("====== action ======")
            # print("output: ", output)
            # print("action: ", action)
            # print("obs and commands: ", session.history[-1]["content"])
            # print("====== end action =======")
            # print("====== ob =======")
            # print("ob: ", observation)
            # print("====== end ob =======\n")
            # print("action: ", action, "ob: ", observation)

            # failure test
            if len(log_info["log"]) > 3:
                pre_logs = log_info["log"][-3:]
                pre_acts = [pre_log["output"] for pre_log in pre_logs]
                if len(list(set(pre_acts))) == 1:
                    print("repeat actions for 3 times: failure")
                    return 0, log_info
            
            if done:
                return reward, log_info
        return 0, log_info