import math
import re
import sys
from os.path import dirname, realpath

sys.path.append(dirname(realpath(__file__)))

import multiprocess as mp
from typing import Dict, Callable, List
import src.tasks.webshop.disturb as disturb
from src.task import Task, Dataset, DataPiece, Session
from .web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

prompt: str = """
You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, \
you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought:
I think ...

Action:
click[something]
"""


def predict(receiver,disturb_type=0):
    env = WebAgentTextEnv(observation_mode="text", human_goals=True)
    while True:
        data_item, session, sender = receiver.recv()
        history = []
        env.reset(data_item)
        if disturb_type == 1:
            session.inject({"role": "user", "content": disturb.prompt_1})
            search_keyword = 'query'
            click_keyword = 'select'
        elif disturb_type == 2:
            session.inject({"role": "user", "content": disturb.prompt_2})
            search_keyword = 'z1'
            click_keyword = 'z2'
        elif disturb_type == 100:
            session.inject({"role": "user", "content": disturb.prompt_100})
            search_keyword = ['search', 'SEARCH']
            click_keyword = ['click', 'CLICK']
        elif disturb_type == 101:
            session.inject({"role": "user", "content": disturb.prompt_101})
            search_keyword = ['search', 'SEARCH']
            click_keyword = ['click', 'CLICK']
        else:
            session.inject({"role": "user", "content": prompt})
            search_keyword = 'search'
            click_keyword = 'click'
        session.inject({"role": "agent", "content": "Ok."})


        observation = env.observation
        reward = 0
        format_fail = False
        original_action_count = 0
        synonym_action_count = 0
        for j in range(10):
            available_actions = env.get_available_actions()
            session.inject({"role": "user", "content": f"Observation:\n{observation}\n\n"
                                                       f"Available Actions:\n{available_actions}"})
            response = session.action()
            try:
                if isinstance(search_keyword, list):
                    # original keyword is index 0, synonym keyword is index 1
                    action = re.search(
                        rf"[Aa]ction: *\n *(({search_keyword[0]}|{search_keyword[1]}|{click_keyword[0]}|{click_keyword[1]})\[.+?])",
                        response,
                    ).group(1)
                    if action.startswith(search_keyword[0]) or action.startswith(click_keyword[0]):
                        original_action_count += 1
                    elif action.startswith(search_keyword[1]) or action.startswith(click_keyword[1]):
                        synonym_action_count += 1
                    action = (action.replace(search_keyword[0], 'search')
                                   .replace(search_keyword[1], 'search')
                                   .replace(click_keyword[0], 'click')
                                   .replace(click_keyword[1], 'click'))
                else:
                    action = re.search(rf"[Aa]ction: *\n *(({search_keyword}|{click_keyword})\[.+?])", response).group(1)
                    action = action.replace(search_keyword, 'search').replace(click_keyword, 'click')
            except:
                format_fail = True
                action = None
            history.append({"observation": observation, "available_actions": available_actions,
                            "response": response, "action": action})
            if not action:
                reward = 0
                break
            #print(action)
            observation, reward, done, info = env.step(action)
            #print(reward)
            history[-1]["reward"] = reward
            history[-1]["done"] = done
            if done:
                break
        print(f'format_fail: {format_fail}')
        print(f'reward: {reward}')
        print(f'original_action_count: {original_action_count}, synonym_action_count: {synonym_action_count}')
        print(f'original_over_synonym: {(original_action_count / synonym_action_count) if synonym_action_count > 0 else None}')
        sender.send({
            "history": history,
            "reward": reward,
            "format_fail": format_fail
            ,
            "original_action_count": original_action_count,
            "synonym_action_count": synonym_action_count,
            "original_over_synonym": (original_action_count / synonym_action_count) if synonym_action_count > 0 else None,
        })


class WebShop(Task[int, Dict, None]):
    def __init__(self, **configs):
        super().__init__(**configs)
        self.ranging = (configs.pop("start", 0), configs.pop("end", 500))
        self.num_envs = min(self.workers, configs.pop("num_envs", 1))
        self.disturb_type = configs.pop("disturb_type", 0)
        self.processes = []
        ctx = mp.get_context('spawn')
        for i in range(self.num_envs):
            receiver, sender = ctx.Pipe(False)
            p = ctx.Process(target=predict, args=(receiver,self.disturb_type))
            p.start()
            self.processes.append((sender, ctx.Lock(), p))

    def get_data(self) -> Dataset[int, None]:
        dataset = Dataset()
        for i in range(*self.ranging):
            dataset.append(DataPiece(i, None))
        return dataset

    def predict_single(self, session: Session, data_item: Dict) -> Dict[str, None]:
        ctx = mp.get_context('spawn')
        receiver, sender = ctx.Pipe(False)
        i = 0
        while True:
            if self.processes[i][1].acquire(timeout=0.2):
                break
            i += 1
            i %= self.num_envs
        self.processes[i][0].send((data_item, session, sender))
        ret = receiver.recv()
        self.processes[i][1].release()
        return ret

    @property
    def metrics(self) -> Dict[str, Callable[[List[Dict], List[None]], float]]:
        def factory(key):
            def f(output, target):
                output = [x for x in output if x]
                if key == "history":
                    return sum([len(x[key]) for x in output]) / len(output) if len(output) > 0 else 0
                if key == "original_over_synonym":
                    total_original = sum([x["original_action_count"] for x in output])
                    total_synonym = sum([x["synonym_action_count"] for x in output])
                    return (total_original / total_synonym) if total_synonym > 0 else None
                if key in ("original_action_count", "synonym_action_count"):
                    return sum([x[key] for x in output])
                return sum([x[key] for x in output]) / len(output) if len(output) > 0 else 0

            return f

        def mp_factory(alpha: float, apply_exp: bool):
            def f(output, target):
                output = [x for x in output if x]
                if len(output) == 0:
                    return None
                values = []
                for x in output:
                    original = x["original_action_count"]
                    synonym = x["synonym_action_count"]
                    values.append(math.log((original + alpha) / (synonym + alpha)))
                mean_log = sum(values) / len(values)
                return math.exp(mean_log) if apply_exp else mean_log

            return f

        return {
            "reward": factory("reward"),
            "format_fail_rate": factory("format_fail"),
            "average_round": factory("history"),
            "original_action_count": factory("original_action_count"),
            "synonym_action_count": factory("synonym_action_count"),
            "original_over_synonym": factory("original_over_synonym"),
            "mp_alpha0.5_exp": mp_factory(0.5, True),
            "mp_alpha1_exp": mp_factory(1.0, True),
            "mp_alpha2_exp": mp_factory(2.0, True),
            "mp_alpha0.5": mp_factory(0.5, False),
            "mp_alpha1": mp_factory(1.0, False),
            "mp_alpha2": mp_factory(2.0, False),
        }

    def release(self):
        for _, _, p in self.processes:
            p.terminate()
            p.join()
