import json
from typing import Any, Mapping

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput
from agentenv.disturb import DisturbManager



                                          
      
                                          


class MazeEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {"from": "human", "loss": None, "value": "You are an expert maze solver."}
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )
    _fully_first_observation = """\
Your objective is to reach the goal in as few steps as possible. At each step you will be given information about where the goal is, your current position,
and the walls that surround you. 

When you move right you increase your y position by 1, when you move down you increase your x position by 1. 
When you move left you decrease your y position by 1, when you move up you decrease your x position by 1. 

Here is an example.

```
environment: The goal is at position 8, 6. Your current position is at position 5, 6. There is a wall above you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 5. There are walls above you, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move up
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 5, 5. There are walls above you, below you.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 5, 6. There is a wall above you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 6, 6. There are walls to your right, to your left.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 7, 6. There are walls to your right, to your left.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 7, 6. There are walls to your right, to your left.
action: move down
environment: Success
```

Your possible actions are "move up", "move down", "move left", "move right". Formally, your return should be in this format:
Thought:\n<Your Thought>\n\nAction:\n<Your Action>

Now let's start a new game. Return your action and your thought in the format above strictly. Now, make the optimal action given the current environment state:
""".strip()
    _partially_first_observation = """\
Your objective is to reach the goal in as few steps as possible. At each step you will see your move history, and the walls that surround you.

Here is an example. 
```
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move left
environment: There is a wall above you.
action: move left
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move right
environment: There is a wall above you.
action: move down
environment: There are walls to your right, to your left.
action: move down
environment: There are walls to your right, to your left.
action: move right
environment: There are walls to your right, to your left.
action: move down
environment: Success
```

Your possible actions are "move up", "move down", "move left", "move right". Formally, your return should be in this format:
Thought:\n<Your Thought>\n\nAction:\n<Your Action>

Now let's start a new game. Return your action and your thought in the format above strictly. Now, make the optimal action given the current environment state:
""".strip()

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._disturb_mgr = DisturbManager(env_key="maze", disturb=getattr(self, "disturb", 0))
        if self._disturb_mgr.enabled:
            self.conversation_start = self._disturb_mgr.apply_to_conversation_start(
                type(self).conversation_start
            )
                                                                                
            self._dist_fully_first_observation = self._disturb_mgr.apply_to_any(
                type(self)._fully_first_observation
            )
            self._dist_partially_first_observation = self._disturb_mgr.apply_to_any(
                type(self)._partially_first_observation
            )
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        print(ok)
        self.env_id = ok["id"]
        self.info = {
            "reward": 0,
            "done": False,
        }
        
    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        obs = self.info.get("observation", "")
        if self._disturb_mgr.enabled:
            obs = self._disturb_mgr.apply_to_any(obs)
        return str(obs)


    def step(self, action: str) -> StepOutput:
        raw = action
        print(raw)
        if isinstance(raw, str) and raw.endswith("</s>"):
            raw = raw[:-5]

        parts = raw.split("Action:")
        if len(parts) > 1:
            action_cmd = parts[-1].strip()
        else:
            action_cmd = parts[0].strip()

                                                                                      
        action_cmd = action_cmd.strip()
        action_cmd = action_cmd.strip("`").strip()
        action_cmd = action_cmd.strip('"').strip("'").strip()

        print(f"Action: {action_cmd}")

        if self._disturb_mgr.enabled:
            translated, err = self._disturb_mgr.translate_prefix_command(action_cmd)
            if err is not None:
                return StepOutput(
                    state=str(err) + "\n\n" + str(self.observe()),
                    reward=0.0,
                    done=False,
                )
            action_cmd = translated

                                                                                      
                                                                                  
        action_cmd = action_cmd.strip().strip('"').strip("'").strip("`").strip()
        action_cmd = action_cmd.rstrip(".,;:!?")
        action_cmd = " ".join(action_cmd.split())

        response = self._post("step", {"action": action_cmd})
        print(response)
        if self._disturb_mgr.enabled:
            response = self._disturb_mgr.apply_to_any(response)
        self.info.update(
            {
                "observation": response["observation"],
                "reward": self.info["reward"] + response["reward"],
                "done": response["done"],
            }
        )
        return StepOutput(
            state=str(response["observation"]),
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"game": idx})
        print(response)
        if self._disturb_mgr.enabled:
            self.first_observation = getattr(self, "_dist_fully_first_observation", type(self)._fully_first_observation)
        else:
            self.first_observation = type(self)._fully_first_observation
        response["observation"] = (
            self.first_observation + "\n" + response["observation"]
        )
        if self._disturb_mgr.enabled:
            response = self._disturb_mgr.apply_to_any(response)
        self.info.update(
            {
                "observation": response["observation"],
                "reward": 0,
                "done": False,
            }
        )
        return response


class MazeTask(BaseTask):
    env_client_cls = MazeEnvClient
    env_name = "LMRL-Gym.maze"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)


                                          
        
                                          


class WordleEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {"from": "human", "loss": None, "value": "You are an expert wordle player."}
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )
    first_observation = """\
Welcome to the game of Wordle. Your objective is to guess a hidden 5 letter word. You have 6 attempts to guess it correctly and you should try to guess it in as few attempts as possible. When guessing the word, you should format your word as a space separated sequence of letters, like "s h i r e" for example. After guessing the word, you will receive feedback from the game environment in the form of a sequence of 5 space separated letters like "b y g g b", where each letter indicates some information about the hidden word. The environment will return one of three letters – "b", "g", or "y" – for each letter in the word you guessed. We describe the meaning of each letter below:

"b": If the environment returns a "b", it means that the letter at that position in your guessed word is not in the hidden word.
"y": If the environment returns a "y", it means that the letter at that position in your guessed word is in the hidden word but is not in the correct position.
"g": If the environment returns a "g", it means that the letter at that position in your guessed word is in the hidden word and is in the correct position.

As a note, if you guess an invalid word (e.g. not a 5 letter word or a word not in the vocabulary), the environment will respond with an "invalid word" message. In general though, you should use this information returned by the environment to update your belief about what the hidden word might be and adjust your next guess accordingly.

Here is the complete list of valid vocabulary words that are accepted by the game:
```
{{vocab}}
```

Here is an example. If the current status of the game is given as:
```
guess 1: p a n i c
feedback 1: b b y b b
guess 2: f e l o n
feedback 2: g b b y g
```
Based on the feedback from the environment, you know that the first letter is "f", the last letter is "n", and there is an "o" somewhere in the word, but it is not in the second to last position. You also know that there is not a "p", "a", "i", "c", "e", or "l" in the word. Knowing this, you might guess the next word to be:
Thought:\nI know that the first letter is "f", the last letter is "n", and there is an "o" somewhere in the word, but it is not in the second to last position. I also know that there is not a "p", "a", "i", "c", "e", or "l" in the word. A good word from the vocabulary to try might therefore be \"f r o w n\", since it is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of "o". Therefore this is a good guess to try next.\n\nAction:\nf r o w n

Formally, your return should be in this format:
Thought:\n<Your Thought>\n\nAction:\n<The Word You Guess>

The guessed word is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of "o", so it is a good guess to try next.

Now let's start a new game. Remember, the word you guess should be strictly in the vocabulary. You should return your thought and your word strictly in the formation mentioned above.
""".strip()

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

                                                                               
        self._disturb_mgr = DisturbManager(
            env_key="wordle", disturb=getattr(self, "disturb", 0)
        )
        if self._disturb_mgr.enabled:
            self.conversation_start = self._disturb_mgr.apply_to_conversation_start(
                type(self).conversation_start
            )
            self.first_observation = self._disturb_mgr.apply_to_any(
                type(self).first_observation
            )
        else:
                                                       
            self.first_observation = type(self).first_observation

        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        print(ok)
        self.env_id = ok["id"]
        vocab = self._get("filtered_vocab")
        self.info = {
            "observation": self.first_observation.replace(
                "{{vocab}}", "\n".join(vocab)
            ),
            "vocab": vocab,
            "reward": 0,
            "done": False,
        }
        print(self.info["observation"])

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        obs = self.info.get("observation", "")
        if getattr(self, "_disturb_mgr", None) is not None and self._disturb_mgr.enabled:
            obs = self._disturb_mgr.apply_to_any(obs)
        return str(obs)


    def step(self, action: str) -> StepOutput:
        raw = action
        print(raw)
        if isinstance(raw, str) and raw.endswith("</s>"):
            raw = raw[:-5]

        def _extract_after_label(text: str, label: str) -> str | None:
            """Find a header label and extract the following guess.

            Supports:
              - '<label>\n<guess>'
              - '<label> <guess>' or '<label><guess>'
              - 'Action: <label> <guess>' (model keeps old header then new one)
            """
            lines = text.splitlines()
            for i, line in enumerate(lines):
                s = line.strip()
                if s == label:
                    for j in range(i + 1, len(lines)):
                        nxt = lines[j].strip()
                        if nxt != "":
                            return nxt
                    return ""
                if s.startswith(label):
                    rest = s[len(label) :].strip()
                    if rest != "":
                        return rest
                    for j in range(i + 1, len(lines)):
                        nxt = lines[j].strip()
                        if nxt != "":
                            return nxt
                    return ""
                                                                              
            idx = text.find(label)
            if idx != -1:
                rest = text[idx + len(label) :]
                for line in rest.splitlines():
                    s = line.strip()
                    if s != "":
                        return s
                return ""
            return None

                                       
                                                                                          
                                                                                                       
        if getattr(self, "_disturb_mgr", None) is not None and self._disturb_mgr.enabled:
            mode = getattr(self._disturb_mgr, "disturb", 0)
            alias_label = self._disturb_mgr.orig2alias.get("Action:", "Action:")

            if mode in (3, 4):
                guess = _extract_after_label(raw, alias_label)
                if guess is not None:
                                                              
                    self._disturb_mgr.disturb1_calls += 1
                    action_to_send = guess.strip()
                else:
                    guess0 = _extract_after_label(raw, "Action:")
                    if guess0 is not None:
                                                                   
                        self._disturb_mgr.disturb0_calls += 1
                        action_to_send = guess0.strip()
                    else:
                                                                                         
                        parts = raw.split("Action:")
                        action_to_send = parts[-1].strip() if len(parts) > 1 else parts[0].strip()
            else:
                                                                               
                expected_label = alias_label
                guess = _extract_after_label(raw, expected_label)
                if guess is None:
                                                                                            
                    if _extract_after_label(raw, "Action:") is not None:
                        return StepOutput(
                            state=self._disturb_mgr._deprecated_error("Action:")
                            + "\n\n"
                            + str(self.observe()),
                            reward=0.0,
                            done=False,
                        )
                                                                                                       
                    return StepOutput(
                        state="Invalid Action.\n\n" + str(self.observe()),
                        reward=0.0,
                        done=False,
                    )
                action_to_send = guess.strip()
        else:
                                
            parts = raw.split("Action:")
            if len(parts) > 1:
                action_to_send = parts[-1].strip()
            else:
                action_to_send = parts[0].strip()

                                                
        action_to_send = action_to_send.strip()
        action_to_send = action_to_send.strip("`").strip()
        action_to_send = action_to_send.strip('"').strip("'").strip()
        action_to_send = action_to_send.rstrip(".,;:!?")

        print(f"Action: {action_to_send}")

        response = self._post("step", {"action": action_to_send})
        print(response)
        obs = response.get("observation")
        if getattr(self, "_disturb_mgr", None) is not None and self._disturb_mgr.enabled:
            obs = self._disturb_mgr.apply_to_any(obs)
        self.info.update(
            {
                "observation": obs,
                "reward": self.info["reward"] + response["reward"],
                "done": response["done"],
            }
        )
        return StepOutput(
            state=str(obs),
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"seed": idx})
        self.info.update(
            {
                "observation": self.first_observation.replace(
                    "{{vocab}}", "\n".join(self.info["vocab"])
                ),
                "reward": 0,
                "done": False,
            }
        )
        return response


class WordleTask(BaseTask):
    env_client_cls = WordleEnvClient
    env_name = "LMRL-Gym.wordle"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
