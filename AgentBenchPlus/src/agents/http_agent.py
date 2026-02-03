from src.agent import Agent
import time
import requests
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Type, TypeVar


class RoleContentPrompter:
    def __init__(self, message_key, role_key, content_key, role_dict):
        self.message_key = message_key
        self.role_key = role_key
        self.content_key = content_key
        # allow overriding default role mapping if provided
        self.role_dict = role_dict or {
            "user": "user",
            "agent": "agent",
        }

    def __call__(self, messages: List[Dict[str, str]]):
        prompt = []
        for item in messages:
            prompt.append({
                self.role_key: self.role_dict[item['role']],
                self.content_key: item['content']
            })
        return {self.message_key: prompt}


class PromptStringPrompter:
    def __init__(self, prefix, suffix, user_format, agent_format, prompt_key):
        self.prefix = prefix
        self.suffix = suffix
        self.user_format = user_format
        self.agent_format = agent_format
        self.prompt_key = prompt_key

    def __call__(self, messages: List[Dict[str, str]]):
        prompt = self.prefix
        for item in messages:
            if item["role"] == "user":
                prompt += self.user_format.format(content=item["content"])
            else:
                prompt += self.agent_format.format(content=item["content"])
        prompt += self.suffix
        return {self.prompt_key: prompt}


class Prompter:
    @staticmethod
    def get_prompter(prompter: Union[Dict[str, Any], None]):
        # check if prompter_name is a method and its variable
        if not prompter:
            return Prompter.default()
        assert isinstance(prompter, dict)
        prompter_name = prompter.get("name", None)
        prompter_args = prompter.get("args", {})
        if hasattr(Prompter, prompter_name) and callable(getattr(Prompter, prompter_name)):
            return getattr(Prompter, prompter_name)(**prompter_args)
        return Prompter.default()
    
    @staticmethod
    def default(
            message_key: str="messages", 
            role_key: str= "role", 
            content_key: str="content", 
            role_dict: Dict[str, str]=None
    ):
        return Prompter.role_content_dict(
            message_key=message_key, 
            role_key=role_key, 
            content_key=content_key, 
            role_dict=role_dict
        )
    
    @staticmethod
    def role_content_dict(
            message_key: str="messages", 
            role_key: str= "role", 
            content_key: str="content", 
            user_role: str="user",
            agent_role: str="agent",
            role_dict: Dict[str, str]=None,
    ):
        # prefer explicit role_dict if provided, otherwise use user_role/agent_role
        mapping = role_dict or {
            "user": user_role,
            "agent": agent_role,
        }
        return RoleContentPrompter(message_key, role_key, content_key, mapping)
    
    @staticmethod
    def prompt_string(
            prefix: str="",
            suffix: str="AGENT:",
            user_format: str="USER: {content}\n\n",
            agent_format: str="AGENT: {content}\n\n",
            prompt_key: str="prompt"
    ):
        return PromptStringPrompter(prefix, suffix, user_format, agent_format, prompt_key)
    
    @staticmethod
    def claude():
        return Prompter.prompt_string(
            prefix="", 
            suffix="Assistant:", 
            user_format="Human: {content}\n\n", 
            agent_format="Assistant: {content}\n\n"
        )

class HTTPAgent(Agent):
    def __init__(
            self, 
            url, 
            body=None, 
            headers=None, 
            return_format="{response}", 
            prompter=None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}
        self.body = body or {}
        self.return_format = return_format
        self.prompter = Prompter.get_prompter(prompter)
        if not self.url:
            raise Exception("Please set 'url' parameter")

    def _handle_history(self, history: List[dict]) -> List[dict]:
        return self.prompter(history)
        
    def inference(self, history: List[dict]) -> str:
        for _ in range(3):
            try:
                body = self.body.copy()
                body.update(self._handle_history(history))
                resp = requests.post(self.url, json=body, headers=self.headers, timeout=120)
                if resp.status_code != 200:
                    raise Exception(f"Invalid status code {resp.status_code}:\n\n{resp.text}")
            except Exception as e:
                print("Warning: ", e)
                pass
            else:
                resp = resp.json()
                return self.return_format.format(response=resp)
            time.sleep(_+2)
        raise Exception("Failed.")