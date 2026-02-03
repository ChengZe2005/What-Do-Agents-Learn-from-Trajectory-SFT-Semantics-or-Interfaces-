from typing import Any, Mapping
import re

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput
from agentenv.disturb import DisturbManager


class SqlGymEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "value": (
                    "Given you a description of a SQlite database system, I will ask you a question, "
                    "then you should help me operate the SQLite database with SQL to answer the question.\n\n"
                    "You have to explain the problem and your solution to me and write down your thoughts.\n"
                    "After thinking and explaining thoroughly, you should give a SQL statement to solve the question.\n\n"
                    "your response should be like this:\n"
                    "Thought: Your thought here.\n\n"
                    "Action: ```sql\n"
                    "SELECT * FROM table WHERE condition;\n"
                    "```\n\n"
                    "You MUST put SQL in markdown format without any other comments. Your SQL should be in one line. "
                    "Every time you can only execute one SQL statement."
                ),
                "loss": None,
            }
        ),
        ConversationMessage({"from": "gpt", "value": "Ok.", "loss": False}),
    )

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
            env_key="sqlgym", disturb=getattr(self, "disturb", 0)
        )
        if self._disturb_mgr.enabled:
            self.conversation_start = self._disturb_mgr.apply_to_conversation_start(
                type(self).conversation_start
            )

        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(
            f"{self.env_server_base}/create",
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["env_idx"] = self.env_id
        max_retries = 5
        for _ in range(max_retries):
            res = requests.post(
                f"{self.env_server_base}/{path}",
                json=data,
                timeout=self.timeout,
            )
            if res.status_code == 503:
                import time

                time.sleep(0.1)
            elif res.status_code == 200:
                break
            else:
                print("---------------------")
                print(res.status_code)
                print(data)
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _extract_sql_from_code_fence(self, text: str, fence_open: str) -> str | None:
        """Extract SQL between `fence_open` and the next closing triple backticks.

        Returns None if the opening fence is not found or closing fence is missing.
        """
        if fence_open not in text:
            return None
        after = text.split(fence_open, 1)[1]
        if "```" not in after:
            return None
        sql = after.split("```", 1)[0].strip()
                                                    
        sql = re.sub(r"\s+", " ", sql).strip()
        return sql

    def step(self, action: str) -> StepOutput:
        raw = action
        if isinstance(raw, str) and raw.endswith("</s>"):
            raw = raw[:-5]

        orig_fence = "```sql"

        if getattr(self, "_disturb_mgr", None) is not None and self._disturb_mgr.enabled:
            expected_fence = self._disturb_mgr.orig2alias.get(orig_fence, orig_fence)

                                                                                            
            sql = self._extract_sql_from_code_fence(raw, expected_fence)
            if sql is not None:
                response = self._post("step", {"action": sql})
                state = response.get("state")
                if self._disturb_mgr.enabled:
                    state = self._disturb_mgr.apply_to_any(state)
                return StepOutput(
                    state=state,
                    reward=response["reward"],
                    done=response["done"],
                )

                                                                                   
            if orig_fence in raw:
                                                                         
                return StepOutput(
                    state=self._disturb_mgr._deprecated_error(orig_fence),
                    reward=0.0,
                    done=False,
                )

                                                                                      
            return StepOutput(
                state="Invalid Action.\n\n" + str(self.observe()),
                reward=0.0,
                done=False,
            )

                                                 
        sql = self._extract_sql_from_code_fence(raw, orig_fence)
        if sql is None:
                                                                
            sql = raw.split(orig_fence)[-1].split("```")[0].strip()
            sql = re.sub(r"\s+", " ", sql).strip()

        response = self._post("step", {"action": sql})
        return StepOutput(
            state=response["state"],
            reward=response["reward"],
            done=response["done"],
        )

    def observe(self) -> dict[str, Any]:
        response = self._get("observation")
        if getattr(self, "_disturb_mgr", None) is not None and self._disturb_mgr.enabled:
            response = self._disturb_mgr.apply_to_any(response)
        return response

    def reset(self, idx: int) -> dict[str, Any]:
        response = self._post("reset", {"item_id": idx})
        return response


class SqlGymTask(BaseTask):
    env_client_cls = SqlGymEnvClient
    env_name = "SQLGym"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
