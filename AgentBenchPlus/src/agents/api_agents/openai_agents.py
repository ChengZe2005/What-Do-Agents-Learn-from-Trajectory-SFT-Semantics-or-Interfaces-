"""
OpenAI API agents.

This file originally used the legacy openai<1.0 API:
  - openai.ChatCompletion.create(...)
  - openai.Completion.create(...)

But openai>=1.0 removed those symbols. We now prefer the new client interface:
  - from openai import OpenAI
  - client.chat.completions.create(...)
  - client.completions.create(...)

We keep a best-effort fallback for older openai versions.
"""

import openai
from src.agent import Agent
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
from typing import List, Callable
import dataclasses
from copy import deepcopy


def _is_openai_v1() -> bool:
    # openai>=1.0 provides `OpenAI` client and removes `ChatCompletion` attribute.
    return hasattr(openai, "OpenAI")


def _extract_chat_content(resp) -> str:
    """
    Support both:
    - openai<1.0 dict-like response: resp["choices"][0]["message"]["content"]
    - openai>=1.0 response object: resp.choices[0].message.content
    """
    # New SDK: pydantic models with attributes
    try:
        return resp.choices[0].message.content
    except Exception:
        pass
    # Old SDK: dict-like
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        pass
    raise ValueError(f"Unsupported OpenAI chat response type: {type(resp)}")


def _extract_completion_text(resp) -> str:
    """
    Support both:
    - openai<1.0 dict-like response: resp["choices"][0]["text"]
    - openai>=1.0 response object: resp.choices[0].text
    """
    try:
        return resp.choices[0].text
    except Exception:
        pass
    try:
        return resp["choices"][0]["text"]
    except Exception:
        pass
    raise ValueError(f"Unsupported OpenAI completion response type: {type(resp)}")


class OpenAIChatCompletion(Agent):
    def __init__(self, api_args=None, **config):
        if not api_args:
            api_args = {}
        print("api_args={}".format(api_args))
        print("config={}".format(config))
        
        api_args = deepcopy(api_args)
        api_key = api_args.pop("key", None) or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required, please assign api_args.key or set OPENAI_API_KEY environment variable.")
        os.environ['OPENAI_API_KEY'] = api_key
        # For openai<1.0 fallback path
        try:
            openai.api_key = api_key
        except Exception:
            pass
        print("OpenAI API key={}".format(api_key))
        api_base = api_args.pop("base", None) or os.getenv('OPENAI_API_BASE')
        if api_base:
            os.environ['OPENAI_API_BASE'] = api_base
            # For openai<1.0 fallback path
            try:
                openai.api_base = api_base
            except Exception:
                pass
        print("openai.api_base={}".format(api_base))
        api_args["model"] = api_args.pop("model", None)
        if not api_args["model"]:
            raise ValueError("OpenAI model is required, please assign api_args.model.")
        self.api_args = api_args
        self._api_key = api_key
        self._api_base = api_base
        self._client = None
        super().__init__(**config)

    def inference(self, history: List[dict]) -> str:
        history = json.loads(json.dumps(history))
        for h in history:
            if h['role'] == 'agent':
                h['role'] = 'assistant'

        # Prefer new SDK (openai>=1.0)
        if _is_openai_v1():
            if self._client is None:
                # openai>=1.0: base_url is the replacement of api_base
                # We keep OPENAI_API_KEY / OPENAI_API_BASE env vars set above too.
                self._client = openai.OpenAI(api_key=self._api_key, base_url=self._api_base)
            resp = self._client.chat.completions.create(messages=history, **self.api_args)
            return _extract_chat_content(resp)

        # Legacy fallback (openai<1.0)
        resp = openai.ChatCompletion.create(messages=history, **self.api_args)
        return _extract_chat_content(resp)


class OpenAICompletion(Agent):
    def __init__(self, api_args=None, **config):
        if not api_args:
            api_args = {}
        api_args = deepcopy(api_args)
        api_key = api_args.pop("key", None) or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required, please assign api_args.key or set OPENAI_API_KEY environment variable.")
        os.environ['OPENAI_API_KEY'] = api_key
        try:
            openai.api_key = api_key
        except Exception:
            pass
        print("OpenAI API key={}".format(api_key))
        api_base = api_args.pop("base", None) or os.getenv('OPENAI_API_BASE')
        if api_base:
            os.environ['OPENAI_API_BASE'] = api_base
            try:
                openai.api_base = api_base
            except Exception:
                pass
        print("openai.api_base={}".format(api_base))
        api_args["model"] = api_args.pop("model", None)
        if not api_args["model"]:
            raise ValueError("OpenAI model is required, please assign api_args.model.")
        self.api_args = api_args
        self._api_key = api_key
        self._api_base = api_base
        self._client = None
        super().__init__(**config)

    def inference(self, history: List[dict]) -> str:
        prompt = ""
        for h in history:
            role = 'Assistant' if h['role'] == 'agent' else h['role']
            content = h['content']
            prompt += f"{role}: {content}\n\n"
        prompt += 'Assistant: '

        if _is_openai_v1():
            if self._client is None:
                self._client = openai.OpenAI(api_key=self._api_key, base_url=self._api_base)
            resp = self._client.completions.create(prompt=prompt, **self.api_args)
            return _extract_completion_text(resp)

        resp = openai.Completion.create(prompt=prompt, **self.api_args)
        return _extract_completion_text(resp)
