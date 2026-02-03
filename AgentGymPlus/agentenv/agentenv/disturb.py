"""Interface disturbance utilities.

This module implements a lightweight *client-side* interface disturbance layer.
It renames action/tool/function identifiers exposed to the agent (prompt +
observations) while translating the agent's disturbed actions back to the
original interface expected by the environment server.

Disturb modes:
  0: no disturbance
  1: synonym disturbance (rename to human-readable synonyms)
  2: token disturbance (rename to meaningless tokens like z1, z2, ...)
  3: dual interface (keep original prompt; allow BOTH original+synonym names)
  4: dual interface (rewrite prompt to synonym; allow BOTH original+synonym names)

The goal is to make perturbations switchable via a single CLI flag, without
requiring changes to each env server.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping, Tuple


                                                                             
                                   
                                                                             


def _token_list(n: int, prefix: str = "z") -> List[str]:
    return [f"{prefix}{i}" for i in range(1, n + 1)]


def _semantic_token(orig: str, i: int) -> str:
    """Create a token alias that keeps a tiny semantic hint.

    Example:
      - "look around" -> "look_aroundzzz1"
      - "go to" -> "go_tozzz7"
    """
    base = orig.strip().lower()
    base = re.sub(r"\s+", "_", base)
    base = re.sub(r"[^a-z0-9_]+", "", base)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "tok"
    return f"{base}zzz{i}"


def _suffix_for_disturb2(idx: int) -> str:
    """Deterministic meaningless suffix for disturb=2 (synonym + suffix).

    Keep it short, stable, and 'nonsense-like'. Avoid characters that might
    break parsers. Use only [_a-z0-9].
    """
    return f"__q{idx}"


def _syn_plus_suffix_mapping(synonyms: Mapping[str, str], actions: List[str]) -> Dict[str, str]:
    """Build orig->(synonym+suffix) mapping for disturb=2, in the given action order."""
    out: Dict[str, str] = {}
    for i, orig in enumerate(actions, 1):
        syn = synonyms.get(orig)
        if syn is None:
                                                                                       
            syn = orig
        out[orig] = f"{syn}{_suffix_for_disturb2(i)}"
    return out


                                                                  
                         
                                                            
                                                      
                                                                      
_ENV_SPECS: Dict[str, Dict[str, Any]] = {
    "webshop": {
        "actions": ["search", "click"],
        "synonyms": {
            "search": "find",
            "click": "press",
        },
    },
    "webarena": {
        "actions": [
            "click",
            "type",
            "hover",
            "press",
            "scroll",
            "new_tab",
            "tab_focus",
            "close_tab",
            "goto",
            "go_back",
            "go_forward",
            "stop",
        ],
        "synonyms": {
            "click": "tap",
            "type": "input",
            "hover": "point",
            "press": "keypress",
            "scroll": "pan",
            "new_tab": "open_tab",
            "tab_focus": "focus_tab",
            "close_tab": "shut_tab",
            "goto": "navigate",
            "go_back": "back",
            "go_forward": "forward",
            "stop": "done",
        },
    },
    "alfworld": {
                                                             
                                                                             
        "actions": [
            "go to",
            "take",
            "put",
            "open",
            "close",
            "toggle",
            "heat",
            "cool",
            "clean",
            "inventory",
            "look",
            "examine",
            "use",
        ],
        "synonyms": {
            "go to": "navigate to",
            "take": "grab",
            "put": "place",
            "open": "unseal",
            "close": "seal",
            "toggle": "switch",
            "heat": "warm",
            "cool": "chill",
            "clean": "wash",
            "inventory": "bag",
            "look": "observe",
            "examine": "inspect",
            "use": "employ",
        },
    },
    "babyai": {
        "actions": [
            "toggle and go through",
            "turn right",
            "turn left",
            "move forward",
            "go to",
            "pick up",
            "go through",
            "toggle",
        ],
        "synonyms": {
            "toggle and go through": "open and enter",
            "turn right": "rotate right",
            "turn left": "rotate left",
            "move forward": "advance",
            "go to": "navigate to",
            "pick up": "grab",
            "go through": "enter",
            "toggle": "switch",
        },
    },
    "maze": {
        "actions": ["move up", "move down", "move left", "move right"],
        "synonyms": {
            "move up": "go north",
            "move down": "go south",
            "move left": "go west",
            "move right": "go east",
        },
        "token_overrides": {
            "move up": "zUPz",
            "move down": "zDOWNz",
            "move left": "zLEFTz",
            "move right": "zRIGHTz",
        },
    },
    "textcraft": {
        "actions": ["get", "inventory", "craft"],
        "synonyms": {
            "get": "take",
            "inventory": "bag",
            "craft": "make",
        },
    },
    "sciworld": {
        "actions": [
            "look around",
            "look at",
            "look in",
            "put down",
            "pick up",
            "focus on",
            "go to",
            "deactivate",
            "disconnect",
            "activate",
            "connect",
            "inventory",
            "examine",
            "move",
            "open",
            "close",
            "use",
            "read",
            "pour",
            "dunk",
            "mix",
            "eat",
            "flush",
            "wait1",
            "wait",
            "task",
        ],
        "synonyms": {
            "look around": "survey",
            "look at": "inspect",
            "look in": "peek",
            "put down": "drop item",
            "pick up": "grab",
            "focus on": "target",
            "go to": "navigate to",
            "deactivate": "stop",
            "disconnect": "unlink",
            "activate": "start",
            "connect": "link",
            "inventory": "bag",
            "examine": "analyze",
            "move": "relocate",
            "open": "unseal",
            "close": "seal",
            "use": "utilize",
            "read": "peruse",
            "pour": "decant",
            "dunk": "immerse",
            "mix": "blend",
            "eat": "consume",
            "flush": "rinse",
            "wait1": "pause1",
            "wait": "pause",
            "task": "objective",
        },
    },
    "searchqa": {
                                                         
        "actions": [
            "<think>",
            "</think>",
            "<search>",
            "</search>",
            "<information>",
            "</information>",
            "<answer>",
            "</answer>",
        ],
        "synonyms": {
            "<think>": "<reason>",
            "</think>": "</reason>",
            "<search>": "<lookup>",
            "</search>": "</lookup>",
            "<information>": "<evidence>",
            "</information>": "</evidence>",
            "<answer>": "<response>",
            "</answer>": "</response>",
        },
                                                                       
        "token_overrides": {
            "<think>": "<z1>",
            "</think>": "</z1>",
            "<search>": "<z2>",
            "</search>": "</z2>",
            "<information>": "<z3>",
            "</information>": "</z3>",
            "<answer>": "<z4>",
            "</answer>": "</z4>",
        },
    },
    "weather": {
        "actions": [
            "get_user_current_date",
            "get_user_current_location",
            "get_historical_temp",
            "get_historical_rain",
            "get_historical_snow",
            "get_snow_forecast",
            "get_current_snow",
            "get_current_temp",
            "get_latitude_longitude",
            "get_elevation",
            "get_temp_forecast",
            "get_rain_forecast",
            "get_current_rain",
            "get_distance",
            "get_historical_air_quality_index",
            "get_current_air_quality_index",
            "get_air_quality_level",
            "check_valid_actions",
            "finish",
        ],
        "synonyms": {
            "get_user_current_date": "fetch_user_current_date",
            "get_user_current_location": "fetch_user_current_location",
            "get_historical_temp": "fetch_past_temperature",
            "get_historical_rain": "fetch_past_rainfall",
            "get_historical_snow": "fetch_past_snowfall",
            "get_snow_forecast": "fetch_snow_forecast",
            "get_current_snow": "fetch_current_snow",
            "get_current_temp": "fetch_current_temperature",
            "get_latitude_longitude": "geo_lookup",
            "get_elevation": "fetch_elevation",
            "get_temp_forecast": "fetch_temperature_forecast",
            "get_rain_forecast": "fetch_rain_forecast",
            "get_current_rain": "fetch_current_rain",
            "get_distance": "compute_distance",
            "get_historical_air_quality_index": "fetch_past_aqi",
            "get_current_air_quality_index": "fetch_current_aqi",
            "get_air_quality_level": "aqi_level",
            "check_valid_actions": "list_actions",
            "finish": "submit",
        },
    },
    "todo": {
        "actions": [
            "get_user_current_date",
            "get_user_current_location",
            "get_projects",
            "update_project",
            "get_tasks",
            "get_task_description",
            "get_task_duration",
            "complete_task",
            "update_task",
            "delete_task",
            "check_valid_actions",
            "finish",
        ],
        "synonyms": {
            "get_user_current_date": "fetch_user_current_date",
            "get_user_current_location": "fetch_user_current_location",
            "get_projects": "list_projects",
            "update_project": "set_project",
            "get_tasks": "list_tasks",
            "get_task_description": "fetch_task_description",
            "get_task_duration": "fetch_task_duration",
            "complete_task": "mark_task_done",
            "update_task": "set_task",
            "delete_task": "remove_task",
            "check_valid_actions": "list_actions",
            "finish": "submit",
        },
    },
    "movie": {
        "actions": [
            "get_search_movie",
            "get_movie_details",
            "get_movie_production_companies",
            "get_movie_production_countries",
            "get_movie_cast",
            "get_movie_crew",
            "get_movie_keywords",
            "get_search_person",
            "get_person_details",
            "get_person_cast",
            "get_person_crew",
            "get_person_external_ids",
            "get_movie_alternative_titles",
            "get_movie_translation",
            "check_valid_actions",
            "finish",
        ],
        "synonyms": {
            "get_search_movie": "find_movie",
            "get_movie_details": "fetch_movie_details",
            "get_movie_production_companies": "fetch_movie_companies",
            "get_movie_production_countries": "fetch_movie_countries",
            "get_movie_cast": "fetch_movie_cast",
            "get_movie_crew": "fetch_movie_crew",
            "get_movie_keywords": "fetch_movie_keywords",
            "get_search_person": "find_person",
            "get_person_details": "fetch_person_details",
            "get_person_cast": "fetch_person_cast",
            "get_person_crew": "fetch_person_crew",
            "get_person_external_ids": "fetch_person_external_ids",
            "get_movie_alternative_titles": "fetch_movie_alt_titles",
            "get_movie_translation": "fetch_movie_translation",
            "check_valid_actions": "list_actions",
            "finish": "submit",
        },
    },
    "sheet": {
        "actions": [
            "open_sheet",
            "del_sheet",
            "freeze_data",
            "get_A1_annotation",
            "insert_cols",
            "insert_rows",
            "delete_batch_data",
            "update_cell",
            "update_cell_by_formula",
            "update_range",
            "sort_sheet_by_col",
            "merge_cells",
            "update_note",
            "get_all_values",
            "get_range_values",
            "get_cell_value",
            "get_value_by_formula",
            "filter_cells",
            "get_note",
            "finish",
        ],
        "synonyms": {
            "open_sheet": "load_sheet",
            "del_sheet": "remove_sheet",
            "freeze_data": "lock_panes",
            "get_A1_annotation": "coord_to_a1",
            "insert_cols": "add_columns",
            "insert_rows": "add_rows",
            "delete_batch_data": "remove_batch",
            "update_cell": "set_cell",
            "update_cell_by_formula": "set_cell_by_formula",
            "update_range": "set_range",
            "sort_sheet_by_col": "order_by_column",
            "merge_cells": "combine_cells",
            "update_note": "set_note",
            "get_all_values": "fetch_all_values",
            "get_range_values": "fetch_range",
            "get_cell_value": "fetch_cell",
            "get_value_by_formula": "compute_value_by_formula",
            "filter_cells": "query_cells",
            "get_note": "fetch_note",
            "finish": "submit",
        },
    },
    "academia": {
        "actions": [
            "loadPaperNet",
            "loadAuthorNet",
            "neighbourCheck",
            "paperNodeCheck",
            "authorNodeCheck",
            "authorEdgeCheck",
            "finish",
        ],
        "synonyms": {
            "loadPaperNet": "initPaperNet",
            "loadAuthorNet": "initAuthorNet",
            "neighbourCheck": "adjacencyCheck",
            "paperNodeCheck": "paperInfo",
            "authorNodeCheck": "authorInfo",
            "authorEdgeCheck": "authorLinkCheck",
            "finish": "submit",
        },
    },

                                                                         
                     
                                                                         
    "sqlgym": {
                                                                                   
                                                             
        "actions": ["```sql"],
        "synonyms": {
            "```sql": "```query",
        },
                                                                         
        "token_overrides": {
            "```sql": "```z",
        },
    },
    "wordle": {
                                                                                
                                                                
        "actions": ["Action:"],
        "synonyms": {
            "Action:": "Guess:",
        },
        "token_overrides": {
            "Action:": "z:",
        },
    },
}




                                                               
_DUAL_INTERFACE_ENVS = {"alfworld", "babyai", "sciworld", "weather", "wordle", "maze"}
                                                                             
                     
                                                                             


_ALNUM_UNDERSCORE_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _build_token_pattern(token: str) -> re.Pattern:
    """Build a safe regex pattern that tries to replace whole tokens/phrases."""
    escaped = re.escape(token)

                                                                        
    if "<" in token or ">" in token:
        return re.compile(escaped)

                                                                              
    if " " in token:
        return re.compile(rf"(?<!\w){escaped}(?!\w)")

                                              
    if _ALNUM_UNDERSCORE_RE.match(token):
        return re.compile(rf"\b{escaped}\b")

    return re.compile(escaped)


def _replace_with_patterns(text: str, mapping: Mapping[str, str], patterns: Mapping[str, re.Pattern]) -> str:
    for k in sorted(mapping.keys(), key=len, reverse=True):
        pat = patterns[k]
        text = pat.sub(mapping[k], text)
    return text


def _deep_transform(obj: Any, transform_str) -> Any:
    if isinstance(obj, str):
        return transform_str(obj)
    if isinstance(obj, list):
        return [_deep_transform(v, transform_str) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_deep_transform(v, transform_str) for v in obj)
    if isinstance(obj, dict):
        return {k: _deep_transform(v, transform_str) for k, v in obj.items()}
    return obj


                                                                             
                 
                                                                             


@dataclass
class DisturbManager:
    """A small helper that applies disturbance on prompts/observations and
    translates disturbed actions back to the original interface.

    Disturb modes:
      0: no disturbance
      1: synonym disturbance (rename to human-readable synonyms) [STRICT]
      2: token disturbance (rename to meaningless tokens like z1, z2, ...) [STRICT]
      3: dual interface (keep original prompt, allow BOTH original+synonym) [NON-STRICT]
      4: dual interface (rewrite prompt to synonym, allow BOTH original+synonym) [NON-STRICT]

    Notes:
      - For STRICT modes (1/2), using the *original* interface is treated as
        deprecated and rejected on the client side.
      - For dual modes (3/4), both interfaces are accepted; we translate synonym
        forms back to original for the env server, and keep per-episode call counts:
          * disturb0_calls: #times the model used original names
          * disturb1_calls: #times the model used synonym names
    """

    env_key: str
    disturb: int = 0

                                                      
    disturb0_calls: int = 0
    disturb1_calls: int = 0

    def reset_call_counts(self) -> None:
        self.disturb0_calls = 0
        self.disturb1_calls = 0

    def __post_init__(self) -> None:
        self.env_key = (self.env_key or "").lower()
        self.disturb = int(self.disturb)

        if self.disturb not in (0, 1, 2, 3, 4):
            raise ValueError(f"disturb must be 0/1/2/3/4, got {self.disturb}")

                                                                         
        if self.disturb in (3, 4) and self.env_key not in _DUAL_INTERFACE_ENVS:
                                                                        
            self.disturb = 0

        self._spec = _ENV_SPECS.get(self.env_key, None)
        if self._spec is None:
                                                   
            self.disturb = 0
            self.orig2alias: Dict[str, str] = {}
            self.alias2orig: Dict[str, str] = {}
            self._orig_patterns: Dict[str, re.Pattern] = {}
            self._alias_patterns: Dict[str, re.Pattern] = {}
            self._display_enabled = False
            self._strict = False
            return

        actions: List[str] = list(self._spec["actions"])

                                                  
        if self.disturb == 0:
            self.orig2alias = {}
        elif self.disturb == 1:
            self.orig2alias = dict(self._spec["synonyms"])
        elif self.disturb == 2:
                                                                     
                                                                              
            if self.env_key in ("alfworld", "webshop", "sciworld"):
                self.orig2alias = _syn_plus_suffix_mapping(self._spec["synonyms"], actions)
            else:
                                   
                if "token_overrides" in self._spec:
                    self.orig2alias = dict(self._spec["token_overrides"])
                else:
                                                                                 
                                                        
                    if self.env_key == "sciworld":
                        self.orig2alias = {
                            a: _semantic_token(a, idx)
                            for idx, a in enumerate(actions, 1)
                        }
                    else:
                        tokens = _token_list(len(actions), prefix="z")
                        self.orig2alias = {a: t for a, t in zip(actions, tokens)}
        else:
                                                                                          
            self.orig2alias = dict(self._spec["synonyms"])

        self.alias2orig = {v: k for k, v in self.orig2alias.items()}

        self._orig_patterns = {k: _build_token_pattern(k) for k in self.orig2alias}
        self._alias_patterns = {k: _build_token_pattern(k) for k in self.alias2orig}

                                                       
        self._display_enabled = self.disturb in (1, 2, 4) and bool(self.orig2alias)
                                        
        self._strict = self.disturb in (1, 2) and bool(self.orig2alias)

    @property
    def enabled(self) -> bool:
                                                                                       
        return self.disturb in (1, 2, 3, 4) and bool(getattr(self, "orig2alias", {}))

    @property
    def dual(self) -> bool:
        return self.disturb in (3, 4) and self.env_key in _DUAL_INTERFACE_ENVS and bool(getattr(self, "orig2alias", {}))

                                                                    
    def apply_to_text(self, text: str) -> str:
        """Replace original identifiers with disturbed identifiers (display-side)."""
        if not self.enabled or not self._display_enabled:
            return text
        return _replace_with_patterns(text, self.orig2alias, self._orig_patterns)

    def _build_dual_interface_note(self) -> str:
        """Build the disturb-3/4 mapping note appended to the first human message."""
        if not self.dual:
            return ""
        synonyms: Mapping[str, str] = self._spec.get("synonyms", {})
        actions: List[str] = list(self._spec.get("actions", []))

        lines: List[str] = []
        if self.disturb == 3:
            lines.append("[Disturb-3: Dual Interface]")
            lines.append("You may use EITHER the original action/tool names (disturb0) OR the synonym names (disturb1).")
            lines.append("When acting, choose EXACTLY ONE name (do NOT include both names in a single action).")
            lines.append("They are equivalent. Mapping (disturb0 -> disturb1):")
            for orig in actions:
                syn = synonyms.get(orig)
                if syn is not None:
                    lines.append(f"- {orig} -> {syn}")
        else:
                          
            lines.append("[Disturb-4: Dual Interface (Reversed Order)]")
            lines.append("You may use EITHER the synonym names (disturb1) OR the original action/tool names (disturb0).")
            lines.append("When acting, choose EXACTLY ONE name (do NOT include both names in a single action).")
            lines.append("They are equivalent. Mapping (disturb1 -> disturb0):")
            for orig in actions:
                syn = synonyms.get(orig)
                if syn is not None:
                    lines.append(f"- {syn} -> {orig}")

        return "\n\n" + "\n".join(lines) + "\n"

    def apply_to_conversation_start(
        self, conversation_start: Tuple[Mapping[str, Any], ...]
    ) -> Tuple[Mapping[str, Any], ...]:
        if not self.enabled:
            return conversation_start

        new_msgs: List[Mapping[str, Any]] = []
        for msg in conversation_start:
            msg2 = dict(msg)
            if isinstance(msg2.get("value"), str):
                                                                       
                msg2["value"] = self.apply_to_text(msg2["value"])
            new_msgs.append(msg2)

                                                                                     
        if self.dual and new_msgs:
            note = self._build_dual_interface_note()
            if note and isinstance(new_msgs[0].get("value"), str):
                new_msgs[0]["value"] = new_msgs[0]["value"] + note

        return tuple(new_msgs)

    def apply_to_any(self, obj: Any) -> Any:
        if not self.enabled or not self._display_enabled:
            return obj
        return _deep_transform(obj, self.apply_to_text)

                                                                                    
    def _to_server_text(self, text: str) -> str:
        if not self.enabled:
            return text
                                                                                  
        return _replace_with_patterns(text, self.alias2orig, self._alias_patterns)

    def _contains_original_token(self, text: str) -> str | None:
        """Return the first original token found in text (for strict invalidation)."""
        if not self.enabled or not self._strict:
            return None
                                                                      
        for tok in sorted(self.orig2alias.keys(), key=len, reverse=True):
            pat = _build_token_pattern(tok)
            if pat.search(text):
                return tok
        return None

                                                                
    _TOOL_ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_]+)")

    _DEPRECATED_MARK = "Invalid Action, the action is deprecated: "

    def _deprecated_error(self, action: str) -> str:
        """Return the standardized deprecated-action error message.

        NOTE: The substring "Invalid Action, the action is deprecated" must appear
        exactly once in the returned string (for reliable counting).
        """
        msg = f"{self._DEPRECATED_MARK}{action}"
        alias = self.orig2alias.get(action)
        if alias:
            msg += f"\nUse '{alias}' instead."
        return msg

    def translate_tool_call_text(self, full_text: str) -> Tuple[str | None, str | None]:
        """Translate disturbed tool name to original.

        Returns (translated_text, error_message). If error_message is not None,
        the caller should treat it as an invalid action and avoid sending to the
        env server.
        """
        if not self.enabled:
            return full_text, None

        m = self._TOOL_ACTION_RE.search(full_text)
        if not m:
                                                                                     
            if self._strict:
                bad = self._contains_original_token(full_text)
                if bad is not None:
                    return None, self._deprecated_error(bad)
                                                                             
            return self._to_server_text(full_text), None

        name = m.group(1)

                                                               
        if self.dual:
            if name in self.alias2orig:
                self.disturb1_calls += 1
                orig = self.alias2orig[name]
                translated = full_text[: m.start(1)] + orig + full_text[m.end(1) :]
                return translated, None
            if name in self.orig2alias:
                self.disturb0_calls += 1
                return full_text, None
            return full_text, None

                                                              
        if self._strict:
            if name in self.orig2alias:
                return None, self._deprecated_error(name)
            if name in self.alias2orig:
                orig = self.alias2orig[name]
                translated = full_text[: m.start(1)] + orig + full_text[m.end(1) :]
                return translated, None
                                              
            return full_text, None

                                                               
        return self._to_server_text(full_text), None

    def translate_prefix_command(self, command: str) -> Tuple[str | None, str | None]:
        """Translate disturbed verb prefix to original.

        Works for commands like:
          - "go to kitchen"
          - "search[query]"
          - "click [123]"

        Returns (translated_command, error_message).
        """
        if not self.enabled:
            return command, None

        raw = command
        s = raw.lstrip()
        leading = raw[: len(raw) - len(s)]
        s_low = s.lower()

                                                     
        for alias in sorted(self.alias2orig.keys(), key=len, reverse=True):
            if s_low.startswith(alias.lower()):
                orig = self.alias2orig[alias]
                if self.dual:
                    self.disturb1_calls += 1
                return leading + orig + s[len(alias):], None

                                                                                
        if self.dual:
            for orig in sorted(self.orig2alias.keys(), key=len, reverse=True):
                pat = self._orig_patterns.get(orig) or _build_token_pattern(orig)
                if pat.match(s):
                    self.disturb0_calls += 1
                    return command, None
            return command, None

                                                        
        if self._strict:
            for orig in sorted(self.orig2alias.keys(), key=len, reverse=True):
                pat = self._orig_patterns.get(orig) or _build_token_pattern(orig)
                if pat.match(s):
                    return None, self._deprecated_error(orig)

        return command, None

    def translate_xml_tags_text(self, full_text: str) -> Tuple[str | None, str | None]:
        """Translate disturbed XML-like tags back to original (SearchQA)."""
        if not self.enabled:
            return full_text, None

        if self._strict:
            bad = self._contains_original_token(full_text)
            if bad is not None:
                return None, self._deprecated_error(bad)

                                                                      
        return self._to_server_text(full_text), None
