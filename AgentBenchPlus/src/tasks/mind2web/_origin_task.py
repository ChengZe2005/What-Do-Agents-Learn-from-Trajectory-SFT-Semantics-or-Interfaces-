from src.task import Task, DataPiece, Dataset
from src.agent import Agent, Session
import pickle
from addict import Dict
from src.tasks.mind2web.dataloader import get_data_split, MultiChoiceDataset, format_input_multichoice
from transformers import AutoTokenizer
from typing import Callable, List, Any, Tuple, Optional, Union
import random
import re
import math
import numpy as np
import json
import time


def fetch_data(session: Session, prompt_template, max_attempts=20):
    for attempt in range(max_attempts):
        try:
            # print('---------prompt_template---------')
            # print(prompt_template)
            # print('---------end prompt_template---------')
            output = session.action(prompt_template)
            # print('---------output---------')
            # print(output)
            # print('---------end output---------')
            assert output is not None
            return output
        except AssertionError:
            print(f"Output is None! Retrying...({attempt + 1}/{max_attempts})")
            time.sleep(3)
    raise RuntimeError("Failed to fetch data after several attempts")

class Mind2Web(Task):
    def __init__(self, **config):
        self.count = config.pop("count", 100000)
        self.range_min = config.pop("range_min", 0)
        self.range_max = config.pop("range_max", 100)
        self.disturb_type = config.pop("disturb_type", 0)
        cfg = Dict(config)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
        # Load rank of candidates
        candidate_results = None
        if cfg.data.score_file is not None:
            with open(cfg.data.score_file, "rb") as f:
                candidate_results = pickle.load(f)

        self.test_dataset_dict = {}
        for test_key, test_split_file in cfg.data.test_split_files.items():
            test_data = get_data_split(
                cfg.data.data_path,
                test_split_file,
                candidate_results=candidate_results,
                cache_dir=cfg.data.cache_path,
                is_debug=bool(cfg.debug)
            )
            self.test_dataset_dict[test_key] = MultiChoiceDataset(
                test_data,
                tokenizer=tokenizer,
                neg_ratio=cfg.train.neg_ratio,
                num_candidates=cfg.train.num_candidates,
                max_context_len=cfg.train.max_context_len,
                disturb_type=self.disturb_type,
            )
        # evaluation configs
        random.seed(cfg.seed)
        self.top_k = cfg.eval.topk
        self.candidates_num = cfg.train.num_candidates
        prompt_path = cfg.llm_prompt
        if self.disturb_type == 1:
            prompt_path = cfg.get(
                "llm_prompt_synonym",
                cfg.llm_prompt.replace(".json", "_synonym.json")
            )
        elif self.disturb_type == 2:
            prompt_path = cfg.get(
                "llm_prompt_nonsense",
                cfg.llm_prompt.replace(".json", "_nonsense.json")
            )
        elif self.disturb_type == 101:
            prompt_path = cfg.get(
                "llm_prompt_101",
                cfg.llm_prompt.replace(".json", "_101.json")
            )
        elif self.disturb_type == 100:
            prompt_path = cfg.get(
                "llm_prompt_100",
                cfg.llm_prompt.replace(".json", "_100.json")
            )
        with open(prompt_path, "r") as f:
            self.prompt_template = json.load(f)

        super().__init__(**config)

    @property
    def metrics(self) -> Dict:
        return {
            "score": self.metric,
        }

    def metric(self, output: List[Dict], target: List[Dict]):
        prediction = []
        for x in output:
            prediction.append(x['final_prediction'])
            # if x is not None:
            #     prediction.append(x['final_prediction'])
            # else:
            #     prediction.append(None)
        all_element_acc, all_action_f1, all_step_sr = [], [], []
        assert len(target) == len(prediction)
        for pred, tar in zip(prediction, target):
            if tar is None or pred is None:
                all_element_acc.append(0)
                all_action_f1.append(0)                
            else:
                if pred[0] in tar['element']:
                    all_element_acc.append(1)
                else:
                    all_element_acc.append(0)
                all_action_f1.append(
                    self.calculate_f1(pred[1], tar['action'])
                )
            if all_element_acc[-1] > 0 and all_action_f1[-1] >= 1.0:
                all_step_sr.append(1)
            else:
                all_step_sr.append(0)
        overall = {
            'element_acc': sum(all_element_acc) / len(all_element_acc) * 100,
            'action_f1': np.mean(all_action_f1) * 100,
            'step_sr': np.mean(all_step_sr) * 100
        }
        if self.disturb_type in (100, 101):
            choose_original_action_count = 0
            choose_synonym_action_count = 0
            mp_logs = {0.5: [], 1.0: [], 2.0: []}
            for item in output:
                if not item:
                    continue
                original_count = item.get("choose_original_action_count", 0)
                synonym_count = item.get("choose_synonym_action_count", 0)
                choose_original_action_count += original_count
                choose_synonym_action_count += synonym_count
                for alpha in mp_logs:
                    mp_logs[alpha].append(math.log((original_count + alpha) / (synonym_count + alpha)))
            if choose_synonym_action_count == 0:
                original_over_synonym_ratio = None
            else:
                original_over_synonym_ratio = (
                    choose_original_action_count / choose_synonym_action_count
                )
            mp_avgs = {
                alpha: (sum(logs) / len(logs) if logs else None)
                for alpha, logs in mp_logs.items()
            }
            overall.update(
                {
                    "choose_original_action_count": choose_original_action_count,
                    "choose_synonym_action_count": choose_synonym_action_count,
                    "original_over_synonym_ratio": original_over_synonym_ratio,
                    "mp_alpha0.5_exp": math.exp(mp_avgs[0.5]) if mp_avgs[0.5] is not None else None,
                    "mp_alpha1_exp": math.exp(mp_avgs[1.0]) if mp_avgs[1.0] is not None else None,
                    "mp_alpha2_exp": math.exp(mp_avgs[2.0]) if mp_avgs[2.0] is not None else None,
                    "mp_alpha0.5": mp_avgs[0.5],
                    "mp_alpha1": mp_avgs[1.0],
                    "mp_alpha2": mp_avgs[2.0],
                }
            )
        return overall
    
    def get_data(self): 
        ret = Dataset()
        for test_dataset in self.test_dataset_dict.values():
            idx = 0
            for sample in test_dataset.data:
                pos_candidates = sample["pos_candidates"]
                pos_candidates = [c for c in pos_candidates if c["rank"] < self.top_k]
                pos_ids = [c["backend_node_id"] for c in pos_candidates]
                sample.pop("pos_candidates")
                sample["pos_ids"] = pos_ids
                if len(pos_ids) == 0:
                    ret.append(DataPiece(sample, None))
                    continue
                _, _, target_out, _ = format_input_multichoice(
                    sample, pos_ids[:1], pos_ids[0], disturb_type=self.disturb_type
                )
                _, target_action = self.postprocess_action(target_out)
                target = {'element': pos_ids, 'action': target_action}
                ret.append(DataPiece(sample, target))
                idx += 1 
                if idx >= self.count:
                    break
        # Candidate generator
        for k in [5, 10, 20, 50]:
            recall_at_k = np.mean(
                [
                    1 if any([c["rank"] < k for c in sample["pos_candidates"]]) else 0
                    for sample in test_dataset.data
                ]
            )
            print(f"Recall Cap @ {k}: {recall_at_k}")
        acc = np.mean(
                [
                    1 if any([c["rank"] == 0 for c in sample["pos_candidates"]]) else 0
                    for sample in test_dataset.data
                ]
            )
        print(f"Candidate generator acc: {acc}")
        return ret          

    def predict_single(self, session: Session, sample: Dict): 
        if len(sample["pos_ids"]) == 0:
            return {"final_prediction":  ('', ''), "outputs": []}
        pos_ids = sample["pos_ids"]        
        neg_candidates = sample["neg_candidates"]
        neg_candidates = [c for c in neg_candidates if c["rank"] < self.top_k]
        neg_ids = [c["backend_node_id"] for c in neg_candidates]
        all_candidates = pos_ids + neg_ids
        random.shuffle(all_candidates)
        final_prediction = None
        outputs = []
        choose_original_action_count = 0
        choose_synonym_action_count = 0
        while len(all_candidates) > 1:
            candidate_ids = all_candidates[:self.candidates_num] # 5
            all_candidates = all_candidates[self.candidates_num:]
            seq_context, seq_in, _, choices = format_input_multichoice(
                sample,
                candidate_ids,
                -1,
                keep_html_brackets=True,
                disturb_type=self.disturb_type,
            )
            outputs.append(
                [candidate_ids, [seq_context, seq_in, choices], None]
            )
            self.prompt_template[-1][
                    "content"
                ] = f"'''\n{seq_context}\n'''\n\n{seq_in}"
            
            session.history = []
            output = fetch_data(session, self.prompt_template)
            # print(session.history[-1])
            # output = "CLICK "
            outputs[-1][-1] = output
            (
                pred_element,
                pred_action,
                choose_original_action,
                choose_synonym_action,
            ) = self.postprocess_action_llm(output)
            if self.disturb_type in (100, 101):
                if choose_original_action:
                    choose_original_action_count += 1
                if choose_synonym_action:
                    choose_synonym_action_count += 1
            if pred_element != "A":
                # convert B, C, D to 0, 1, 2
                pred_element = ord(pred_element) - ord("B")
                try:
                    pred_element = choices[pred_element][0]
                    all_candidates.append(pred_element)
                    final_prediction = (pred_element, pred_action)
                except IndexError:
                    print(f"IndexError: {output}")
        if final_prediction == None or len(all_candidates) == 0:
            final_prediction = ('', '')
        result = {"final_prediction": final_prediction, "outputs": outputs}
        if self.disturb_type in (100, 101):
            result.update(
                {
                    "choose_original_action_count": choose_original_action_count,
                    "choose_synonym_action_count": choose_synonym_action_count,
                }
            )
        return result
    
    def postprocess_action(self, text):
        # C.
        # Action: SELECT
        # Value: Queen
        text = text.strip()
        selected_option = text[0]
        action = re.search(r"Action: (CLICK|SELECT|TYPE)", text)
        action = action.group(1) if action is not None else ""
        value = re.search(r"Value: (.*)$", text, re.MULTILINE)
        value = value.group(1) if value is not None else ""
        return selected_option, action.strip() + " " + value.strip()

    def postprocess_action_llm(self, text):
        # Parse LLM output, tolerating prompt perturbations.
        # Examples:
        # Answer: C. / Pick: C. / RUNE: C.
        # Action: CLICK / SELECT / TYPE / PRESS / CHOOSE / WRITE / ZAP / ZIG / ZUG
        # Value: xxx
        text = text.strip()
        # option letter
        opt_match = re.search(r"(Answer|Pick|RUNE)\s*:\s*([A-F])", text, re.IGNORECASE)
        selected_option = opt_match.group(2) if opt_match else "A"
        # action with aliases
        action_match = re.search(
            r"Action:\s*(CLICK|SELECT|TYPE|PRESS|CHOOSE|WRITE|ZAP|ZIG|ZUG)",
            text,
            re.IGNORECASE,
        )
        action_raw = action_match.group(1).upper() if action_match else ""
        action_alias = {
            "PRESS": "CLICK",
            "CHOOSE": "SELECT",
            "WRITE": "TYPE",
            # nonsensical tokens mapped to canonical intents
            "ZAP": "CLICK",
            "ZIG": "SELECT",
            "ZUG": "TYPE",
        }
        choose_original_action = False
        choose_synonym_action = False
        if action_raw in action_alias:
            # 说明使用了别名
            choose_synonym_action = True
        elif action_raw:
            choose_original_action = True
        action = action_alias.get(action_raw, action_raw)
        # value
        value_match = re.search(r"Value:\s*(.*)$", text, re.MULTILINE | re.IGNORECASE)
        value = value_match.group(1) if value_match else ""
        return (
            selected_option,
            action.strip() + " " + value.strip(),
            choose_original_action,
            choose_synonym_action,
        )

    def calculate_f1(self, pred, label):
        pred = set(pred.strip().split())
        label = set(label.strip().split())
        if len(pred) == 0 and len(label) == 0:
            return 1
        if len(pred) == 0 or len(label) == 0:
            return 0

        tp = len(pred & label)
        fp = len(pred - label)
        fn = len(label - pred)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision == 0 or recall == 0:
            return 0
        f1 = 2 * precision * recall / (precision + recall)
        return f1
