import json
import re
import math
from typing import Callable, Dict, List, Any

import multiprocess as mp

from src.task import Task, Dataset, DataPiece, Session
from .Interaction import Container

import src.tasks.dbbench.disturb as disturb
import copy
big_prompt = """
I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose to operate or to answer.
your operation should be like this:
Action: Operation
```sql
SELECT * FROM table WHERE condition;
```
You MUST put SQL in markdown format without any other comments. Your SQL should be in one line.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If you are done operating, and you want to commit your final answer, then write down:
Action: Answer
Final Answer: ["ANSWER1", "ANSWER2", ...]
DO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
Your input will be raw MySQL response, you have to deal with it by yourself.
"""


def build_sql(entry, conn):
    name = entry["table"]["table_name"]
    columns = ",".join(
        [f"`{escape(column['name'], conn)}` TEXT" for column in entry["table"]["table_info"]["columns"]])
    column_names = ",".join(
        [f"`{escape(column['name'], conn)}`" for column in entry["table"]["table_info"]["columns"]])
    items = []
    for row in entry["table"]["table_info"]["rows"]:
        item = "("
        for col in row:
            item += f"'{escape(col, conn)}',"
        item = item[:-1] + ")"
        items.append(item)
    items = ",".join(items)
    sql = f"""CREATE DATABASE IF NOT EXISTS `{name}`;
USE `{name}`;
CREATE TABLE IF NOT EXISTS `{name}` ({columns});
INSERT INTO `{name}` ({column_names}) VALUES {items}; 
COMMIT;
"""
    return sql


def escape(string: str, conn):
    if type(string) is not str:
        string = str(string)
    return conn._cmysql.escape_string(string).decode("utf-8")


def process(receiver, max_round, disturb_type=0):
    container = Container()
    while True:
        data_item, session, sender = receiver.recv()

        if data_item is None and session is None and sender is None:
            break

        entry = data_item
        # container = self.container
        init = build_sql(entry, container.conn)
        container.execute(init)
        db = entry['table']['table_name']
        if disturb_type == 1:
            session.inject({"role": "user", "content": copy.deepcopy(disturb.prompt_1)})
        elif disturb_type == 2:
            session.inject({"role": "user", "content": copy.deepcopy(disturb.prompt_2)})
        elif disturb_type == 100:
            session.inject({"role": "user", "content": copy.deepcopy(disturb.prompt_100)})
        elif disturb_type == 101:
            session.inject({"role": "user", "content": copy.deepcopy(disturb.prompt_101)})
        else:
            session.inject({"role": "user", "content": big_prompt})
        session.inject({"role": "agent", "content": "Ok."})
        prompt = entry["description"] + "\n" + entry["add_description"]
        session.inject({"role": "user", "content": prompt})
        res = session.action()
        operation_keyword = 'Operation'
        answer_keyword = 'Answer'
        if disturb_type == 1:
            operation_keyword = 'Execute'
            answer_keyword = 'Reply'
        elif disturb_type == 2:
            operation_keyword = 'z1'
            answer_keyword = 'z2'
        elif disturb_type == 100 or disturb_type == 101:
            operation_keyword = ['Operation', 'Execute']
            answer_keyword = ['Answer', 'Reply']
        answer = ""
        error = ""
        original_action_count = 0
        synonym_action_count = 0
        if disturb_type == 100 or disturb_type == 101:
            try:
                # ignore case to be tolerant to model capitalization variations
                action = re.search(r"Action: (.*?)\n", res, flags=re.IGNORECASE)
                rounds = 0
                while action and rounds < max_round:
                    choose_original_action = False
                    choose_synonym_action = False
                    if action.group(1) == operation_keyword[0]:
                        choose_original_action = True
                    elif action.group(1) == operation_keyword[1]:
                        choose_synonym_action = True
                    if choose_original_action and choose_synonym_action:
                        raise ValueError("Both original and synonym actions are chosen")
                    if not(choose_original_action or choose_synonym_action):
                        break
                    if choose_original_action:
                        original_action_count += 1
                    if choose_synonym_action:
                        synonym_action_count += 1
                    res = re.search(r"```sql\n([\s\S]*?)\n```", res, flags=re.IGNORECASE)
                    if not res:
                        answer = ""
                        break
                    sql = res.group(1).strip()
                    sql = sql.replace("\n", " ")
                    response = container.execute(sql, db)
                    if response:
                        session.inject({"role": "user", "content": response})
                    else:
                        session.inject({"role": "user", "content": ""})
                    res = session.action()
                    action = re.search(r"Action: (.*?)\n", res)
                    rounds += 1
                else:
                    answer_original = re.search(rf"\nFinal {answer_keyword[0]}:(.*)", res, flags=re.IGNORECASE)
                    answer_synonym = re.search(rf"\nFinal {answer_keyword[1]}:(.*)", res, flags=re.IGNORECASE)
                    if answer_original:
                        original_action_count += 1
                        answer = answer_original.group(1)
                    elif answer_synonym:
                        synonym_action_count += 1
                        answer = answer_synonym.group(1)
                    else:
                        answer = ""
            except Exception as e:
                error = str(e)
                answer = ""
            else:
                error = ""
        else:
            try:
                # ignore case to be tolerant to model capitalization variations
                action = re.search(r"Action: (.*?)\n", res, flags=re.IGNORECASE)
                rounds = 0
                while action and action.group(1) == operation_keyword and rounds < max_round:
                    res = re.search(r"```sql\n([\s\S]*?)\n```", res, flags=re.IGNORECASE)
                    if not res:
                        answer = ""
                        break
                    sql = res.group(1).strip()
                    sql = sql.replace("\n", " ")
                    response = container.execute(sql, db)
                    if response:
                        session.inject({"role": "user", "content": response})
                    else:
                        session.inject({"role": "user", "content": ""})
                    res = session.action()
                    action = re.search(r"Action: (.*?)\n", res)
                    rounds += 1
                else:
                    answer = re.search(rf"\nFinal {answer_keyword}:(.*)", res, flags=re.IGNORECASE)
                    if answer:
                        answer = answer.group(1)
                    else:
                        answer = ""
            except Exception as e:
                error = str(e)
                answer = ""
            else:
                error = ""
        if data_item["type"][0] in ("INSERT", "DELETE", "UPDATE"):
            columns = ",".join([f"`{escape(column['name'], container.conn)}`"
                                for column in entry["table"]["table_info"]["columns"]])
            md5_query = f"select md5(group_concat(rowhash order by rowhash)) as hash " \
                        f"from( SELECT substring(MD5(CONCAT_WS(',', {columns})), 1, 5) AS rowhash FROM `{db}`) as sub;"
            answer = container.execute(md5_query, db)
        container.execute(f"drop database `{db}`")
        sender.send({
            "answer": str(answer),
            "type": entry["type"][0],
            "history": session.history,
            "original_action_count": original_action_count,
            "synonym_action_count": synonym_action_count,
            "error": error,
        })
    container.delete()


class DBBench(Task[Dict, Dict[str, Any], str]):
    def __init__(self, **configs):
        super().__init__(**configs)
        self.data_file = configs.pop("data_file")
        self.max_round = configs.pop("max_round", 5)
        self.disturb_type = configs.pop("disturb_type", 0)
        self.processes = []
        ctx = mp.get_context('spawn')
        for i in range(self.workers):
            receiver, sender = ctx.Pipe(False)
            p = ctx.Process(target=process, args=(receiver, self.max_round, self.disturb_type))
            p.start()
            self.processes.append((sender, ctx.Lock(), p))

    def escape(self, string: str, conn=None):
        conn = conn or self.conn
        if type(string) is not str:
            string = str(string)
        return conn._cmysql.escape_string(string).decode("utf-8")

    def get_data(self) -> Dataset[Dict, str]:
        dataset = Dataset()
        with open(self.data_file) as f:
            if self.data_file.endswith("json"):
                data = json.loads(f.read())
            else:
                data = [json.loads(line) for line in f.readlines()]

        for entry in data:
            if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
                ans = entry.pop("answer_md5")
            else:
                ans = entry.pop("label")
            inp = entry
            dataset.append(DataPiece(inp, ans))

        return dataset

    def predict_single(self, session: Session, data_item: Dict) -> Dict[str, Any]:
        ctx = mp.get_context('spawn')
        receiver, sender = ctx.Pipe(False)
        i = 0
        while True:
            if self.processes[i][1].acquire(timeout=0.2):
                break
            i += 1
            i %= self.workers
        self.processes[i][0].send((data_item, session, sender))
        ret = receiver.recv()
        self.processes[i][1].release()
        return ret

    @property
    def metrics(self) -> Dict[str, Callable[[List[Dict[str, Any]], List[str]], float]]:
        def factory(typ):
            def acc(inp: List[Dict[str, Any]], tar: List[str]) -> float:
                correct = 0
                total = 0
                for entry, cor in zip(inp, tar):
                    if not entry:
                        continue
                    ans, t = entry["answer"], entry["type"]
                    if t != typ and not (typ == "SELECT" and t not in ("INSERT", "UPDATE")):
                        continue
                    if t in ("INSERT", "DELETE", "UPDATE"):
                        correct += ans == cor
                    else:
                        try:
                            ans = list(eval(ans))
                        except:
                            ans = [ans]
                        if len(ans) == 1 and len(cor) == 1:
                            try:
                                correct += float(ans[0]) == float(cor[0])
                            except (ValueError, TypeError):
                                correct += ans[0] == cor[0]
                            else:
                                print(ans, cor)
                        else:
                            try:
                                cor = set(cor)
                                ans = set(ans)
                                correct += ans == cor
                            except:
                                pass
                    total += 1
                if total == 0:
                    print(f"WARNING: {typ} does not exist!")
                    return 0
                return correct / total

            return acc

        types = ['other', 'counting', 'comparison', 'ranking', 'aggregation-SUM', 'aggregation-MIN', 'aggregation-MAX',
                 'aggregation-AVG', 'SELECT', 'INSERT', 'UPDATE']

        ret = {}
        for typ in types:
            ret[typ + "_accuracy"] = factory(typ)

        ret["overall_cat_accuracy"] = lambda inp, tar: sum([ret[typ + "_accuracy"](inp, tar)
                                                            for typ in ("SELECT", "INSERT", "UPDATE")]) / 3

        def original_action_count(inp: List[Dict[str, Any]], tar: List[str]) -> int:
            if self.disturb_type not in (100, 101):
                return None
            return sum(entry.get("original_action_count", 0) for entry in inp if entry)

        def synonym_action_count(inp: List[Dict[str, Any]], tar: List[str]) -> int:
            if self.disturb_type not in (100, 101):
                return None
            return sum(entry.get("synonym_action_count", 0) for entry in inp if entry)

        def original_to_synonym_ratio(inp: List[Dict[str, Any]], tar: List[str]) -> float:
            if self.disturb_type not in (100, 101):
                return None
            original_count = original_action_count(inp, tar)
            synonym_count = synonym_action_count(inp, tar)
            if synonym_count == 0:
                return None
            return original_count / synonym_count

        def mp_factory(alpha: float, apply_exp: bool):
            def f(inp: List[Dict[str, Any]], tar: List[str]) -> float:
                if self.disturb_type not in (100, 101):
                    return None
                entries = [entry for entry in inp if entry]
                if len(entries) == 0:
                    return None
                values = []
                for entry in entries:
                    original = entry.get("original_action_count", 0)
                    synonym = entry.get("synonym_action_count", 0)
                    values.append(math.log((original + alpha) / (synonym + alpha)))
                mean_log = sum(values) / len(values)
                return math.exp(mean_log) if apply_exp else mean_log

            return f

        ret["original_action_count"] = original_action_count
        ret["synonym_action_count"] = synonym_action_count
        ret["original_to_synonym_ratio"] = original_to_synonym_ratio
        ret["mp_alpha0.5_exp"] = mp_factory(0.5, True)
        ret["mp_alpha1_exp"] = mp_factory(1.0, True)
        ret["mp_alpha2_exp"] = mp_factory(2.0, True)
        ret["mp_alpha0.5"] = mp_factory(0.5, False)
        ret["mp_alpha1"] = mp_factory(1.0, False)
        ret["mp_alpha2"] = mp_factory(2.0, False)

        def average_round(inp: List[Dict[str, Any]], tar: List[str]) -> float:
            count = 0
            total = 0
            for entry, cor in zip(inp, tar):
                if not entry:
                    continue
                count += len(entry["history"])
                total += 1
            return count / total if total else 0, total

        ret["average_round"] = average_round

        return ret

    def release(self):
        for sender, _, _ in self.processes:
            sender.send((None, None, None))
        for _, _, p in self.processes:
            p.join()
