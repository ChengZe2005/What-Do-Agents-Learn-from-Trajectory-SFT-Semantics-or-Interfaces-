import os

from openai_eval import EvalArguments, main

args = EvalArguments(
    api_key=os.environ.get("api_key", ""),
    base_url=os.environ.get("base_url", ""),
    model=os.environ.get("model", ""),
    inference_file="/path/to/inference.json",
    output_dir="/path/to/output_dir",
    task_name="alfworld",
    max_round=20,
    env_server_base=os.environ.get("env_server_base", "http://127.0.0.1:36002"),
)

if __name__ == "__main__":
    main(vars(args))
