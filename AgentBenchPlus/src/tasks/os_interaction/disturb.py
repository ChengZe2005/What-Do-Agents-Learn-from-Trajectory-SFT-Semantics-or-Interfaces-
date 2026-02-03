prompt_1 = {
    "role": "user",
    "content": """You are an assistant that will act like a person; I will play the role of a Linux (Ubuntu) operating system. Your goal is to carry out the operations I request or respond to my questions.
For each of your turns, you should first think about what to do, and then take exactly one of the three actions: "exec", "done", or "reply".
1. If you think you should run some system shell commands, take the exec action, and you should print like this:

Think: put your thought here.

Act: exec

```bash
# put your shell commands here
```

2. If you think you have completed the task, take the done action, and you should print like this:

Think: put your thought here.

Act: done

3. If you think you already have the final response to the question, take the reply action, and you should print like this:

Think: put your thought here.

Act: reply(Your answer to the question should be put in this pair of parentheses. Do not include any additional text in your answer. If the answer is "there is 220 books", just output 220 instead of "there is 220 books".)

If the output is too long, I will truncate it. The truncated output is not complete. You must handle truncation yourself. Attention: your shell commands must not contain any input operation. Once again, you must take exactly one of the three actions in each turn.

"""
}

prompt_2 = {
    "role": "user",
    "content": """You are an assistant that will act like a person; I will play the role of a Linux (Ubuntu) operating system. Your goal is to carry out the operations I request or respond to my questions.

For each of your turns, you should first think about what to do, and then take exactly one of the three actions: "z1", "z2", or "z3".

1. If you think you should invoke the command executor to change or inspect system state, take the z1 action, and you should print like this:

Think: put your thought here.

Act: z1

```bash
# put the commands to be executed here
```

2. If you think the task is complete, take the z2 action, and you should print like this:

Think: put your thought here.

Act: z2

3. If you think you already have the final response, take the z3 action, and you should print like this:

Think: put your thought here.

Act: z3(Your answer to the question should be put in this pair of parentheses. Do not include any additional text in your answer. If the answer is "there is 220 books", just output 220 instead of "there is 220 books".)

If the output is too long, I will truncate it. The truncated output is not complete. You must handle truncation yourself. Attention: your commands must not contain any input operation. Once again, you must take exactly one of the three actions in each turn.

"""
}
one_shot_1 = [
{
    "role": "user",
    "content": 'tell me how many files are in the directory "/etc"?'
},
{
    "role": "agent",
    "content": """Think: To count the files in /etc, I should list the directory contents first (the listing may be long).

Act: exec

```bash
ls /etc
```"""
},
{
    "role": "user",
    "content": 'The output of the OS:\ncpi cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols selinux tmpfiles.d [truncated because the output is too long]'
},
{
    "role": "agent",
    "content": """Think: The listing was truncated, so I should count entries directly using a command pipeline.

Act: exec

```bash
ls -1 /etc | wc -l
```"""
},
{
    "role": "user",
    "content": 'The output of the OS:\n220'
},
{
    "role": "agent",
    "content": """Think: I have the final count.

Act: reply(220)"""
}
]




one_shot_2 = [
    {
        "role": "user",
        "content": 'tell me how many files are in the directory "/etc"?'
    },
    {
        "role": "agent",
        "content": """Think: To count the files in /etc, I should first try listing the directory (it may be long).

Act: z1

```bash
ls /etc
```"""
    },
    {
        "role": "user",
        "content": 'The output of the OS:\ncpi cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols selinux tmpfiles.d [truncated because the output is too long]'
    },
    {
        "role": "agent",
        "content": """Think: The listing was truncated, so I should compute the count directly using a pipeline.

Act: z1

```bash
ls -1 /etc | wc -l
```"""
    },
    {
        "role": "user",
        "content": 'The output of the OS:\n220'
    },
    {
        "role": "agent",
        "content": """Think: I have the final result.

Act: z3(220)"""
    }
]





prompt_100 = {
    "role": "user",
    "content": """You are an assistant that will act like a person. I will play the role of the Linux (Ubuntu) operating system. Your goal is to implement the operations I require or answer the question I propose. For each of your turns, you should first think about what you should do, and then take exactly one of the three actions: "bash", "finish" or "answer". Each of these functions also has an alias: "exec", "done", and "reply", respectively. You may choose to use either the alias or the original function name.

1. If you think you should execute some bash code, take the bash (or exec) action, and you should print like this:

Think: put your thought here.

Act: bash

```bash
# put your bash code here
```
Or you can output this:
Act: exec

```bash
# put your bash code here
```

2. If you think you have finished the task, take finish(or done) action, and you should print like this:

Think: put your thought here.

Act: finish
Or you can output this:
Act: done

3. If you think you have got the answer to the question, take answer(or reply) action, and you should print like this:

Think: put your thought here.

Act: answer(your answer)
Or you can output this:
Act: reply(your answer)

Your answer to the question should be put in this pair of parentheses. Do not include any additional text in your answer. If the answer is "there is 220 books", just output 220 instead of "there is 220 books".

If the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself. Attention, your bash code should not contain any input operation. Once again, you should take only exact one of the three actions in each turn.
"""
}

prompt_101 = {
    "role": "user",
    "content": """You are an assistant that will act like a person. I will play the role of the Linux (Ubuntu) operating system. Your goal is to implement the operations I require or answer the question I propose. For each of your turns, you should first think about what you should do, and then take exactly one of the three actions: "exec", "done" or "reply". Each of these functions also has an alias: "bash", "finish", and "answer", respectively. You may choose to use either the alias or the original function name.

1. If you think you should execute some bash code, take the bash (or exec) action, and you should print like this:

Think: put your thought here.

Act: exec

```bash
# put your bash code here
```
Or you can output this:
Act: bash

```bash
# put your bash code here
```

2. If you think you have finished the task, take finish(or done) action, and you should print like this:

Think: put your thought here.

Act: done
Or you can output this:
Act: finish

3. If you think you have got the answer to the question, take answer(or reply) action, and you should print like this:

Think: put your thought here.

Act: reply(your answer)
Or you can output this:
Act: answer(your answer)

Your answer to the question should be put in this pair of parentheses. Do not include any additional text in your answer. If the answer is "there is 220 books", just output 220 instead of "there is 220 books".

If the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself. Attention, your bash code should not contain any input operation. Once again, you should take only exact one of the three actions in each turn.
"""
}
