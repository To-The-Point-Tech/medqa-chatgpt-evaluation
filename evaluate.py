import json
from argparse import ArgumentParser
import os
import tiktoken
from os.path import exists, join
from time import sleep

from transformers import pipeline, AutoTokenizer
import torch

from utils.tothepoint_io import ttp_retrieve, ttp_renew_token, ttp_validate_token

import openai
from tqdm import tqdm

import tiktoken


enc = tiktoken.get_encoding("cl100k_base")


def _merge_with_index(answers):
    return "\n".join([f"{i}) {answer}" for i, answer in enumerate(answers)])

def generate_message(question: str, possible_answers: list[str], additional_context: dict = None):
    possible_answers_example_prompt = _merge_with_index([
        "Decreased ability to copy", "Decreased execution", "Deficit of expression by gesture", "Deficit of fluent speech"
    ])
    message = [{
        "role": "system",
        "content": f"""
        Answer the question provided you by the user. For the question you will have few possible answers.
        Make sure to answer the question accurately, as any mistakes could have serious consequences for the patient.
        The response options are: {', '.join([str(i) for i in range(len(possible_answers))])}. Other options may be partially correct, but there is only ONE BEST answer. For example:
        For the question 'Hypomimia is ?', possible answers are: {possible_answers_example_prompt}
        We expect you to return just a single number: 2. Additional information may be provided at times.
        """,
    }]

    possible_answers_prompt = _merge_with_index(possible_answers)
    if additional_context is not None:
        message += [{
            "role": "assistant",
            "content": f"Here the knowledge that I have. May it could help me to answer the question",
        }]
        for i, context in additional_context.items():
            message += [{
                "role": "assistant",
                "content": f"Few facts about answer {i}) {possible_answers[i]}:\n" + "\n - ".join([c[:1000] for c in context]),
            }]
    message += [
        {
            "role": "user", 
            "content": f"Question: {question}\n\n" + 
                    f"Possible answers: \n{possible_answers_prompt}\n\n" + 
                    f"Choose only one of the answers, and return the number of {', '.join([str(i) for i in range(len(possible_answers))])} correspomdimg to the correct snswer." +
                    "No need to add any natural language explanation, only the number of the right answer"
        }
    ]
    return message

def to_llama_format(message: list):
    system_message, *other_messages, user_message = message
    system_promt = system_message["content"]
    user_prompt = user_message["content"]
    return f"""<s>[INST] <<SYS>>
    {system_promt}
    <</SYS>>
    {user_prompt} [/INST]
    """ 


def run_evaluation(model: str, task: str, split: str):
    data = []
    with open(join("data", task, f"{split}.json"), "r") as f:
        for line in f:
            data.append(json.loads(line))

    if "gpt4" in model:
        openai_model_name = "gpt-4"
    elif "chatgpt" in model:
        openai_model_name = "gpt-3.5-turbo"
    elif "llama" in model:
        tokenizer = AutoTokenizer.from_pretrained(f"models/{model}-hf")
        pipe = pipeline(
            "text-generation",
            model=f"models/{model}-hf",
            torch_dtype=torch.float16,
            device_map="auto",
            device=0
        )
    else:
        raise ValueError(f"Unknown model {model}")
    
    done = []
    if exists(join("data", task, f"{model}-{split}.json")):
        with open(join("data", task, f"{model}-{split}.json"), "r") as f:
            for line in f:
                jsn = json.loads(line)
                done.append(jsn["id"])

    token = ""
    with open(join("data", task, f"{model}-{split}.json"), "a") as f:
        for line in tqdm(data):
            id_ = line["id"]
            if id_ in done:
                continue

            question = line["sent1"]
            n_classes = len([k for k in line if "ending" in k])
            answers = [line[f'ending{i}'] for i in range(n_classes)]

            try:
                if "gpt" in model:
                    if model == "tothepoint-chatgpt":
                        if not ttp_validate_token(token=token):
                            token = ttp_renew_token()
                        try:
                            context = {
                                i: [e['summary'] for e in ttp_retrieve(answer, token=token, k=3)] 
                                for i, answer in enumerate(answers)
                            }
                        except:
                            continue
                    elif model == "chatgpt":
                        context = None
                    elif model == "gpt4":
                        context = None
                    else:
                        raise ValueError(f"Unsupported model {model}")
                    message = generate_message(question=question, possible_answers=answers, additional_context=context)
                    response = openai.ChatCompletion.create(
                        model=openai_model_name, 
                        messages=message, 
                        temperature=0.2,
                        n=1, 
                        max_tokens=3,
                    )
                    class_ = [e["message"]["content"].strip() for e in response["choices"]][0]
                    sleep(2)
                elif "llama" in model:
                    context = None
                    message = generate_message(question=question, possible_answers=answers, additional_context=context)
                    message = to_llama_format(message)
                    sequences = pipe(
                        message,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        max_length=4096,
                        temperature=0.2,
                        max_new_tokens=3,
                        return_full_text=False,
                    )
                    class_ = sequences[0]["generated_text"]
                else:
                    raise ValueError(f"Unsupported model {model}")
                print(class_)
            except:
                print("Fail")
                continue
            outputs = {
                "answer": class_,
                "id": id_,
            }
            f.write(json.dumps(outputs) + "\n")

    id2answer = {}
    with open(join("data", task, f"{model}-{split}.json"), "r") as f:
        for line in f:
            jsn = json.loads(line)
            try:
                answer = int(jsn["answer"].replace(":", "").split(" ")[0])
                id2answer[jsn["id"]] = answer
            except:
                continue

    correct, total = 0, 0
    for line in data:
        id_, label = line["id"], line["label"]
        if id_ in id2answer:
            answer = id2answer[id_]
            if label == answer:
                correct += 1
            total += 1
    acc = round(100 * correct / total, 2)
    print(f"Acctuacy of model {model} on {task} is {acc}%")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", choices=[
        "chatgpt", "gpt4", 
        "llama-2-7b", "llama-2-13b",
        "llama-2-7b-chat", "llama-2-13b-chat",
        "tothepoint-chatgpt",
    ])
    parser.add_argument("--task", choices=["medmcqa", "medqa", "mmlu", "medcoding"])
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    args_ = parser.parse_args()

    run_evaluation(model=args_.model, task=args_.task, split=args_.split)