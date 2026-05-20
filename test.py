# (1,1),(1,2)-(1,5)*2 = 9
# (2,2),(2,3)-(2,4)*2 = 5
# (3,3) = 1

# 15/36 = 5/12

# p = 7/12 + 5/12*5/12*p

# p=12*7/119 = 84/119 = 12/17

# 7/12 + 5/12*5/12*7/12 + ...

# 7/12*(1+25/144+.^2+...) = 7/12*(1-)

# a+b+c = 2025

# c(2,2024) = 2024*2023/2

# 2,2,2

# a<=b<=c,

# 1,1,2023

# 1012,1012,1

# 1012 zu

# 1012*3-2

# (2024*2023/2-1012*3+2)/6+1012

import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from distilabel.llms import OpenAILLM
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import TextGeneration, EvolQuality, UltraFeedback
from tqdm import tqdm
from utils.utils import find_files

TMP_PATH        = "/archive/share/cql/aaa/tmp"
TRAIN_SUBSET    = 5
DATA_FILES      = find_files(["rl"],"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reward/data")
MODEL_A_PATH    = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen3-8B-Instruct"
MODEL_B_PATH    = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B-Instruct"

ds = load_dataset("parquet", data_files=DATA_FILES,
                        split="train", cache_dir=TMP_PATH)
if TRAIN_SUBSET > 0:
    ds = ds.select(range(TRAIN_SUBSET))
questions = ds["question"]

llm_a = TransformersLLM(model=MODEL_A_PATH)
llm_a.load()
llm_b = TransformersLLM(model=MODEL_B_PATH)
llm_b.load()

judge_llm = OpenAILLM(
    model="deepseek-chat",
    base_url=r"https://api.deepseek.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", "")
)
judge_llm.load()

evol_quality   = EvolQuality(llm=judge_llm, num_evolutions=1) 
evol_quality.load()

ultrafeedback  = UltraFeedback(llm=judge_llm)           
ultrafeedback.load()

records = []

def chat(msg): 
    return [{"role": "user", "content": msg}]

GEN_KWARGS = dict(max_new_tokens=256, temperature=0.7)

for q in tqdm(questions, desc="Generating / Evolving / Scoring"):
    # (i) 两个模型独立生成
    gen_a = llm_a.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]
    gen_b = llm_b.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]

    # (ii) EvolQuality 迭代 1 次
    evo_a = evol_quality.process([{"instruction": q, "response": gen_a}])
    evo_b = evol_quality.process([{"instruction": q, "response": gen_b}])
    evo_a = next(evo_a)[0]["evolved_response"]
    evo_b = next(evo_b)[0]["evolved_response"]

    # (iii) UltraFeedback 打分
    feedback = ultrafeedback.process([{
        "instruction": q,
        "generations": [evo_a, evo_b]
    }])
    feedback = next(feedback)
    ratings     = feedback[0]["ratings"]      # e.g. [4, 5]
    rationales  = feedback[0]["rationales"]   # 逐条解释

    records.append({
        "question"          : q,
        "model_a_raw"       : gen_a,
        "model_b_raw"       : gen_b,
        "model_a_evolved"   : evo_a,
        "model_b_evolved"   : evo_b,
        "ratings"           : ratings,
        "rationales"        : rationales
    })

df = pd.DataFrame(records)
df.to_parquet("synthetic_qa_with_scores.parquet", index=False)
print(df.head())