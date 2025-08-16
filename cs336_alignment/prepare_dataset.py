import json
import os
import datasets
from datasets import Dataset, DatasetDict
from math_verify import parse, verify


DS_PATH = './data/tulu_math/tulu'
ds = datasets.load_from_disk(DS_PATH)
print(ds)

# def generate_data(data_path: str):
#     with open(data_path, "r", encoding="utf-8") as f:
#         for line in f:
#             data = json.loads(line)
#             yield {
#                 "problem": data["problem"],
#                 "answer": data["answer"],
#             }


# def main():
#     trainset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("prm800k", "math_splits", "train.jsonl")})
#     testset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("prm800k", "math_splits", "test.jsonl")})
#     dataset = DatasetDict({"train": trainset, "test": testset})
#     # dataset.push_to_hub("hiyouga/math12k")


# if __name__ == "__main__":
#     main()