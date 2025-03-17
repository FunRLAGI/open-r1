import os
import json
import glob
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm

# 定义路径
QWEN_RESULTS_DIR = "./qwen_results"
EVALUATION_RESULTS_DIR = "./evaluation_results_multi"
OUTPUT_FILE = "./OpenThoughts-114k-Qwen2.5-7B-Instruct-evaluated-general-4.9k/integrated_dataset.parquet"

# 加载原始数据集
print("加载原始数据集...")
dataset = load_dataset("./OpenThoughts-114k/data", split="train")
print(f"原始数据集大小: {len(dataset)}")
print(f"原始数据集示例: {dataset[0].keys()}")

# 创建索引字典，用于快速查找
dataset_dict = {i: item for i, item in enumerate(dataset)}
print(f"创建索引字典完成")

# 加载qwen结果
print("加载qwen结果...")
qwen_results = {}
qwen_files = glob.glob(os.path.join(QWEN_RESULTS_DIR, "results_batch_*.json"))

for file_path in tqdm(qwen_files, desc="处理qwen结果文件"):
    with open(file_path, 'r', encoding='utf-8') as f:
        batch_results = json.load(f)
        
    # 计算当前批次的起始索引
    batch_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
    start_idx = batch_num * 100  # 每个批次100条数据
    
    for i, result in enumerate(batch_results):
        item_idx = start_idx + i
        if item_idx < len(dataset):
            qwen_results[item_idx] = {
                "qwen_solution": result.get("qwen_solution"),
                "qwen_solution_w_reasoning": result.get("qwen_solution_w_reasoning")
            }

print(f"加载了 {len(qwen_results)} 条qwen结果")

# 加载评估结果
print("加载评估结果...")
evaluation_results = {}
evaluation_files = glob.glob(os.path.join(EVALUATION_RESULTS_DIR, "evaluation_batch_*.json"))

for file_path in tqdm(evaluation_files, desc="处理评估结果文件"):
    with open(file_path, 'r', encoding='utf-8') as f:
        batch_results = json.load(f)
    
    for result in batch_results:
        if "item_index" in result:
            item_idx = result["item_index"]
            evaluation_results[item_idx] = {
                "evaluation": result.get("evaluation")
            }

print(f"加载了 {len(evaluation_results)} 条评估结果")

# 整合数据
print("整合数据...")
integrated_data = []

for idx in tqdm(range(len(dataset)), desc="整合数据"):
    if idx in qwen_results and idx in evaluation_results:
        item = dataset_dict[idx]
        
        # 创建新的数据项
        new_item = item
        new_item["qwen_solution"] = qwen_results[idx]["qwen_solution"]
        new_item["qwen_solution_w_reasoning"] = qwen_results[idx]["qwen_solution_w_reasoning"]
        new_item["evaluation"] = evaluation_results[idx]["evaluation"]
        
        integrated_data.append(new_item)

print(f"整合了 {len(integrated_data)} 条数据")
print(f"整合后数据集示例: {integrated_data[0].keys()}")

# 转换为DataFrame并保存
df = pd.DataFrame(integrated_data)
df.to_parquet(OUTPUT_FILE)
print(f"已保存整合后的数据集到 {OUTPUT_FILE}")

# 输出一些统计信息
print("\n数据集统计信息:")
print(f"总数据条数: {len(integrated_data)}")

# 按领域统计
if "domain" in df.columns:
    domain_counts = df["domain"].value_counts()
    print("\n按领域统计:")
    for domain, count in domain_counts.items():
        print(f"{domain}: {count}条")

# 按评分统计
if "evaluation" in df.columns and not df["evaluation"].isna().all():
    # 提取评分
    scores = []
    for eval_data in df["evaluation"]:
        if isinstance(eval_data, dict) and "score" in eval_data:
            scores.append(eval_data["score"])
    
    if scores:
        score_counts = pd.Series(scores).value_counts().sort_index()
        print("\n按评分统计:")
        for score, count in score_counts.items():
            print(f"评分 {score}: {count}条")
        print(f"平均评分: {sum(scores)/len(scores):.2f}")
