import os
import time
import json
import asyncio
import random
from datasets import load_dataset
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import numpy as np

# 加载配置文件
CONFIG_PATH = "./config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# 指定使用的模型
MODEL_NAME = "Qwen2.5-7B-Instruct"

# 从配置文件中获取API配置和模型配置
def get_api_configs_for_model(config, model_name):
    """根据模型名称获取可用的API配置"""
    api_configs = []
    model_config = config["model_configs"].get(model_name, {})
    providers_config = model_config.get("providers", {})
    
    # 遍历所有可用的提供商
    for provider_name, provider_config in providers_config.items():
        # 获取该提供商的模型名称和限制
        model_name_for_provider = provider_config.get("model_name")
        rpm_limit = provider_config.get("rpm_limit")
        tpm_limit = provider_config.get("tpm_limit")
        
        # 查找该提供商的API配置
        for api_config in config["api_configs"]:
            if api_config["provider"] == provider_name:
                # 创建新的API配置，包含模型信息
                new_config = api_config.copy()
                new_config["model_name"] = model_name_for_provider
                new_config["rpm_limit"] = rpm_limit
                new_config["tpm_limit"] = tpm_limit
                api_configs.append(new_config)
    
    return api_configs

# 获取指定模型的API配置
API_CONFIGS = get_api_configs_for_model(CONFIG, MODEL_NAME)
print(f"为模型 {MODEL_NAME} 找到 {len(API_CONFIGS)} 个可用API配置")

# 创建客户端池
class ClientPool:
    def __init__(self, api_configs):
        self.sync_clients = []
        self.async_clients = []
        self.client_stats = []  # 用于记录每个客户端的使用情况
        self.api_configs = api_configs  # 存储API配置信息
        
        for config in api_configs:
            sync_client = OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"]
            )
            async_client = AsyncOpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"]
            )
            
            self.sync_clients.append(sync_client)
            self.async_clients.append(async_client)
            self.client_stats.append({
                "requests": 0,
                "tokens": 0,
                "errors": 0,
                "last_used": 0,
                "rpm_limit": config.get("rpm_limit", 0),
                "tpm_limit": config.get("tpm_limit", 0)
            })
    
    def get_sync_client(self):
        """获取同步客户端，使用简单的负载均衡策略"""
        if not self.sync_clients:
            raise ValueError("没有可用的API客户端")
        
        # 选择请求数最少的客户端
        client_idx = min(range(len(self.client_stats)), 
                         key=lambda i: self.client_stats[i]["requests"])
        
        # 更新统计信息
        self.client_stats[client_idx]["requests"] += 1
        self.client_stats[client_idx]["last_used"] = time.time()
        
        return self.sync_clients[client_idx], client_idx
    
    async def get_async_client(self):
        """获取异步客户端，使用考虑多因素的负载均衡策略"""
        if not self.async_clients:
            raise ValueError("没有可用的API客户端")
        
        # 计算每个客户端的权重分数（考虑请求数、错误率、最后使用时间和API限制）
        current_time = time.time()
        scores = []
        
        # 获取最大rpm和tpm限制，用于归一化
        max_rpm = max(stats["rpm_limit"] for stats in self.client_stats) if any(stats["rpm_limit"] > 0 for stats in self.client_stats) else 1
        max_tpm = max(stats["tpm_limit"] for stats in self.client_stats) if any(stats["tpm_limit"] > 0 for stats in self.client_stats) else 1
        
        for i, stats in enumerate(self.client_stats):
            # 计算时间因子（越久未使用分数越高）
            time_factor = min(10, current_time - stats["last_used"]) / 10
            
            # 计算请求负载因子（请求越少分数越高）
            if max(s["requests"] for s in self.client_stats) > 0:
                request_factor = 1 - (stats["requests"] / max(s["requests"] for s in self.client_stats))
            else:
                request_factor = 1
            
            # 计算错误因子（错误越少分数越高）
            error_rate = stats["errors"] / max(1, stats["requests"])
            error_factor = 1 - min(1, error_rate * 10)
            
            # 计算API限制因子（限制越高分数越高）
            rpm_factor = stats["rpm_limit"] / max_rpm if max_rpm > 0 else 0
            tpm_factor = stats["tpm_limit"] / max_tpm if max_tpm > 0 else 0
            limit_factor = (rpm_factor + tpm_factor) / 2
            
            # 综合分数（增加API限制因子的权重）
            score = (0.3 * time_factor) + (0.3 * request_factor) + (0.1 * error_factor) + (0.3 * limit_factor)
            scores.append(score)
        
        # 选择分数最高的客户端，但加入一些随机性以避免所有请求都集中到一个客户端
        if random.random() < 0.8:  # 80%的概率选择最佳客户端
            client_idx = scores.index(max(scores))
        else:  # 20%的概率随机选择，但权重仍然基于分数
            total = sum(scores)
            if total > 0:
                weights = [s/total for s in scores]
                client_idx = random.choices(range(len(scores)), weights=weights)[0]
            else:
                client_idx = random.randint(0, len(scores)-1)
        
        # 更新统计信息
        self.client_stats[client_idx]["requests"] += 1
        self.client_stats[client_idx]["last_used"] = current_time
        
        return self.async_clients[client_idx], client_idx
    
    def update_stats(self, client_idx, tokens=0, error=False):
        """更新客户端使用统计"""
        if 0 <= client_idx < len(self.client_stats):
            self.client_stats[client_idx]["tokens"] += tokens
            if error:
                self.client_stats[client_idx]["errors"] += 1
    
    def get_stats(self):
        """获取所有客户端的使用统计"""
        return self.client_stats

# 创建客户端池
client_pool = ClientPool(API_CONFIGS)

# 从配置文件中获取数据集路径和结果目录
dataset_config = CONFIG.get("dataset", {"path": "openthoughts/OpenThoughts-114k", "split": "train", "start_index": 0})
dataset = load_dataset(dataset_config["path"], split=dataset_config.get("split", "train"))
print(f"数据集大小: {len(dataset)}")
print(f"数据集字段: {dataset[0].keys()}")

# 创建结果存储目录
results_dir = CONFIG.get("results_dir", f"./qwen_results_{MODEL_NAME.lower().replace('.', '_').replace('-', '_')}")
os.makedirs(results_dir, exist_ok=True)

# 定义提示模板
def get_prompt_without_reasoning(problem):
    return f"""请解决以下问题，并一步一步地展示你的推理过程：

{problem}

请详细解释你的思考过程，然后给出最终答案。"""

def get_prompt_with_reasoning(problem, reasoning):
    return f"""请解决以下问题：

{problem}

以下是一些推理思路，请参考并基于此给出你的解答：
{reasoning}

请给出你的最终答案。"""

# 调用API的函数
async def query_qwen_async(prompt, model=None):
    # 获取客户端
    client, client_idx = await client_pool.get_async_client()
    
    try:
        # 构建消息
        messages = [
            {"role": "system", "content": "你是一个擅长解决问题的AI助手，请一步一步思考并给出答案。"},
            {"role": "user", "content": prompt}
        ]
        
        # 获取模型名称（从API配置中获取对应客户端的模型名称）
        model_name = client_pool.api_configs[client_idx]["model_name"] if model is None else model
        
        # 异步调用API
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=4096
        )
        
        content = response.choices[0].message.content
        
        # 直接从API响应中获取token信息
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        # 更新客户端统计信息
        client_pool.update_stats(client_idx, tokens=total_tokens)
        
        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "client_idx": client_idx
        }
    except Exception as e:
        print(f"API调用错误 (客户端 {client_idx}): {e}")
        # 更新错误统计
        client_pool.update_stats(client_idx, error=True)
        
        # 等待一段时间再重试
        await asyncio.sleep(5)
        
        return {
            "content": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": str(e),
            "client_idx": client_idx
        }

# 处理单个数据项
async def process_item(item, item_index, total_items):
    # 提取必要字段
    problem = item['problem']
    deepseek_reasoning = item.get('deepseek_reasoning', '')
    deepseek_solution = item.get('deepseek_solution', '')
    domain = item.get('domain', '')
    
    # 准备提示
    prompt1 = get_prompt_without_reasoning(problem)
    prompt2 = get_prompt_with_reasoning(problem, deepseek_reasoning)
    
    # 并行发送两个请求
    print(f"处理第 {item_index+1}/{total_items} 条数据 - 发送两个请求")
    results = await asyncio.gather(
        query_qwen_async(prompt1),
        query_qwen_async(prompt2)
    )
    
    qwen_solution_result = results[0]
    qwen_solution_w_reasoning_result = results[1]
    
    # 保存结果
    result = {
        "problem": problem,
        "deepseek_reasoning": deepseek_reasoning,
        "deepseek_solution": deepseek_solution,
        "qwen_solution": qwen_solution_result["content"],
        "qwen_solution_w_reasoning": qwen_solution_w_reasoning_result["content"],
        "domain": domain,
        "token_stats": {
            "request1": {
                "input_tokens": qwen_solution_result["input_tokens"],
                "output_tokens": qwen_solution_result["output_tokens"],
                "total_tokens": qwen_solution_result["total_tokens"],
                "client_idx": qwen_solution_result.get("client_idx")
            },
            "request2": {
                "input_tokens": qwen_solution_w_reasoning_result["input_tokens"],
                "output_tokens": qwen_solution_w_reasoning_result["output_tokens"],
                "total_tokens": qwen_solution_w_reasoning_result["total_tokens"],
                "client_idx": qwen_solution_w_reasoning_result.get("client_idx")
            },
            "combined_total_tokens": qwen_solution_result["total_tokens"] + qwen_solution_w_reasoning_result["total_tokens"]
        }
    }
    
    return result

# 主处理函数
async def process_dataset(batch_size=100, concurrent_requests=5):
    results = []
    token_stats = []
    
    # 计算API限制（考虑多个提供商的不同限制）
    rpm_limits = [config.get("rpm_limit", 0) for config in API_CONFIGS]
    tpm_limits = [config.get("tpm_limit", 0) for config in API_CONFIGS]
    
    # 使用加权平均计算有效限制
    # 对于每个客户端，计算其限制占总限制的比例，然后根据这个比例分配请求
    total_rpm_weight = sum(rpm for rpm in rpm_limits if rpm > 0)
    total_tpm_weight = sum(tpm for tpm in tpm_limits if tpm > 0)
    
    # 计算有效的RPM和TPM限制（考虑不同客户端的限制差异）
    rpm_limit = max(rpm_limits) if total_rpm_weight > 0 else 0
    tpm_limit = max(tpm_limits) if total_tpm_weight > 0 else 0
    
    # 每个请求的平均token估计值（初始值，将根据实际使用情况调整）
    avg_tokens_per_request = 1000
    
    # 计算安全的并发请求数 (考虑多账号)
    account_count = len(API_CONFIGS)
    safe_concurrent_requests = min(
        concurrent_requests,
        (rpm_limit * account_count) // 120,  # 考虑到每个数据项有两个请求，并留有安全余量
        (tpm_limit * account_count) // (avg_tokens_per_request * 120)  # 同样考虑token限制
    )
    
    print(f"使用账号数: {account_count}")
    print(f"使用并发请求数: {safe_concurrent_requests}")
    
    try:
        # 创建任务队列
        tasks = []
        for i, item in enumerate(dataset):
            if i < CONFIG["dataset"]["start_index"]:
                continue
            tasks.append(process_item(item, i, len(dataset)))
            
            # 当任务数达到并发限制或处理到最后一个项目时执行
            if len(tasks) >= safe_concurrent_requests or i == len(dataset) - 1:
                # 并行执行任务
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                
                # 收集token统计信息
                batch_tokens = [r["token_stats"]["combined_total_tokens"] for r in batch_results]
                token_stats.extend(batch_tokens)
                
                # 计算平均token使用量并调整并发请求数
                if token_stats:
                    avg_tokens_per_request = np.mean(token_stats) / 2  # 每个数据项有两个请求
                    new_safe_concurrent_requests = min(
                        concurrent_requests,
                        (rpm_limit * account_count) // 120,
                        (tpm_limit * account_count) // (int(avg_tokens_per_request) * 120)
                    )
                    
                    if new_safe_concurrent_requests != safe_concurrent_requests:
                        safe_concurrent_requests = new_safe_concurrent_requests
                        print(f"调整并发请求数为: {safe_concurrent_requests}")
                
                # 保存批次结果
                if len(results) >= batch_size or i == len(dataset) - 1:
                    batch_index = i // batch_size
                    batch_file = os.path.join(results_dir, f"results_batch_{batch_index}.json")
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"已保存批次结果到 {batch_file}")
                    
                    # 打印token使用统计
                    if token_stats:
                        print(f"Token使用统计:")
                        print(f"  平均每条数据总token: {np.mean(token_stats):.2f}")
                        print(f"  最小token: {np.min(token_stats)}")
                        print(f"  最大token: {np.max(token_stats)}")
                        print(f"  中位数token: {np.median(token_stats):.2f}")
                    
                    # 打印客户端使用统计
                    client_stats = client_pool.get_stats()
                    print(f"客户端使用统计:")
                    for i, stats in enumerate(client_stats):
                        print(f"  客户端 {i}: 请求数={stats['requests']}, token数={stats['tokens']}, 错误数={stats['errors']}")
                    
                    results = []  # 清空结果列表，准备下一批次
                
                # 清空任务列表
                tasks = []
                
                # 短暂暂停以避免超过API限制
                await asyncio.sleep(0.5)
    
    except KeyboardInterrupt:
        print("处理被用户中断")
        # 保存已处理的结果
        if results:
            interrupt_file = os.path.join(results_dir, f"results_interrupted.json")
            with open(interrupt_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"已保存中断时的结果到 {interrupt_file}")
            
            # 打印token使用统计
            if token_stats:
                print(f"Token使用统计:")
                print(f"  平均每条数据总token: {np.mean(token_stats):.2f}")
                print(f"  最小token: {np.min(token_stats)}")
                print(f"  最大token: {np.max(token_stats)}")
                print(f"  中位数token: {np.median(token_stats):.2f}")
            
            # 打印客户端使用统计
            client_stats = client_pool.get_stats()
            print(f"客户端使用统计:")
            for i, stats in enumerate(client_stats):
                print(f"  客户端 {i}: 请求数={stats['requests']}, token数={stats['tokens']}, 错误数={stats['errors']}")
    
    # 保存token使用统计
    if token_stats:
        stats_file = os.path.join(results_dir, "token_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "average_tokens_per_item": float(np.mean(token_stats)),
                "min_tokens": int(np.min(token_stats)),
                "max_tokens": int(np.max(token_stats)),
                "median_tokens": float(np.median(token_stats)),
                "total_tokens": int(np.sum(token_stats)),
                "item_count": len(token_stats),
                "detailed_stats": token_stats
            }, f, ensure_ascii=False, indent=2)
        print(f"已保存token统计信息到 {stats_file}")
    
    # 保存客户端使用统计
    client_stats_file = os.path.join(results_dir, "client_stats.json")
    with open(client_stats_file, 'w', encoding='utf-8') as f:
        json.dump(client_pool.get_stats(), f, ensure_ascii=False, indent=2)
    print(f"已保存客户端统计信息到 {client_stats_file}")

# 合并所有批次结果
def merge_results():
    all_results = []
    for filename in os.listdir(results_dir):
        if filename.startswith("results_batch_") and filename.endswith(".json"):
            file_path = os.path.join(results_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
    
    # 保存合并结果
    merged_file = os.path.join(results_dir, "all_results.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"已合并所有结果到 {merged_file}")

# 主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='使用多账号并行处理OpenThoughts数据集')
    parser.add_argument('--batch_size', type=int, default=100, help='每批处理的数据量')
    parser.add_argument('--concurrent', type=int, default=10, help='并发请求数')
    parser.add_argument('--merge', action='store_true', help='是否合并所有批次结果')
    
    args = parser.parse_args()
    
    if args.merge:
        merge_results()
    else:
        # 运行主处理函数
        asyncio.run(process_dataset(args.batch_size, args.concurrent))
        print("处理完成！")
