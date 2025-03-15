import os
import json
import time
import argparse
import asyncio
import random
from openai import OpenAI, AsyncOpenAI
import numpy as np
from tqdm import tqdm

# 多账号API密钥配置
API_CONFIGS = [
    # 添加更多账号配置，格式如下：
    {
        "api_key": "sk-zcjkxltmsursqcfqareybswtborbpsdinrnexkhmtxbccotg",
        "base_url": "https://api.siliconflow.cn/v1"
    },
    # 添加更多账号配置，格式如下：
    {
        "api_key": "sk-sodcaviuawnqrrhnzixrmjhewcfjkwcsolccpakxugtwjrwb",
        "base_url": "https://api.siliconflow.cn/v1"
    },
    {
        "api_key": "sk-qkobyqwwlsvlzrsbbwksjkxrkmzzoihbdafgpoduvorwfgdz",
        "base_url": "https://api.siliconflow.cn/v1"
    },
    {
        "api_key": "sk-fispkvthpyvwcamsgblcgtkuzkjydspjlualwntyykzczizt",
        "base_url": "https://api.siliconflow.cn/v1"
    },
]

# 创建客户端池
class ClientPool:
    def __init__(self, api_configs):
        self.sync_clients = []
        self.async_clients = []
        self.client_stats = []  # 用于记录每个客户端的使用情况
        
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
                "last_used": 0
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
        
        # 计算每个客户端的权重分数（考虑请求数、错误率和最后使用时间）
        current_time = time.time()
        scores = []
        
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
            
            # 综合分数
            score = (0.4 * time_factor) + (0.4 * request_factor) + (0.2 * error_factor)
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

# 创建结果存储目录
results_dir = "./evaluation_results_multi"
os.makedirs(results_dir, exist_ok=True)

# 定义评估提示模板
def get_evaluation_prompt(problem, deepseek_solution, qwen_solution):
    return f"""请严格评估以下两个解决方案的一致性和准确性：

问题：
{problem}

参考解决方案（DeepSeek）：
{deepseek_solution}

待评估解决方案（Qwen）：
{qwen_solution}

请分析待评估解决方案是否准确表达了参考解决方案的核心意思。评估应考虑以下几点：
1. 核心概念的覆盖程度 - 待评估解决方案是否涵盖了参考解决方案中的所有关键概念
2. 关键结论的一致性 - 待评估解决方案的结论是否与参考解决方案一致
3. 重要细节的准确性 - 待评估解决方案中的细节是否与参考解决方案中的细节一致
4. 整体观点的一致性 - 待评估解决方案的整体观点是否与参考解决方案一致

请严格执行以下标准给出评分（1-5分）：
1分：完全不一致 - 待评估解决方案与参考解决方案在核心概念、关键结论、重要细节和整体观点上存在根本性差异
2分：大部分不一致 - 待评估解决方案仅包含少量与参考解决方案一致的内容，大部分内容不一致或有误
3分：部分一致 - 待评估解决方案包含约一半与参考解决方案一致的内容，但也有明显的遗漏或错误
4分：大部分一致 - 待评估解决方案包含大部分与参考解决方案一致的内容，仅有少量遗漏或细微差异
5分：高度一致 - 待评估解决方案几乎完全涵盖参考解决方案的所有核心内容，无明显遗漏或错误

请注意：
- 避免给出中间评分（如3分）作为默认选择
- 如果解决方案之间存在明显差异，应给予较低分数（1-2分）
- 如果解决方案之间基本一致但有少量差异，应给予较高分数（4-5分）
- 只有当解决方案既有一致的部分又有不一致的部分且大致相当时，才给予3分

请详细解释您的评分理由，针对每个评估点给出具体分析。
最后，请明确回答：待评估解决方案是否准确表达了参考解决方案的意思？（是/否）

您的评分必须客观公正，避免不必要的宽容，请根据实际一致程度严格评分。"""

# 异步调用API的函数
async def query_model_async(prompt, model="deepseek-ai/DeepSeek-V3", max_retries=3):
    # 获取客户端
    client, client_idx = await client_pool.get_async_client()
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的解决方案评估专家，擅长分析和比较不同解决方案的异同点。你的评估必须客观严格，避免给出中庸的评分。当解决方案之间存在明显差异时，应给予较低分数；当解决方案高度一致时，应给予较高分数。请避免将3分作为默认选择。"}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 使用更低的温度以获得更确定性的评估
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            
            # 获取token信息
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
            print(f"API调用错误 (客户端 {client_idx}, 尝试 {attempt+1}/{max_retries}): {e}")
            # 更新错误统计
            client_pool.update_stats(client_idx, error=True)
            
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "content": f"评估失败: {str(e)}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "error": str(e),
                    "client_idx": client_idx
                }

# 解析评估结果，提取评分和是否一致的结论
def parse_evaluation(evaluation_text):
    try:
        # 尝试提取评分（1-5分）
        score = None
        for line in evaluation_text.split('\n'):
            if '评分' in line and any(str(i) in line for i in range(1, 6)):
                for i in range(1, 6):
                    if str(i) in line:
                        score = i
                        break
                if score:
                    break
        
        # 如果没有找到明确的评分，尝试从文本中推断
        if score is None:
            # 寻找可能包含评分的句子
            score_indicators = {
                1: ['1分', '一分', '完全不一致', '根本性差异', '完全不同'],
                2: ['2分', '二分', '大部分不一致', '主要不一致', '很多错误'],
                3: ['3分', '三分', '部分一致', '约一半一致', '有明显遗漏'],
                4: ['4分', '四分', '大部分一致', '基本一致', '少量遗漏'],
                5: ['5分', '五分', '高度一致', '完全一致', '几乎完全涵盖']
            }
            
            # 计算每个分数的指标出现次数
            score_counts = {i: 0 for i in range(1, 6)}
            for score_value, indicators in score_indicators.items():
                for indicator in indicators:
                    if indicator in evaluation_text:
                        score_counts[score_value] += 1
            
            # 选择出现次数最多的分数
            if any(score_counts.values()):
                max_count = max(score_counts.values())
                if max_count > 0:
                    for score_value, count in score_counts.items():
                        if count == max_count:
                            score = score_value
                            break
        
        # 尝试提取是/否结论
        is_consistent = None
        lower_text = evaluation_text.lower()
        
        # 直接寻找明确的是/否回答
        if '是/否' in lower_text or '是否' in lower_text:
            # 获取是/否问题后的文本
            after_question = lower_text.split('是/否')[-1] if '是/否' in lower_text else lower_text.split('是否')[-1]
            first_line = after_question.split('\n')[0]
            
            if '是' in first_line and '否' not in first_line:
                is_consistent = True
            elif '否' in first_line and '是' not in first_line:
                is_consistent = False
        
        # 如果没有明确找到，尝试从整体评估中推断
        if is_consistent is None:
            # 扩展指标列表
            positive_indicators = ['一致', '准确', '相符', '相同', '相似', '表达了', '涵盖', '符合', '吻合']
            negative_indicators = ['不一致', '不准确', '不相符', '不相同', '不相似', '没有表达', '遗漏', '缺失', '差异', '不符合', '不吻合']
            
            # 计算正面和负面指标的出现次数，并给予权重
            positive_count = sum(2 if indicator in lower_text.split('最终')[-1] else 1 
                               for indicator in positive_indicators if indicator in lower_text)
            negative_count = sum(2 if indicator in lower_text.split('最终')[-1] else 1 
                                for indicator in negative_indicators if indicator in lower_text)
            
            if positive_count > negative_count * 1.5:  # 正面指标显著多于负面指标
                is_consistent = True
            elif negative_count > positive_count * 1.5:  # 负面指标显著多于正面指标
                is_consistent = False
            else:  # 如果指标数量接近，根据评分决定
                if score is not None:
                    is_consistent = score >= 4  # 4分及以上认为是一致的
        
        return {
            "score": score,
            "is_consistent": is_consistent,
            "full_evaluation": evaluation_text
        }
    except Exception as e:
        print(f"解析评估结果时出错: {e}")
        return {
            "score": None,
            "is_consistent": None,
            "full_evaluation": evaluation_text,
            "parse_error": str(e)
        }

# 处理单个数据项
async def process_item(item, item_index, total_items):
    try:
        # 提取必要字段
        problem = item['problem']
        deepseek_solution = item['deepseek_solution']
        qwen_solution = item['qwen_solution']
        domain = item.get('domain', '')
        
        # 构建评估提示
        prompt = get_evaluation_prompt(problem, deepseek_solution, qwen_solution)
        
        # 调用模型进行评估
        print(f"\n评估第 {item_index + 1}/{total_items} 条数据")
        evaluation_result = await query_model_async(prompt)
        
        # 解析评估结果
        parsed_evaluation = parse_evaluation(evaluation_result["content"])
        
        # 保存结果
        result = {
            "problem": problem,
            "domain": domain,
            "evaluation": parsed_evaluation,
            "item_index": item_index,
            "token_stats": {
                "input_tokens": evaluation_result["input_tokens"],
                "output_tokens": evaluation_result["output_tokens"],
                "total_tokens": evaluation_result["total_tokens"],
                "client_idx": evaluation_result.get("client_idx")
            }
        }
        
        # 打印简短的评估摘要
        score_str = f"{parsed_evaluation['score']}分" if parsed_evaluation['score'] else "未知分数"
        consistency_str = "一致" if parsed_evaluation['is_consistent'] else "不一致" if parsed_evaluation['is_consistent'] is not None else "未知"
        print(f"评估结果: {score_str}, {consistency_str}")
        
        return result
    except Exception as e:
        print(f"处理数据项时出错: {e}")
        return {
            "error": str(e),
            "item_index": item_index
        }

# 主处理函数
async def evaluate_solutions_async(input_files, batch_size=10, concurrent_requests=5):
    all_results = []
    all_items = []
    token_stats = []
    
    # 加载所有数据
    for input_file in input_files:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                items = json.load(f)
                all_items.extend(items)
                print(f"从 {input_file} 加载了 {len(items)} 条数据")
        except Exception as e:
            print(f"加载文件 {input_file} 时出错: {e}")
    
    print(f"总共加载了 {len(all_items)} 条数据")
    
    # 计算API限制 (每个账号的限制)
    rpm_limit = 1000  # 每分钟请求数限制
    tpm_limit = 50000  # 每分钟token数限制
    
    # 每个请求的平均token估计值（初始值，将根据实际使用情况调整）
    avg_tokens_per_request = 2000
    
    # 计算安全的并发请求数 (考虑多账号)
    account_count = len(API_CONFIGS)
    safe_concurrent_requests = min(
        concurrent_requests,
        (rpm_limit * account_count) // 60,  # 考虑到每分钟的请求限制
        (tpm_limit * account_count) // (avg_tokens_per_request * 60)  # 考虑token限制
    )
    
    print(f"使用账号数: {account_count}")
    print(f"使用并发请求数: {safe_concurrent_requests}")
    
    try:
        # 批量处理数据
        for batch_idx in range(0, len(all_items), batch_size):
            batch = all_items[batch_idx:batch_idx + batch_size]
            batch_results = []
            
            # 创建任务队列
            tasks = []
            for i, item in enumerate(batch):
                tasks.append(process_item(item, batch_idx + i, len(all_items)))
                
                # 当任务数达到并发限制或处理到最后一个项目时执行
                if len(tasks) >= safe_concurrent_requests or i == len(batch) - 1:
                    # 并行执行任务
                    results = await asyncio.gather(*tasks)
                    batch_results.extend(results)
                    
                    # 收集token统计信息
                    batch_tokens = [r.get("token_stats", {}).get("total_tokens", 0) for r in results if "token_stats" in r]
                    token_stats.extend(batch_tokens)
                    
                    # 计算平均token使用量并调整并发请求数
                    if token_stats:
                        avg_tokens_per_request = np.mean(token_stats)
                        new_safe_concurrent_requests = min(
                            concurrent_requests,
                            (rpm_limit * account_count) // 60,
                            (tpm_limit * account_count) // (int(avg_tokens_per_request) * 60)
                        )
                        
                        if new_safe_concurrent_requests != safe_concurrent_requests:
                            safe_concurrent_requests = new_safe_concurrent_requests
                            print(f"调整并发请求数为: {safe_concurrent_requests}")
                    
                    # 清空任务列表
                    tasks = []
                    
                    # 短暂暂停以避免超过API限制
                    await asyncio.sleep(0.5)
            
            # 保存批次结果
            all_results.extend(batch_results)
            batch_file = os.path.join(results_dir, f"evaluation_batch_{batch_idx//batch_size}.json")
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)
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
    
    except KeyboardInterrupt:
        print("处理被用户中断")
        # 保存已处理的结果
        if all_results:
            interrupt_file = os.path.join(results_dir, f"evaluation_interrupted.json")
            with open(interrupt_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"已保存中断时的结果到 {interrupt_file}")
    
    # 生成统计报告
    generate_report(all_results)
    
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
    
    return all_results

# 生成评估统计报告
def generate_report(results):
    try:
        # 过滤掉有错误的结果
        valid_results = [r for r in results if 'evaluation' in r and 'error' not in r]
        
        # 提取评分和一致性结果
        scores = [r['evaluation']['score'] for r in valid_results if r['evaluation']['score'] is not None]
        consistency = [r['evaluation']['is_consistent'] for r in valid_results if r['evaluation']['is_consistent'] is not None]
        
        # 按领域分组
        domains = {}
        for r in valid_results:
            domain = r.get('domain', 'unknown')
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(r)
        
        # 计算统计数据
        stats = {
            "total_items": len(results),
            "valid_evaluations": len(valid_results),
            "average_score": np.mean(scores) if scores else None,
            "score_distribution": {i: scores.count(i) for i in range(1, 6)},
            "consistency_rate": sum(consistency) / len(consistency) if consistency else None,
            "domain_stats": {}
        }
        
        # 计算每个领域的统计数据
        for domain, domain_results in domains.items():
            domain_scores = [r['evaluation']['score'] for r in domain_results if r['evaluation']['score'] is not None]
            domain_consistency = [r['evaluation']['is_consistent'] for r in domain_results if r['evaluation']['is_consistent'] is not None]
            
            stats["domain_stats"][domain] = {
                "count": len(domain_results),
                "average_score": np.mean(domain_scores) if domain_scores else None,
                "score_distribution": {i: domain_scores.count(i) for i in range(1, 6)} if domain_scores else {},
                "consistency_rate": sum(domain_consistency) / len(domain_consistency) if domain_consistency else None
            }
        
        # 检测评分偏差
        bias_warnings = []
        if scores:
            # 检查是否有过多的中间评分（3分）
            middle_score_rate = scores.count(3) / len(scores) if scores else 0
            if middle_score_rate > 0.4:  # 如果超过40%的评分为3分，发出警告
                bias_warnings.append(f"警告：有{middle_score_rate*100:.1f}%的评分为3分，可能存在评分偏向中间值的问题")
            
            # 检查评分分布是否过于集中
            most_common_score = max(range(1, 6), key=lambda i: scores.count(i))
            most_common_rate = scores.count(most_common_score) / len(scores)
            if most_common_rate > 0.6:  # 如果超过60%的评分为同一个分数，发出警告
                bias_warnings.append(f"警告：有{most_common_rate*100:.1f}%的评分为{most_common_score}分，评分分布过于集中")
            
            # 检查评分与一致性结论是否匹配
            mismatch_count = sum(1 for i, score in enumerate(scores) if 
                              i < len(consistency) and consistency[i] is not None and 
                              ((score >= 4 and not consistency[i]) or (score <= 2 and consistency[i])))
            if mismatch_count > 0:
                bias_warnings.append(f"警告：有{mismatch_count}个评分与一致性结论不匹配，可能需要人工审核")
        
        # 将警告添加到统计数据中
        stats["bias_warnings"] = bias_warnings
        
        # 保存统计报告
        report_file = os.path.join(results_dir, "evaluation_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"已保存评估报告到 {report_file}")
        
        # 打印摘要
        print("\n评估摘要:")
        print(f"总数据项: {stats['total_items']}")
        print(f"有效评估: {stats['valid_evaluations']}")
        if stats['average_score']:
            print(f"平均评分: {stats['average_score']:.2f}/5.0")
        if stats['consistency_rate'] is not None:
            print(f"一致性比率: {stats['consistency_rate']*100:.2f}%")
        print("\n评分分布:")
        for score, count in stats['score_distribution'].items():
            print(f"  {score}分: {count} 项 ({count/len(scores)*100:.2f}%)")
        
        # 打印警告
        if bias_warnings:
            print("\n评分偏差警告:")
            for warning in bias_warnings:
                print(f"  {warning}")
        
        # 生成详细的HTML报告
        generate_html_report(valid_results, stats)
        
    except Exception as e:
        print(f"生成报告时出错: {e}")

# 生成HTML格式的详细报告
def generate_html_report(results, stats):
    try:
        # 使用字符串拼接而不是直接在f-string中使用HTML标签
        html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>解决方案评估报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .chart {{ margin: 20px 0; height: 300px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .consistent {{ color: green; }}
        .inconsistent {{ color: red; }}
        .unknown {{ color: gray; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>解决方案评估报告</h1>
    
    <div class="summary">
        <h2>评估摘要</h2>
        <p>总数据项: {stats['total_items']}</p>
        <p>有效评估: {stats['valid_evaluations']}</p>
        <p>平均评分: {stats['average_score']:.2f if stats['average_score'] else 'N/A'}/5.0</p>
        <p>一致性比率: {stats['consistency_rate']*100:.2f if stats['consistency_rate'] is not None else 'N/A'}%</p>
    </div>
    
    <h2>评分分布</h2>
    <div class="chart">
        <canvas id="scoreChart"></canvas>
    </div>
    
    <h2>领域统计</h2>
    <table>
        <tr>
            <th>领域</th>
            <th>数据项数量</th>
            <th>平均评分</th>
            <th>一致性比率</th>
        </tr>
        {{
            ''.join([f"""
        <tr>
            <td>{domain}</td>
            <td>{stats['domain_stats'][domain]['count']}</td>
            <td>{stats['domain_stats'][domain]['average_score']:.2f if stats['domain_stats'][domain]['average_score'] else 'N/A'}</td>
            <td>{stats['domain_stats'][domain]['consistency_rate']*100:.2f if stats['domain_stats'][domain]['consistency_rate'] is not None else 'N/A'}%</td>
        </tr>
            """ for domain in stats['domain_stats']])
        }}
    </table>
    
    <h2>详细评估结果</h2>
    <table>
        <tr>
            <th>序号</th>
            <th>问题领域</th>
            <th>评分</th>
            <th>一致性</th>
            <th>问题</th>
        </tr>
        {{
            ''.join([f"""
        <tr>
            <td>{r['item_index']+1}</td>
            <td>{r.get('domain', 'unknown')}</td>
            <td>{r['evaluation']['score'] if r['evaluation']['score'] else 'N/A'}</td>
            <td class="{'consistent' if r['evaluation']['is_consistent'] else 'inconsistent' if r['evaluation']['is_consistent'] is not None else 'unknown'}">
                {'一致' if r['evaluation']['is_consistent'] else '不一致' if r['evaluation']['is_consistent'] is not None else '未知'}
            </td>
            <td>{r['problem'][:100]}...</td>
        </tr>
            """ for r in results])
        }}
    </table>
    
    <script>
        // 绘制评分分布图表
        const scoreCtx = document.getElementById('scoreChart').getContext('2d');
        new Chart(scoreCtx, {{
            type: 'bar',
            data: {{
                labels: ['1分', '2分', '3分', '4分', '5分'],
                datasets: [{{
                    label: '评分分布',
                    data: [{stats['score_distribution'].get(1, 0)}, 
                           {stats['score_distribution'].get(2, 0)}, 
                           {stats['score_distribution'].get(3, 0)}, 
                           {stats['score_distribution'].get(4, 0)}, 
                           {stats['score_distribution'].get(5, 0)}],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.5)',
                        'rgba(255, 159, 64, 0.5)',
                        'rgba(255, 205, 86, 0.5)',
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(54, 162, 235, 0.5)'
                    ],
                    borderColor: [
                        'rgb(255, 99, 132)',
                        'rgb(255, 159, 64)',
                        'rgb(255, 205, 86)',
                        'rgb(75, 192, 192)',
                        'rgb(54, 162, 235)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: '数量'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''
        
        # 保存HTML报告
        html_file = os.path.join(results_dir, "evaluation_report.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"已保存HTML报告到 {html_file}")
        
    except Exception as e:
        print(f"生成HTML报告时出错: {e}")

# 合并所有批次结果
def merge_results():
    all_results = []
    for filename in os.listdir(results_dir):
        if filename.startswith("evaluation_batch_") and filename.endswith(".json"):
            file_path = os.path.join(results_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
    
    # 保存合并结果
    merged_file = os.path.join(results_dir, "all_evaluation_results.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"已合并所有结果到 {merged_file}")
    
    # 生成统计报告
    generate_report(all_results)
    
    return all_results

# 主函数
def main():
    parser = argparse.ArgumentParser(description='使用多账号并行评估解决方案')
    # parser.add_argument('input_files', nargs='+', help='输入文件路径，包含要评估的解决方案')
    parser.add_argument('--batch-size', type=int, default=10, help='批处理大小')
    parser.add_argument('--concurrent', type=int, default=5, help='并发请求数')
    parser.add_argument('--merge-only', action='store_true', help='仅合并已有的批次结果')
    args = parser.parse_args()

    input_files = ["qwen_results/results_batch_{}.json".format(i) for i in range(50)]
    
    if args.merge_only:
        print("仅合并已有的批次结果...")
        merge_results()
    else:
        print(f"开始评估解决方案，使用 {len(API_CONFIGS)} 个账号...")
        # asyncio.run(evaluate_solutions_async(args.input_files, args.batch_size, args.concurrent))
        asyncio.run(evaluate_solutions_async(input_files, args.batch_size, args.concurrent))

if __name__ == "__main__":
    main()