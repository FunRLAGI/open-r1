import os
import json
import time
import argparse
from openai import OpenAI
import numpy as np
from tqdm import tqdm

# 设置API密钥和基础URL
client = OpenAI(
    api_key="sk-zcjkxltmsursqcfqareybswtborbpsdinrnexkhmtxbccotg", 
    base_url="https://api.siliconflow.cn/v1"
)

# 创建结果存储目录
results_dir = "./evaluation_results"
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

请严格按照以下标准给出评分（1-5分）：
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

# 调用API的函数
def query_model(prompt, model="deepseek-ai/DeepSeek-V3", max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的解决方案评估专家，擅长分析和比较不同解决方案的异同点。你的评估必须客观严格，避免给出中庸的评分。当解决方案之间存在明显差异时，应给予较低分数；当解决方案高度一致时，应给予较高分数。请避免将3分作为默认选择。"}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 使用更低的温度以获得更确定性的评估
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用错误 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                return f"评估失败: {str(e)}"

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
        
        # 检查评分与一致性结论是否匹配
        if score is not None and is_consistent is not None:
            if (score >= 4 and not is_consistent) or (score <= 2 and is_consistent):
                print("警告：评分与一致性结论不匹配，可能需要人工审核")
        
        # 检查是否存在评分偏向中间值的情况
        if score == 3:
            # 检查是否有明确的理由支持3分评分
            has_clear_reason = '部分一致' in evaluation_text and '部分不一致' in evaluation_text
            if not has_clear_reason:
                print("警告：评分为3分但缺乏明确理由，可能需要人工审核")
        
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

# 处理数据集并评估解决方案
def evaluate_solutions(input_files, batch_size=10):
    all_results = []
    all_items = []
    
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
    
    # 批量处理数据
    for batch_idx in range(0, len(all_items), batch_size):
        batch = all_items[batch_idx:batch_idx + batch_size]
        batch_results = []
        
        for i, item in enumerate(tqdm(batch, desc=f"处理批次 {batch_idx//batch_size + 1}/{(len(all_items)-1)//batch_size + 1}")):
            try:
                # 提取必要字段
                problem = item['problem']
                deepseek_solution = item['deepseek_solution']
                qwen_solution = item['qwen_solution']
                domain = item.get('domain', '')
                
                # 构建评估提示
                prompt = get_evaluation_prompt(problem, deepseek_solution, qwen_solution)
                
                # 调用模型进行评估
                print(f"\n评估第 {batch_idx + i + 1}/{len(all_items)} 条数据")
                evaluation = query_model(prompt)
                
                # 解析评估结果
                parsed_evaluation = parse_evaluation(evaluation)
                
                # 保存结果
                result = {
                    "problem": problem,
                    "domain": domain,
                    "evaluation": parsed_evaluation,
                    "item_index": batch_idx + i
                }
                batch_results.append(result)
                all_results.append(result)
                
                # 打印简短的评估摘要
                score_str = f"{parsed_evaluation['score']}分" if parsed_evaluation['score'] else "未知分数"
                consistency_str = "一致" if parsed_evaluation['is_consistent'] else "不一致" if parsed_evaluation['is_consistent'] is not None else "未知"
                print(f"评估结果: {score_str}, {consistency_str}")
                
                # 避免API限制
                time.sleep(1)
                
            except Exception as e:
                print(f"处理数据项时出错: {e}")
                batch_results.append({
                    "error": str(e),
                    "item_index": batch_idx + i
                })
        
        # 保存批次结果
        batch_file = os.path.join(results_dir, f"evaluation_batch_{batch_idx//batch_size}.json")
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        print(f"已保存批次结果到 {batch_file}")
    
    # 生成统计报告
    generate_report(all_results)
    
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
                              consistency[i] is not None and 
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
        generate_html_report(results, stats)
        
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
        <p>平均评分: {stats['average_score']:.2f}/5.0</p>
        <p>一致性比率: {stats['consistency_rate']*100:.2f}%</p>
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
            """ for r in valid_results])
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

# 评估qwen_solution_w_reasoning与deepseek_solution的一致性
def evaluate_solutions_with_reasoning(input_files, batch_size=10):
    # 类似evaluate_solutions函数，但使用qwen_solution_w_reasoning而不是qwen_solution
    all_results = []
    all_items = []
    
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
    
    # 批量处理数据
    for batch_idx in range(0, len(all_items), batch_size):
        batch = all_items[batch_idx:batch_idx + batch_size]
        batch_results = []
        
        for i, item in enumerate(tqdm(batch, desc=f"处理批次 {batch_idx//batch_size + 1}/{(len(all_items)-1)//batch_size + 1}")):
            try:
                # 提取必要字段
                problem = item['problem']
                deepseek_solution = item['deepseek_solution']
                qwen_solution_w_reasoning = item['qwen_solution_w_reasoning']  # 使用带推理的解决方案
                domain = item.get('domain', '')
                
                # 构建评估提示
                prompt = get_evaluation_prompt(problem, deepseek_solution, qwen_solution_w_reasoning)
                
                # 调用模型进行评估
                print(f"\n评估第 {batch_idx + i + 1}/{len(all_items)} 条数据 (带推理)")
                evaluation = query_model(prompt)
                
                # 解析评估结果
                parsed_evaluation = parse_evaluation(evaluation)
                
                # 保存结果
                result = {
                    "problem": problem,
                    "domain": domain,
                    "evaluation": parsed_evaluation,
                    "item_index": batch_idx + i,
                    "with_reasoning": True
                }
                batch_results.append(result)
                all_results.append(result)
                
                # 打印简短的评估摘要
                score_str = f"{parsed_evaluation['score']}分" if parsed_evaluation['score'] else "未知分数"
                consistency_str = "一致" if parsed_evaluation['is_consistent'] else "不一致" if parsed_evaluation['is_consistent'] is not None else "未知"
                print(f"评估结果 (带推理): {score_str}, {consistency_str}")
                
                # 避免API限制
                time.sleep(1)
                
            except Exception as e:
                print(f"处理数据项时出错: {e}")
                batch_results.append({
                    "error": str(e),
                    "item_index": batch_idx + i,
                    "with_reasoning": True
                })
        
        # 保存批次结果
        batch_file = os.path.join(results_dir, f"evaluation_with_reasoning_batch_{batch_idx//batch_size}.json")
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        print(f"已保存批次结果到 {batch_file}")
    
    # 生成统计报告
    generate_report_with_reasoning(all_results)
    
    return all_results

# 生成带推理的评估统计报告
def generate_report_with_reasoning(results):
    # 类似generate_report函数，但针对带推理的评估结果
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
                "consistency_rate": sum(domain_consistency) / len(domain_consistency) if domain_consistency else None
            }
        
        # 保存统计报告
        report_file = os.path.join(results_dir, "evaluation_with_reasoning_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"已保存带推理的评估报告到 {report_file}")
        
        # 打印摘要
        print("\n带推理的评估摘要:")
        print(f"总数据项: {stats['total_items']}")
        print(f"有效评估: {stats['valid_evaluations']}")
        if stats['average_score']:
            print(f"平均评分: {stats['average_score']:.2f}/5.0")
        if stats['consistency_rate'] is not None:
            print(f"一致性比率: {stats['consistency_rate']*100:.2f}%")
        print("\n评分分布:")
        for score, count in stats['score_distribution'].items():
            print(f"  {score}分: {count} 项 ({count/len(scores)*100:.2f}%)")
        
    except Exception as e:
        print(f"生成带推理的报告时出错: {e}")

# 比较有无推理的评估结果
def compare_evaluations(input_files, batch_size=10):
    # 评估不带推理的解决方案
    print("\n===== 评估不带推理的解决方案 =====\n")
    results_without_reasoning = evaluate_solutions(input_files, batch_size)
    
    # 评估带推理的解决方案
    print("\n===== 评估带推理的解决方案 =====\n")
    results_with_reasoning = evaluate_solutions_with_reasoning(input_files, batch_size)
    
    # 比较结果
    print("\n===== 比较评估结果 =====\n")
    
    # 提取有效结果
    valid_without = [r for r in results_without_reasoning if 'evaluation' in r and 'error' not in r and r['evaluation']['score'] is not None]
    valid_with = [r for r in results_with_reasoning if 'evaluation' in r and 'error' not in r and r['evaluation']['score'] is not None]
    
    # 计算平均分数
    avg_score_without = np.mean([r['evaluation']['score'] for r in valid_without])
    avg_score_with = np.mean([r['evaluation']['score'] for r in valid_with])
    
    # 计算一致性比率
    consistency_without = [r['evaluation']['is_consistent'] for r in valid_without if r['evaluation']['is_consistent'] is not None]
    consistency_with = [r['evaluation']['is_consistent'] for r in valid_with if r['evaluation']['is_consistent'] is not None]
    
    consistency_rate_without = sum(consistency_without) / len(consistency_without) if consistency_without else None
    consistency_rate_with = sum(consistency_with) / len(consistency_with) if consistency_with else None
    
    # 打印比较结果
    print(f"不带推理的平均评分: {avg_score_without:.2f}/5.0")
    print(f"带推理的平均评分: {avg_score_with:.2f}/5.0")
    print(f"评分差异: {avg_score_with - avg_score_without:.2f}")
    
    print(f"\n不带推理的一致性比率: {consistency_rate_without*100:.2f}%")
    print(f"带推理的一致性比率: {consistency_rate_with*100:.2f}%")
    print(f"一致性比率差异: {(consistency_rate_with - consistency_rate_without)*100:.2f}%")

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估Qwen解决方案与DeepSeek解决方案的一致性")
    parser.add_argument("--input", "-i", nargs="+", default=["./qwen_results/results_batch_0.json", "./qwen_results/results_batch_1.json"],
                        help="输入JSON文件路径，可以指定多个文件")
    parser.add_argument("--batch_size", "-b", type=int, default=10,
                        help="批处理大小，每处理多少条数据保存一次结果")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="是否比较带推理和不带推理的解决方案")
    parser.add_argument("--only_with_reasoning", "-r", action="store_true",
                        help="是否只评估带推理的解决方案")
    parser.add_argument("--only_without_reasoning", "-w", action="store_true",
                        help="是否只评估不带推理的解决方案")
    
    args = parser.parse_args()
    
    if args.compare:
        # 比较带推理和不带推理的解决方案
        compare_evaluations(args.input, args.batch_size)
    elif args.only_with_reasoning:
        # 只评估带推理的解决方案
        evaluate_solutions_with_reasoning(args.input, args.batch_size)
    elif args.only_without_reasoning:
        # 只评估不带推理的解决方案
        evaluate_solutions(args.input, args.batch_size)
    else:
        # 默认评估两个解决方案
        evaluate_solutions(args.input, args.batch_size)
        evaluate_solutions_with_reasoning(args.input, args.batch_size)
    
    print("\n评估完成！结果已保存到", results_dir)
