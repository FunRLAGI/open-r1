import io
import re
import sys
import json
import argparse
import multiprocessing
from datasets import load_dataset

def extract_code(text):
    # 使用正则表达式查找代码块
    try:
        code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        # assert len(code_blocks) == 1, f"提取的代码列表长度应为 1, 但实际长度 {len(code_blocks)}"
        return code_blocks[0]
    except:
        return "fail to extract code"

def execute_code(code, input_case, output_queue):
    """在单独的进程中执行代码"""
    # 创建一个新的 StringIO 对象，用于模拟标准输入
    sys.stdin = io.StringIO(input_case)
    # 创建一个新的 StringIO 对象，用于捕获标准输出
    sys.stdout = io.StringIO()

    try:
        # 执行代码字符串
        exec(code, globals())
        # 获取输出结果
        output = sys.stdout.getvalue().strip()
    except Exception as e:
        output = str(e)
    finally:
        # 恢复标准输入和输出
        sys.stdin = sys.__stdin__
        sys.stdout = sys.__stdout__
    
    output_queue.put(output)

def run_code_with_timeout(code, timeout, input_case):
    """使用多进程来执行代码并设置超时"""
    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code, args=(code, input_case, output_queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return "Execution timed out"

    return output_queue.get()

def main(llm_result_path, test_case_path, timeout_threshold, pass_rate_save_path):
    test_case_dataset = load_dataset('csv', data_files=test_case_path, split='train')
    assert len(test_case_dataset) == 993, f"hard code"

    with open(llm_result_path, 'r', encoding='utf-8') as problem_answer_pair_file:
        problem_answer_pair_data = json.load(problem_answer_pair_file)
        problem_answer_pair_num = len(problem_answer_pair_data)

        for problem_answer_pair_index in range(problem_answer_pair_num):

            qwen_solution = problem_answer_pair_data[problem_answer_pair_index]['qwen_solution']
            qwen_solution_extracted_code = extract_code(qwen_solution)

            # 测试用例
            _test_cases = json.loads(test_case_dataset[problem_answer_pair_index]['test_cases'])  # TODO:
            # assert set(_test_cases.keys()) == {"inputs", "outputs"}, f"_test_cases keys are invalid: {_test_cases.keys()}"
            assert 'inputs' in _test_cases.keys(), "The key 'inputs' is missing in test_cases"
            assert 'outputs' in _test_cases.keys(), "The key 'outputs' is missing in test_cases"

            # 执行每个测试用例
            total_case_num = 0
            passed_case_num = 0
            failed_case_num = 0

            for case_index, input_case in enumerate(_test_cases["inputs"]):
                # input_case -> str(input_case), case 325
                _case_output = run_code_with_timeout(qwen_solution_extracted_code, 5, str(input_case))

                # 检查输出是否与预期结果匹配
                expected_output = str(_test_cases["outputs"][case_index])
                if _case_output.strip() == expected_output.strip():
                    print(f"Test Case {case_index + 1}: {'Passed'}")
                    passed_case_num += 1
                else:
                    print(f"Test Case {case_index + 1}: {'Failed'}")
                    failed_case_num += 1

                total_case_num += 1

                print(f"Expected: {expected_output}, Got: {_case_output}")

            pass_rate = passed_case_num / total_case_num
            print(f"case通过率: {pass_rate}")
            # 将结果写入文件
            with open(pass_rate_save_path, 'a', encoding='utf-8') as result_file:
                result_file.write(f"Problem {problem_answer_pair_index + 1}: case通过率: {pass_rate}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify file paths for processing.')
    parser.add_argument('--llm_result_path', default='/home/xuweixiang01/qwen_results/all_results.json', 
                            type=str, help='通过 ask_qwen_7b_multi_account 得到的LLM回答结果')
    parser.add_argument('--test_case_path', default='/home/xuweixiang01/code_domain_data_tiny.csv', 
                            type=str, help='通过 select.py 从 openthought 抽取的code domain的数据(1/20), 993条')
    parser.add_argument('--timeout_threshold', default=10, 
                            type=int, help='一个case的最大运行时间阈值, 超过10s视为失败')
    parser.add_argument('--pass_rate_save_path', default='results123.txt',
                            type=str, help='每个 problem 的 case 通过率, 保存到一个txt')
    args = parser.parse_args()

    main(args.llm_result_path, args.test_case_path, args.timeout_threshold, args.pass_rate_save_path)