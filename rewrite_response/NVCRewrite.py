"""
This file rewrite some responses in the dataset.
These responses should be detected as non-nvc responses, as they are detected some features in their responses or cots.

The detection results are sorted in dataset/PsyDTCorpus_cot/cot_verification.jsonl
The rewritten results are sorted in dataset/PsyDTCorpus_rewritten/nvc_dataset.jsonl
"""
import json
from tqdm import tqdm


def clustering():
    classes = ['【安慰】', '【教育】', '【推卸感受责任】', '【同情】', '【纠正】', '【未经允许的建议】', '【自身分享】']


def main():
    with open('dataset/PsyDTCorpus_cot/cot_verification.jsonl', 'r', encoding='utf-8') as fp:
        cot_verification = [json.loads(line) for line in fp.readlines()]
    
    counter = 0
    for line in tqdm(cot_verification, total=len(cot_verification)):
        if line['label'] != 'dataset':
            continue
        
        resp_verify = line['response_verify']
        cot_verify = line['cot_verify']
        
        if "不存在列举的问题。" in resp_verify and "不存在列举的意图。" in cot_verify:
            continue
        
        if '【未经允许的建议】' in resp_verify or '【未经允许的建议】' in cot_verify:
            conversation = line['conversation']
            query = conversation[-1]["content"]
            if "建议" in query:
                print(query)
                counter += 1
            continue
            print(f'query: {query}')
            response = line['response']
            if response[-1] == '？':
                continue
            print(f'response: {response} ***** {resp_verify}')
            
            cot = line['cot'].replace('```markdown', '').replace('```', '')
            print(f'cot: {cot} ***** {cot_verify}')
            counter += 1
    
    print(counter)


if __name__ == '__main__':
    main()
