"""
This file is to cluster the response cots.
nohup python -m cot_computation.cot_verify > cot_verify.log 2>&1 &
"""

import os
import logging
import random
from fire import Fire
from multiprocessing import Pool, cpu_count, Manager
from typing import Tuple, List, Union
from tqdm import tqdm
from utils.template_utils import get_template
from utils.api_utils import OpenAIChatBot
from utils.message_utils import Message
from utils.config_utils import *

logging.getLogger().setLevel(logging.INFO)
model = OpenAIChatBot(model=MODEL_PATH['gpt-4o'])


def api_single_infer(args: Tuple[List[Message], List[Message], int]) -> Tuple[int, str, str]:
    cot_message, response_message, index = args
    if type(cot_message) == str:
        return index, response_message, cot_message
    end_points = [
        'https://gcraoai5sw1.openai.azure.com/', 'https://gcrgpt4aoai5c.openai.azure.com/',
        'https://gcrgpt4aoai5.openai.azure.com/', 'https://gcraoai5sw2.openai.azure.com/',
        'https://gcraoai5sw3.openai.azure.com/', 'https://gcraoai9sw1.openai.azure.com/'
    ]
    random.shuffle(end_points)
    end_point = end_points[-1]
    while True:
        response_verify = model.query(
            role_desc='根据要求判断倾听者回复是否存在列举的问题。',
            history_messages=response_message,
            azure_endpoint=end_point
        ).replace('\n', '')
        if '<<<<<<END_OF_CONVERSATION>>>>>>' not in response_verify:
            break
        end_point = end_points.pop(0)
        end_points.append(end_point)
    if cot_message:
        end_point = end_points[-1]
        while True:
            cot_verify = model.query(
                role_desc='根据要求判断倾听者是否存在列举的意图。',
                history_messages=cot_message,
                azure_endpoint=end_point
            ).replace('\n', '')
            if '<<<<' not in cot_verify:
                break
            end_point = end_points.pop(0)
            end_points.append(end_point)
    else:
        cot_verify = ''
    return index, response_verify, cot_verify


def api_single_revise(args: Tuple[Union[Message, str], Union[Message, str], int]) -> Tuple[int, str, str]:
    end_points = [
        'https://gcraoai5sw1.openai.azure.com/', 'https://gcrgpt4aoai5c.openai.azure.com/',
        'https://gcrgpt4aoai5.openai.azure.com/', 'https://gcraoai5sw2.openai.azure.com/',
        'https://gcraoai5sw3.openai.azure.com/', 'https://gcraoai9sw1.openai.azure.com/'
    ]
    random.shuffle(end_points)
    response_message, cot_message, index = args
    rewritten = type(response_message) == Message
    if not rewritten:
        revised_response = response_message
    else:
        end_point = end_points[-1]
        while True:
            revised_response = model.query(
                role_desc='根据非暴力沟通原则和要求改写倾听者回复。',
                history_messages=[response_message],
                azure_endpoint=end_point
            )
            if '<<<<<<END_OF_CONVERSATION>>>>>>' not in revised_response:
                break
            end_point = end_points.pop(0)
            end_points.append(end_point)
    revised = type(cot_message) == Message
    if not revised:
        revised_cot = cot_message
    else:
        description = '任务：\n扮演一位与来访者对话的倾听者，描述倾听者在与来访者指定回复的思考过程，最终补充完整的思维链（……部分）。\n思维链内容包括：\n1. 倾听者对来访者状态的关注点（观察、情绪、需求或者请求），这个关注点直接影响倾听者的后续回复；\n2. 倾听者回复的策略（例如：建议、教育、安慰、回忆、否定、同情、询问等）和意图。\n\n要求：\n1. 视角：以倾听者的视角与口吻展开分析；\n2. 描述：详细说明倾听者回复背后的思维链；\n3. 思维过程：\n - 基于与来访者的对话历史作出推导；\n - 在推导过程中，倾听者不应预知或者提及后续回复的具体内容；\n - 通过思维链能够自然推导得出后续回复。'
        while True:
            revised_cot = model.query(role_desc=description, history_messages=[cot_message], azure_endpoint=end_point)
            if '<<<<<<END_OF_CONVERSATION>>>>>>' not in revised_cot:
                break
            end_point = end_points.pop(0)
            end_points.append(end_point)
    logging.info(
        f'Message: {response_message.content}\n\nRevised Response: {revised_response}\n\nRevised Cot: {revised_cot}')
    return index, revised_response, revised_cot


def statistic():
    classes = ['【安慰】', '【教育】', '【推卸感受责任】', '【同情】', '【纠正】', '【未经允许的建议】', '【自身分享】']
    existing_cot_issues, existing_response_issues = set(), set()
    with open('dataset/PsyDTCorpus_cot/response_verification.jsonl', 'r', encoding='utf-8') as fp:
        response_verification = [json.loads(line) for line in fp.readlines()]
    for line in response_verification:
        if line['label'] != 'dataset':
            continue
        cot_verify = line['cot_verify']
        response_verify = line['response_verify']
        for cls in classes:
            if cls in cot_verify and f'不存在{cls}' not in cot_verify:
                existing_cot_issues.add(cls)
            if cls in response_verify and f'不存在{cls}' not in response_verify:
                existing_response_issues.add(cls)
        if '【纠正】' in response_verify:
            print(line['response'])
    
    logging.info(f"Following issues existing in the CoTs: {existing_cot_issues}")
    logging.info(f"Following issues existing in the responses: {existing_response_issues}")


def revision(batch_size: int = 32):
    classes = {
        '【安慰】': '来访者：我怎么会做这么愚蠢的事？倾听者：没有人是完美的。你对自己太苛刻了。\t倾听者回复可以更改为：听到你这样说，我感受到你对自己的失望。如果你愿意的话，可以和我详细聊一聊这件事是怎样带给你这样的感受的。',
        '【教育】': '来访者：在我看来，所有这些移民，从哪儿来就该送回哪儿去。倾听者：这样的做法是不能解决所有问题的。\t倾听者回复可以更改为：我听到你对移民问题的强烈感受。能否分享一下你对这个问题的担忧和想法？我们可以一起探讨可能的解决方案。',
        '【推卸感受责任】': '来访者：他怎么可以这样和我说话？倾听者：他那样说话，你是不是很伤心？\t倾听者回复可以更改为：你是不是有些伤心，你希望他能同意你的请求？',
        '【同情】': '倾听者：你太可怜了。\t倾听者回复可以更改为：我能感受到你正在经历的痛苦。',
        '【纠正】': '倾听者：这件事情的经过不是这样的。\t倾听者回复可以更改为：这件事情有没有其他可能呢？',
        '【未经允许的建议】': '来访者：我讨厌自己变得越来越胖。倾听者：慢跑也许能帮助你。\t倾听者回复可以更改为：如果你想改变的话，慢跑也许能帮助你。你觉得这个建议会有帮助吗？'
    }
    # load dataset
    dataset = json.load(open('dataset/PsyDTCorpus_train/vanilla_train.json', 'r'))
    logging.info(f'There are totally {len(dataset)} lines in the training dataset.')
    logging.info(f'The data format is as follows: {dataset[0]}')
    # load cot dataset
    with open('dataset/PsyDTCorpus_cot/response_verification.jsonl', 'r') as fp:
        response_verification = [json.loads(line) for line in fp.readlines()]
    logging.info(f'There are totally {len(response_verification)} lines in the cot dataset.')
    logging.info(f'The data format is as follows: {response_verification[0]}')
    # revision
    negative_responses, positive_responses = [], []
    inputs = []
    positive_counter = 0
    generation_template = get_template('generate_cot')
    for idx, line in tqdm(enumerate(response_verification), total=len(response_verification)):
        response = line['response']
        cot = line['cot']
        cot_verify = line['cot_verify']
        response_verify = line['response_verify']
        if line['label'] == 'dataset':
            positive_counter += 1
            if '不存在列举的意图。' in cot_verify and '不存在列举的问题。' in response_verify:
                inputs.append((response, cot, idx))
            else:
                # the response and its cot should be modified can be classified as positive responses.
                cot_issues, response_issues = set(), set()
                for cls in classes.keys():
                    if cls in cot_verify and f'不存在{cls}' not in cot_verify:
                        cot_issues.add(cls)
                    if cls in response_verify and f'不存在{cls}' not in response_verify:
                        response_issues.add(cls)
                if len(response_issues) == 0:
                    cot_message = generation_template.format_example(target_data=line + {'user_state': ''})
                    inputs.append((response, cot_message, idx))
                else:
                    conversation = '\n'.join(
                        [f'{ROLE_MAP[l["role"]]}: {l["content"]}' for l in line["conversation"][-5:]])
                    next_query = dataset[positive_counter]
                    
                    response_issues_str = '，'.join(response_issues)
                    example_str = "\n\t".join([f"{cls} :{classes[cls]}" for cls in response_issues])
                    if next_query['conv_id'] == line['conv_id'] and next_query['turn_id'] == line['turn_id'] + 2:
                        post_context = f"倾听者回复（下文）：{next_query['conversation'][-1]['content']}\n\n"
                    else:
                        post_context = ''
                    response_message = Message(
                        content=f'根据非暴力沟通的原则改写指定倾听者回复，要求改写后的回复够回应上文和下文的来访者话语。\n\n'
                                f'历史对话（上文）：\n{conversation}\n\n需要改写的回复：{response}\n\n{post_context}'
                                f'当前回复“{response}”存在的问题为{response_issues_str}。以上问题的例子和可以更改的方式如下：\n\n'
                                f'{example_str}\n\n请仅输出更改后的倾听者回复。'
                    )
                    cot_message = Message(
                        content=''
                    )
                    inputs.append((response_message, cot_message, idx))
        
        
        else:
            if '不存在列举的意图。' in cot_verify and '不存在列举的问题。' in response_verify:
                continue
            new_line = {key: value for key, value in line.items()}
            new_line['label'] = 'negative'
            negative_responses.append(new_line)
    
    if os.path.exists('dataset/PsyDTCorpus_cot/revised_responses.jsonl'):
        with open('dataset/PsyDTCorpus_cot/revised_responses.jsonl', 'r') as fp:
            start_index = len(fp.readlines())
    else:
        start_index = 0
    
    data_writer = open('dataset/PsyDTCorpus_cot/revised_responses.jsonl', 'a+', encoding='utf-8')
    with Manager() as manager:
        for i in tqdm(range(start_index, len(inputs), batch_size), desc='Revising Responses', position=0, leave=False):
            args = inputs[i: i + batch_size]
            with Pool(cpu_count()) as pool:
                result_list = list(tqdm(
                    pool.imap_unordered(
                        api_single_revise,
                        args
                    ),
                    desc='Inferring batches',
                    position=1,
                    leave=False,
                    total=len(args)
                ))
            result_list.sort(key=lambda x: x[0])
            for result in result_list:
                index, response, cot = result
                new_line = {key: value for key, value in response_verification[index].items()}
                new_line['response'] = response
                new_line['cot'] = cot
                data_writer.write(json.dumps(new_line) + '\n')
    data_writer.close()


def main(batch_size: int = 128):
    """
    there will be two parts:
    1. whether each sentence in the cot is related to the user states
    2. whether the response/cot contains appraise
    """
    
    # load cot data.
    with open('dataset/PsyDTCorpus_cot/response_cot.jsonl', 'r') as f:
        cot_data = [json.loads(line) for line in f.readlines()]
    if os.path.exists('dataset/PsyDTCorpus_cot/response_verification.jsonl'):
        current_counter = len(open('dataset/PsyDTCorpus_cot/response_verification.jsonl', 'r').readlines())
    else:
        current_counter = 0
    input_list = []
    data_writer = open('dataset/PsyDTCorpus_cot/response_verification.jsonl', 'a+', encoding='utf-8')
    rewritten_data = json.load(open('dataset/PsyDTCorpus_backup/verify_rewritten.json', 'r'))
    original_data = json.load(open('dataset/PsyDTCorpus_backup/verify_original.json', 'r'))
    
    for index, line in tqdm(enumerate(cot_data), total=len(cot_data)):
        l = cot_data[index]['label']
        c_id = str(cot_data[index]['conv_id'])
        t_id = str(cot_data[index]['turn_id'])
        if l == 'rewritten' and rewritten_data.get(c_id, {}).get(t_id, None) is not None:
            input_list.append(
                (rewritten_data[c_id][t_id]['cot_verify'], rewritten_data[c_id][t_id]['response_verify'], index))
            del rewritten_data[c_id][t_id]
        elif l == 'dataset' and original_data.get(c_id, {}).get(t_id, None) is not None:
            input_list.append(
                (original_data[c_id][t_id]['cot_verify'], original_data[c_id][t_id]['response_verify'], index))
            del original_data[c_id][t_id]
        else:
            conversation, response, cot = line['conversation'], line['response'], line['cot']
            query = conversation[-1]['content']
            cot = cot.replace('```', '').replace('markdown', '')
            
            cot_message = (
                f'【来访者表述】：{query}\n\n【倾听者思考】：{cot}\n\n'
                '判断倾听者的在思考过程是是否有以下意图：\n'
                '【安慰】，例如：来访者：我怎么会做这么愚蠢的事？倾听者：没有人是完美的。你对自己太苛刻了。\n'
                '【教育】，例如：来访者：在我看来，所有这些移民，从哪儿来就该送回哪儿去。倾听者：这样的做法是不能解决所有问题的。\n'
                '【推卸感受责任】，例如：来访者：他怎么可以这样和我说话？倾听者：他那样说话，你是不是很伤心？（如果用同理的方式来回应来访者, 倾听者可以说：“你是不是有些伤心，你希望他能同意你的请求？”这个情感的来自于来访者的需求或者请求没有被完成。）\n'
                '【同情】，例如：倾听者：你太可怜了。\n'
                '【纠正】，例如：倾听者：这件事情的经过不是这样的。\n'
                '【未经允许的建议】，例如：来访者：我讨厌自己变得越来越胖。倾听者：慢跑也许能帮助你。（当来访者并没有明确表达自己不知道怎么做到某件事情或者需要建议的时候，倾听者给出的建议。如果来访者表述为：我想找到一个减肥的办法。则倾听着的回复就不是未经允许的建议。）\n'
                '【自身分享】，例如：来访者：当亲戚们不请自来时，我感到被侵犯。这让我想起，我的父母在过去经常无视我的需要，替我做安排。倾听者：我明白你的感受，曾经我也是这样。\n\n'
                '回复开头如下：``不存在列举的意图。``或者``存在【``'
            )
            response_message = (
                f'【来访者表述】：{query}\n【倾听者回复】：{response}\n\n'
                '判断倾听者的回复是否有以下特点：\n\n'
                '【安慰】，例如：来访者：我怎么会做这么愚蠢的事？倾听者：没有人是完美的。你对自己太苛刻了。\n'
                '【教育】，例如：来访者：在我看来，所有这些移民，从哪儿来就该送回哪儿去。倾听者：这样的做法是不能解决所有问题的。\n'
                '【推卸感受责任】，例如：来访者：他怎么可以这样和我说话？倾听者：他那样说话，你是不是很伤心？（如果用同理的方式来回应来访者, 倾听者可以说：“你是不是有些伤心，你希望他能同意你的请求？”这个情感的来自于来访者的需求或者请求没有被完成。）\n'
                '【同情】，例如：倾听者：你太可怜了。\n'
                '【纠正】，例如：倾听者：这件事情的经过不是这样的。\n'
                '【未经允许的建议】，例如：来访者：我讨厌自己变得越来越胖。倾听者：慢跑也许能帮助你。（当来访者并没有明确表达自己不知道怎么做到某件事情或者需要建议的时候，倾听者给出的建议。如果来访者表述为：我想找到一个减肥的办法。则倾听着的回复就不是未经允许的建议。）\n'
                '【自身分享】，例如：来访者：当亲戚们不请自来时，我感到被侵犯。这让我想起，我的父母在过去经常无视我的需要，替我做安排。倾听者：我明白你的感受，曾经我也是这样。\n\n'
                '回复开头如下：\n不存在列举的问题。\n或者：存在【'
            )
            if '来访者' not in cot:
                input_list.append((None, [Message(content=response_message)], index))
            else:
                input_list.append(([Message(content=cot_message)], [Message(content=response_message)], index))
    
    with Manager() as manager:
        for i in tqdm(
                range(current_counter, len(input_list), batch_size), desc='Inferring CoTs', position=0, leave=False):
            batch_input_list = input_list[i: i + batch_size]
            # try:
            with Pool(processes=cpu_count()) as pool:
                result_list = list(tqdm(
                    pool.imap_unordered(api_single_infer, batch_input_list),
                    desc='Inferring batches',
                    position=1,
                    leave=False,
                    total=len(batch_input_list)
                ))
                result_list.sort(key=lambda x: x[0])
                for result in result_list:
                    index, response_verify, cot_verify = result
                    output_line = {
                        'conv_id': cot_data[index]['conv_id'],
                        'turn_id': cot_data[index]['turn_id'],
                        'conversation': cot_data[index]['conversation'],
                        'response': cot_data[index]['response'],
                        'cot': cot_data[index]['cot'],
                        'label': cot_data[index]['label'],
                        'response_verify': response_verify,
                        'cot_verify': cot_verify
                    }
                    data_writer.write(json.dumps(output_line, ensure_ascii=False) + '\n')
    
    data_writer.close()
    statistic()


def double_check():
    """
    this function is manually designed based on the statistic of the response_verification.jsonl
    
    the outputs of statistic() is:
    ```
    Following issues existing in the CoTs: {'【安慰】', '【同情】', '【未经允许的建议】', '【教育】', '【推卸感受责任】'}
    Following issues existing in the responses: {'【安慰】', '【同情】', '【未经允许的建议】', '【教育】', '【纠正】', '【推卸感受责任】'}
    ```
    """
    classes = ['【安慰】', '【教育】', '【推卸感受责任】', '【同情】', '【纠正】', '【未经允许的建议】', '【自身分享】']
    with open('dataset/PsyDTCorpus_cot/response_verification.jsonl', 'r', encoding='utf-8') as fp:
        response_verification = [json.loads(line) for line in fp.readlines()]
    
    verified_data = []
    for line in response_verification:
        if line['label'] != 'dataset':
            continue
        cot_verify = line['cot_verify']
        response_verify = line['response_verify']
        response = line['response']
        cot = line['cot']
        
        response_issues, cot_issues = set(), set()
        for cls in classes:
            if cls in cot_verify and f'不存在{cls}' not in cot_verify:
                cot_issues.add(cls)
            if cls in response_verify and f'不存在{cls}' not in response_verify:
                response_issues.add(cls)
        
        if set(response_issues) | set(cot_issues) == {'【未经允许的建议】'}:
            if '建议' in line['conversation'][-1]['content']:
                # print(line['conversation'][-1]['content'])
                # print(cot_verify)
                cot_verify = '不存在列举的意图。'
                response_verify = '不存在列举的问题。'
                # TODO: Use GPT to rewrite the cot, add the "来访者向我寻求帮助"。
                # print(cot)
                # print("------------------------------")
            elif '？' == response[-1]:
                cot_verify = '不存在列举的意图。'
                response_verify = '不存在列举的问题。'
                # TODO: Use GPT to rewrite the cot, add "试探性提出建议，并且询问来访者反馈"。
            elif '咨询' in response:
                response = ''
                # TODO: Use GPT to rewrite the cot, "我非常想要帮助你，但是我的能力有限。如果你想的话，可以寻求专业的人士。"
                # print('\n'.join([f"{l['role']}: {l['content']}" for l in line['conversation']]))
                # print()
                # print(cot_verify)
                # print(response_verify)
                # print(response)
                # print("**************************************************")
            # else:
            #     print(line['conversation'][-1]['content'])
            #     print(response)
            #     print()
        elif len(response_issues) > 0 or len(cot_issues) > 0:
            print(response_issues)
            print(response_verify)
            print(cot_issues)
            print(cot_verify)
            print(response)
            print("***********************************")
        
        output_line = {
            'conv_id': line['conv_id'],
            'turn_id': line['turn_id'],
            'conversation': line['conversation'],
            'response': response,
            'cot': cot,
            'label': line['label'],
            'response_verify': response_verify,
            'cot_verify': cot_verify
        }
        verified_data.append(json.dumps(output_line))
    
    with open('dataset/PsyDTCorpus_cot/dataset_verification.jsonl', 'w') as fp:
        fp.write('\n'.join(verified_data))


if __name__ == '__main__':
    Fire(main)
    # statistic()
    # double_check()
    # revision()
