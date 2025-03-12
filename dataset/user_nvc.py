# nohup python -m cot_computation.reason_user_info --function reason-description > user_description.log 2>&1 &
# nohup python -m cot_computation.reason_user_info --function reason-user-state > user_state.log 2>&1 &
import os
import random
from fire import Fire
from tqdm import tqdm, trange
import logging
from typing import Tuple, List
from multiprocessing import Pool, cpu_count, Manager
from utils.config_utils import *
from utils.api_utils import OpenAIChatBot
from utils.message_utils import Message
from time import sleep

logging.getLogger().setLevel(logging.INFO)
model = OpenAIChatBot(model=MODEL_PATH['gpt-4o'])


def state_single_infer(args: Tuple[Tuple[Message, int], List[str]]) -> Tuple[int, str]:
    """
    :param args:
    :return:
    """
    role_desc = '请扮演一个精通非暴力沟通和同理回应的心理学专家。根据来访者和心理倾听者的对话内容推断当前来访者的观察、感受、需要和请求。'
    identify_desc = '根据历史对话判断给定的来访者状态中哪些为尚未被来访者确认的猜测内容，哪些为来访者描述过或者确认过的推断内容。并以json格式输出。'
    
    (message, index), end_points = args
    current_end_points = random.sample(list(end_points), len(end_points))
    
    end_point = current_end_points[-1]
    while True:
        user_state = model.query(
            role_desc=role_desc, history_messages=[message], temperature=0, azure_endpoint=end_point
        )
        if '<<<<<<END_OF_CONVERSATION>>>>>>' not in user_state:
            break
        # Rotate endpoints safely with Manager list
        end_point = current_end_points.pop(0)
        current_end_points.append(end_point)
    
    end_point = current_end_points[-1]
    while True:
        inference_facts = Message(
            content=f'其中哪些为尚未被确认的猜测内容，哪些为来访者描述或者确认的内容。输出json格式：\n\n{{"确认内容": {{"观察": [], "感受": [], "需求": [], "请求": []}}, "猜测内容": {{"观察": [], "感受": [], "需求": [], "请求": []}}}}')
        user_state_json = model.query(
            role_desc=identify_desc,
            history_messages=[message, Message(content=user_state), inference_facts],
            temperature=0,
            azure_endpoint=end_point
        )
        if '<<<<<<END_OF_CONVERSATION>>>>>>' not in user_state_json:
            break
        end_point = current_end_points.pop(0)
        current_end_points.append(end_point)
        user_state_json = user_state_json.split('```json')[-1].split('```')[0]
    
    return index, user_state_json


def reason_user_state():
    """
    Reason the user state (观察、感受、需求和请求) after each user utterance using gpt-4o
    When the api does not work, the code running will be interrupted. But the processed data can be saved.
    The reset data can be processed once the api works again and the codes are run.

    :return: save the user states during the conversations in the file USER_STATE_PATH
    """
    # all end points
    end_points = [end for end in END_POINTS]
    
    # prepare dataset
    conv_data = json.load(open(TRAIN_DATA_PATH, 'r', encoding='utf-8'))
    logging.info('There are {} conversations in the dataset.'.format(len(conv_data)))
    
    # user_state_data = []
    if os.path.exists(USER_STATE_PATH):
        with open(USER_STATE_PATH, 'r', encoding='utf-8') as fp:
            # start_conv_idx = len(fp.readlines()) + 100
            start_conv_idx = len(fp.readlines())
        logging.info('There are {} user states in the dataset.'.format(start_conv_idx))
    else:
        start_conv_idx = 0
    
    # data_writer = open(f'{ROOT_DIR}/datasets/PsyDTCorpus/PsyDTCorpus_half_user_state.jsonl', 'a+', encoding='utf-8')
    data_writer = open(USER_STATE_PATH, 'a+', encoding='utf-8')
    for conv_idx in trange(start_conv_idx, len(conv_data), desc='Formatting batches', position=0, leave=False):
        utterances = conv_data[conv_idx]['messages'][1:]
        inputs = []
        for turn_idx, turn in enumerate(utterances):
            if turn['role'] == 'user' and turn_idx > 0:
                conv_context = '\n'.join([f'{ROLE_MAP[t["role"]]}: {t["content"]}' for t in utterances[:turn_idx]])
                base_message = Message(
                    content=f'请根据以下对话内容推断来访者当前最想或者最需要讨论的的观察、感受、需要和请求。在非暴力沟通中，这四个概念分别是以下含义：\n'
                            f'观察：描述你在具体情况下观察到的事实，而不包含任何评价或判断。例如：我的学习成绩差劲/我的家庭不幸福，是评论。在过去的5次考试中，我的排名都是倒数第一/我的家中父母每周只回家一次，是观察。\n'
                            f'感受：表达你对于这些观察到的事实的感受，如快乐、悲伤、恐惧、愤怒等。例如：我对朋友的离开感到伤心。\n'
                            f'需求：识别并表达引发这些感受的内在需求（自由选择、爱、尊重、信心、支持等，而不是具体事情）。例如：我需要受到尊重，是一个内在需求。而我想要朋友能认真听他讲话，则不是一个好的需求描述。\n'
                            f'请求：清晰、具体、可行地提出请求，以满足这些需求。例如：我希望儿子把儿子自己的房间打扫干净，是一个明确的请求。而我希望我的丈夫多多照顾我，则不是一个明确的请求。\n'
                            f'注意在识别这四个方面内容的时候，尤其是观察（例如：我每周都有三天失眠），请不要将任何评论（例如：我经常失眠）与这些概念混淆，并且不要使用指代不清的代词例如：“这件事”、“他”、“这个”等。\n\n'
                            f'对话历史：\n{conv_context}\n\n来访者当前的输入：{turn["content"]}')
                inputs.append((base_message, turn_idx))
        
        assert (len(inputs) + 1) * 2 == len(utterances)
        while True:
            user_state_list = []
            try:
                with Manager() as manager:
                    with Pool(cpu_count()) as pool:
                        result_list = list(tqdm(
                            pool.imap_unordered(
                                state_single_infer,
                                [(arg, random.sample(list(end_points), len(end_points))) for arg in inputs]
                            ),
                            desc='Inferring per conversation',
                            position=2,
                            leave=False,
                            total=len(inputs)
                        ))
                    result_list.sort(key=lambda x: x[0])
                    for result in result_list:
                        index, user_state_json = result
                        user_state_json = user_state_json.split('```json')[-1].split('```')[0]
                        user_state_list.append({index + 1: json.loads(user_state_json)})
                assert (len(user_state_list) + 1) * 2 == len(utterances)
                break
            except Exception as e:
                continue
        new_line = {
            'id': conv_data[conv_idx]['id'],
            'normalizedTag': conv_data[conv_idx]['normalizedTag'],
            'messages': conv_data[conv_idx]['messages'],
            'user_states': user_state_list
        }
        data_writer.write(json.dumps(new_line, ensure_ascii=False) + '\n')
    
    data_writer.close()


def desc_single_infer(args: Tuple[Tuple[Message, int], List[str]]) -> Tuple[int, str]:
    role_desc = '根据对话描述来访者的来访原因和遇到的问题。要求输出内容给定的例子形式和格式上相似。'
    
    (message, index), current_end_points = args
    end_point = current_end_points[-1]
    while True:
        user_description = model.query(
            role_desc=role_desc, history_messages=[message], temperature=0, azure_endpoint=end_point)
        if '<<<<<<END_OF_CONVERSATION>>>>>>' not in user_description:
            break
        # Rotate endpoints safely with Manager list
        sleep(random.randint(10, 30))
        end_point = current_end_points.pop(0)
        current_end_points.append(end_point)
    
    return index, user_description


def reason_description(batch_size: int = 4):
    # load dataset
    raw_data = json.load(open('dataset/PsyQA_full.json', 'r', encoding='utf-8'))
    description_examples = [line['description'] for line in raw_data]
    dataset = json.load(open(TRAIN_DATA_PATH, 'r', encoding='utf-8'))
    
    inputs = []
    for i in trange(len(dataset), desc='Formatting data', position=0, leave=False):
        conversation = dataset[i]['messages'][1:]
        context = '\n'.join([f"{ROLE_MAP[turn['role']]}: {turn['content']}" for turn in conversation])
        samples = '\n'.join(random.sample(description_examples, 5))
        message = f'倾听者和来访者对话内容如下：\n{context}\n\n描述来访者来访的原因和遇到的问题。以下是一些描述的例子：\n{samples}\n\n请描述对话中来访者的来访原因和问题。'
        inputs.append((Message(content=message), i))
    
    if os.path.exists(f'{ROOT_DIR}/datasets/PsyDTCorpus/PsyDTCorpus_train_user_description.jsonl'):
        with open(f'{ROOT_DIR}/PsyDTCorpus/PsyDTCorpus_train_user_description.jsonl', 'r', encoding='utf-8') as fp:
            start_conv_idx = len(fp.readlines())
        logging.info('There are {} user descriptions in the dataset.'.format(start_conv_idx))
    else:
        start_conv_idx = 0
    
    data_writer = open(f'{ROOT_DIR}/datasets/PsyDTCorpus/PsyDTCorpus_train_user_description.jsonl', 'a+',
                       encoding='utf-8')
    end_points = [end for end in END_POINTS]
    with Manager() as manager:
        for i in tqdm(range(start_conv_idx, len(inputs), batch_size), desc='Inferring User Information', position=0,
                      leave=False):
            args = inputs[i: i + batch_size]
            with Pool(cpu_count()) as pool:
                result_list = list(tqdm(
                    pool.imap_unordered(
                        desc_single_infer,
                        [(arg, random.sample(list(end_points), len(end_points))) for arg in args]
                    ),
                    desc='Inferring batches',
                    position=1,
                    leave=False,
                    total=len(args)
                ))
            result_list.sort(key=lambda x: x[0])
            for result in result_list:
                index, user_description = result
                new_line = {
                    'id': dataset[index]['id'],
                    'normalizedTag': dataset[index]['normalizedTag'],
                    'messages': dataset[index]['messages'],
                    'description': user_description
                }
                data_writer.write(json.dumps(new_line, ensure_ascii=False) + '\n')
            random.shuffle(end_points)
    data_writer.close()


def main(function: str = ''):
    if function == 'reason-user-state':
        reason_user_state()
    elif function == 'reason-description':
        reason_description()


if __name__ == '__main__':
    Fire(main)
