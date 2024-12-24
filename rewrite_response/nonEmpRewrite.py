"""
nohup python -m rewrite_response.nonEmpRewrite > nonEmpRewrite.log 2>&1 &
"""
import os
import json
import random
from tqdm import tqdm
from fire import Fire
import logging
from typing import Tuple, List
from time import sleep
from multiprocessing import Pool, cpu_count, Manager
from utils.message_utils import Message
from utils.config_utils import *
from utils.api_utils import OpenAIChatBot

logging.getLogger().setLevel(logging.INFO)

REWRITE_SAMPLE_STR = (
    "来访者：我怎么会做这么愚蠢的事？\n"
    "倾听者：没有人是完美的。你对自己太苛刻了。\n"
    "【特点】倾听者是在安慰来访者。\n\n"
    "来访者：在我看来，所有这些移民，从哪儿来就该送回哪儿去。\n"
    "倾听者：这样的做法是不能解决所有问题的。\n"
    "【特点】倾听者在试图教育来访者。\n\n"
    "来访者：他怎么可以这样和我说话？\n"
    "倾听者：他那样说话，你是不是很伤心？\n"
    "【特点】倾听者在为的来访者感受推卸责任，如果用同理的方式来回应来访者, 倾听者可以说：“你是不是有些伤心，你希望他能同意你的请求？”\n\n"
    "来访者：想到我先生，我就很生气。我需要他的时候，他从来都不在。\n"
    "倾听者：你认为他应该多陪陪你？\n"
    "【特点】我认为倾听者回应了来访者的想法。然而，同理感受和需要而不只是回应想法，更能够促进人与人的连结。如果用同理的方式来回应来访者，倾听者可以说：“听起来，你很生气，因为你希望他能多陪陪你？”\n\n"
    "来访者：我讨厌自己变得越来越胖。\n"
    "倾听者：慢跑也许能帮助你。\n"
    "【特点】倾听者在提建议。\n\n"
    "来访者：当亲戚们不请自来时，我感到被侵犯。这让我想起，我的父母在过去经常无视我的需要，替我做安排。\n"
    "倾听者：我明白你的感受，曾经我也是这样。\n"
    "【特点】倾听者认为自己理解了来访者，并且谈论起自己的感受，而非同理回应来访者的体验。\n\n"
    "来访者：我觉得自己特别失败，什么事情都做不好。\n"
    "倾听者：别这么想，你其实很优秀！\n"
    "【特点】倾听者试图用安慰的方式减轻来访者的痛苦，但没有真正进入来访者的体验，无法形成深层的共鸣。\n\n"
    "来访者：我觉得所有孩子都应该被严格管教，否则无法成才。\n"
    "倾听者：其实现在的教育理念更强调尊重孩子，你的想法有些过时了。\n"
    "【特点】倾听者采取了教育的方式，试图纠正来访者的观点，而非理解来访者的内心感受和需求。\n\n"
    "来访者：我最近总是失眠，心里很焦虑。\n"
    "倾听者：你可以试试睡前喝杯热牛奶，或者做点冥想。\n"
    "【特点】倾听者直接提供解决方案，而没有关注来访者失眠背后的情绪或需求。\n\n"
    "来访者：我觉得自己越来越老了，没什么价值了。\n"
    "倾听者：别瞎想，年龄根本不是问题，心态最重要！\n"
    "【特点】倾听者通过否定来访者的感受试图“打气”，但没有接纳或验证来访者真实的情绪状态。\n\n"
    "来访者：最近我工作很累，觉得压力特别大。\n"
    "倾听者：你还算好的，我前段时间工作比你还忙，天天加班到半夜。\n"
    "【特点】倾听者将话题转移到自己身上，削弱了对来访者的关注和共情。\n\n"
    "来访者：我总是无法专注工作，总是拖延。\n"
    "倾听者：可能是你没有找到正确的目标，或者你对工作不够热爱。\n"
    "【特点】倾听者试图分析来访者的行为原因，而不是去理解来访者的情绪或体验。\n\n"
    "来访者：我觉得自己什么都做不好，真的很无助。\n"
    "倾听者：没关系，失败是成功之母，你一定会走出来的！\n"
    "【特点】倾听者用正能量的语言掩盖了来访者的负面情绪，没有真正倾听和认同来访者的感受。\n\n"
    "来访者：我和我的伴侣最近总是吵架，感觉关系越来越差了。\n"
    "倾听者：其实夫妻之间吵架很正常，过一阵就好了，不用太担心。\n"
    "【特点】倾听者试图淡化问题或迅速解决，而没有深入理解来访者的情绪或需求。\n\n"
    "来访者：我觉得我的朋友最近对我很冷淡，可能不想和我做朋友了。\n"
    "倾听者：也许她最近很忙，或者有其他事情要处理，你不应该太敏感。\n"
    "【特点】倾听者通过逻辑分析试图解释问题，但忽略了来访者的情绪和对友谊的需求。\n\n"
    "来访者：我觉得自己在公司里没有人重视，我很孤独。\n"
    "倾听者：怎么会呢？你明明很优秀，大家一定都很喜欢你。\n"
    "【特点】倾听者直接否定了来访者的感受，没有承认或理解来访者的孤独体验。\n\n"
    "来访者：我觉得和父母交流很困难，他们从来不听我的想法。\n"
    "倾听者：我以前也和你一样，总觉得父母不理解我，后来我学会了换位思考。\n"
    "【特点】倾听者将话题转向自己的经历，缺乏对来访者感受的同理心。\n\n"
    "来访者：我觉得很不公平，为什么领导总是偏心别人？\n"
    "倾听者：生活本来就不公平，你要学会接受现实。\n"
    "【特点】倾听者试图用大道理劝解来访者，但无法让来访者感觉被理解和接纳。\n\n"
)
REWRITE_SAMPLES = REWRITE_SAMPLE_STR.split('\n\n')
model = OpenAIChatBot(model=MODEL_PATH['gpt-4o'])


def template(context, response):
    sample = random.choice(REWRITE_SAMPLES)
    message_content = f"对话历史：\n{context}\n\n参考例句：\n{sample}\n\n原始倾听者回复：{response}\n\n请根据参考例句及其特点，改写倾听者的回复。改写需符合以下要求：\n1. 改写后的回复需与参考例句的特点一致，保持风格相符。\n2. 改写后的回复需与对话历史内容连贯，确保逻辑合理。\n\n改写的回复为：\n"
    return message_content


def api_single_rewrite(args: Tuple[Tuple[Message, int], List[str]]) -> Tuple[int, str]:
    rewrite_desc = "请根据以下要求，改写提供的倾听者回复：\n- 风格相似性：回复应与例句的特点一致，保持相同的风格和表达方式。\n- 对话连贯性：改写后的回复需与对话历史内容相符，确保逻辑通顺且情境合理。\n请严格遵循以上要求进行改写。"
    (message, index), current_end_points = args
    end_point = current_end_points[-1]
    while True:
        rewritten_sent = model.query(
            role_desc=rewrite_desc, history_messages=[message], temperature=0, azure_endpoint=end_point)
        if '<<<<<<END_OF_CONVERSATION>>>>>>' not in rewritten_sent:
            break
        # Rotate endpoints safely with Manager list
        end_point = current_end_points.pop(0)
        current_end_points.append(end_point)
    
    return index, rewritten_sent


def main(batch_size: int = 128):
    end_points = [
        'https://gcraoai5sw1.openai.azure.com/', 'https://gcrgpt4aoai5c.openai.azure.com/',
        'https://gcrgpt4aoai5.openai.azure.com/', 'https://gcraoai5sw2.openai.azure.com/',
        'https://gcraoai5sw3.openai.azure.com/', 'https://gcraoai9sw1.openai.azure.com/',
        'https://gcrgpt4aoai9spot.openai.azure.com/'
    ]
    data = json.load(open(TRAIN_DATA_PATH, 'r', encoding='utf-8'))
    save_dir = 'dataset/PsyDTCorpus_rewritten'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = f'{save_dir}/nonEmp_rewritten.jsonl'
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            rewritten_responses = f.readlines()
            current_len = len(rewritten_responses)
    else:
        current_len = 0
    save_fp = open(f'{save_dir}/nonEmp_rewritten.jsonl', 'a+', encoding='utf-8')
    count = 0
    inputs, outputs = [], []
    for line_index, line in tqdm(enumerate(data), total=len(data)):
        context = line['messages'][1:]
        for turn_id, utterance in enumerate(context):
            if utterance['role'] == 'assistant' and turn_id > 0:
                conv_hist = '\n'.join([f'{l["role"]}: {l["content"]}' for l in context[:turn_id][-5:]])
                response = utterance["content"]
                messages = Message(agent_name='', content=template(conv_hist, response))
                inputs.append((messages, count))
                outputs.append({
                    'conv_id': line['id'],
                    'normalizedTag': line['normalizedTag'],
                    'turn_id': turn_id + 1,
                    'messages': line['messages'][:turn_id + 1],
                    'context': conv_hist,
                    'response': response,
                })
                count += 1
    with Manager() as manager:
        for i in tqdm(range(current_len, len(inputs), batch_size), desc='Rewriting', position=0,
                      leave=False):
            args = inputs[i: i + batch_size]
            with Pool(cpu_count()) as pool:
                result_list = list(tqdm(
                    pool.imap_unordered(
                        api_single_rewrite,
                        [(arg, random.sample(list(end_points), len(end_points))) for arg in args]
                    ),
                    desc='Inferring batches',
                    position=1,
                    leave=False,
                    total=len(args)
                ))
            result_list.sort(key=lambda x: x[0])
            for result in result_list:
                index, rewritten_response = result
                res_line = outputs[index]
                res_line['NonEmp-response'] = rewritten_response
                save_fp.write(json.dumps(res_line, ensure_ascii=False) + '\n')
                logging.info(f'Original response: {res_line["response"]}')
                logging.info(f'Rewritten response: {rewritten_response}')
    save_fp.close()


if __name__ == '__main__':
    Fire(main)
