import logging
import os

from prometheus_client.decorator import append
from tqdm import tqdm
from utils.config_utils import *
import string

punctuation = '，。！？【】（）（）<>“”‘’：；、|《》' + string.punctuation + string.whitespace
logging.getLogger().setLevel(logging.INFO)


def user_simulator_dataset():
    """
    this function should be processed after reasoning the user description.
    :return:
    """
    with open(f'{ROOT_DIR}/datasets/PsyDTCorpus/PsyDTCorpus_train_user_description.jsonl', 'r') as fp:
        dataset = [json.loads(line) for line in fp.readlines()]
    with open(USER_STATE_PATH, 'r') as fp:
        user_states = [json.loads(line)['user_states'] for line in fp.readlines()]
    
    user_simulator_data, end_of_conversation = [], []
    for idx, line in tqdm(enumerate(dataset), total=len(dataset)):
        messages = line['messages'][1:]
        user_state_dict = {list(lst.keys())[0]: list(lst.values())[0] for lst in user_states[idx]}
        instruction = f'【任务】：根据历史对话和来访者自我描述，推理对话过程中来访者的观测事实、感受、需求和请求（状态信息）。并请扮演来访者和倾听者进行对话，在对话过程中倾诉自己的烦恼。\n'
        
        for turn_id, msg in enumerate(messages):
            if msg['role'] == 'user':
                # load the user state
                input_text = f'【来访者自我描述】：{line["description"]}\n\n'
                state_content = user_state_dict.get(str(turn_id + 1), '')
                if state_content:
                    fact_aspect = [k for k, v in state_content['确认内容'].items() if v != []]
                    inference_aspect = [k for k, v in state_content['猜测内容'].items() if v != []]
                    state_fact = '\n'.join(
                        [f'\t\t{k}： {"".join(state_content["确认内容"][k]).strip(punctuation)}' for k in fact_aspect])
                    state_inference = '\n'.join(
                        [f'\t\t{k}: {"".join(state_content["猜测内容"][k]).strip(punctuation)}' for k in
                         inference_aspect])
                    fact_str = f'来访者在对话中讲述的{"、".join(fact_aspect)}如下：\n{state_fact}。' if fact_aspect != [] else ''
                    inference_str = f'来访者心里隐含的{"、".join(inference_aspect)}如下：\n{state_inference}。' if inference_aspect != [] else ''
                    user_state = '\n\t'.join(['【来访者感受认知】：', fact_str, inference_str])
                else:
                    user_state = ''
                if turn_id == len(messages) - 2:
                    user_state += '\n\t来访者认为对话可以结束。'
                # load the input text
                if len(messages[:turn_id]) > 0:
                    # add the conversation history
                    input_text += '【历史对话】：\n\t' + '\n\t'.join(
                        [f'[{ROLE_MAP[msg["role"]]}]：{msg["content"]}' for msg in messages[:turn_id]])
                    input_text += '\n\n请输出来访者接下来来访者的状态信息（如有）和回复。\n'
                    # prepare the user state and the response.
                    # the model is to predict the user state and the response given the dialogue history and the description.
                    if user_state:
                        output_text = f"{user_state}\n\n【来访者对话】：{msg['content']}"
                    else:
                        output_text = f"【来访者对话】：{msg['content']}"
                else:
                    input_text += '请输出来访者的开场对话。\n'
                    output_text = f"【来访者对话】：{msg['content']}"
                # add the user response
                reply_line = {
                    'conv_id': line['id'],
                    'turn_id': turn_id,
                    'normalizedTag': line['normalizedTag'],
                    'conversation': messages[:turn_id],
                    'description': line['description'],
                    'instruction': instruction,
                    'input': input_text,
                    'output': output_text,
                }
                if user_state:
                    reply_line['user_state'] = user_state
                user_simulator_data.append(reply_line)
        input_text = f'【来访者自我描述】：{line["description"]}\n\n'
        input_text += '【历史对话】：\n\t' + '\n\t'.join(
            [f'[{ROLE_MAP[msg["role"]]}]：{msg["content"]}' for msg in messages])
        input_text += '\n\n请输出来访者接下来来访者的状态信息（如有）和回复。\n'
        user_simulator_data.append({
            'conv_id': line['id'],
            'turn_id': len(messages),
            'normalizedTag': line['normalizedTag'],
            'conversation': messages,
            'description': line['description'],
            'instruction': instruction,
            'input': input_text,
            'output': "【来访者对话】：EOC"
        })
        end_of_conversation.append({
            'conv_id': line['id'],
            'turn_id': len(messages),
            'normalizedTag': line['normalizedTag'],
            'conversation': messages,
            'description': line['description'],
            'instruction': instruction,
            'input': input_text,
            'output': "【来访者对话】：EOC"
        })
    
    save_dir = f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_train'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'user_simulator_train.json'), 'w', encoding='utf-8') as fp:
        json.dump(user_simulator_data, fp, indent=4)
    logging.info(f'There are {len(user_simulator_data)} lines in the user simulator dataset.')
    with open(os.path.join(save_dir, 'user_eoc_train.json'), 'w', encoding='utf-8') as fp:
        json.dump(end_of_conversation, fp, indent=4)
    logging.info(f'There are {len(end_of_conversation)} lines in the end of conversation dataset.')


def non_nvc_dataset():
    with open('dataset/PsyDTCorpus_rewritten/nonEmp_rewritten.jsonl', 'r', encoding='utf-8') as fp:
        dataset = fp.readlines()
    non_nvc_data = []
    for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
        data = json.loads(data)
        messages = data['messages'][1:]
        non_nvc_data.append({
            'conv_id': data['conv_id'],
            'turn_id': data['turn_id'],
            'conversation': messages,
            'response': data['response'],
            'contrast': data['NonEmp-response'],
            'label': 'nvc'
        })
        non_nvc_data.append({
            'conv_id': data['conv_id'],
            'turn_id': data['turn_id'],
            'conversation': messages,
            'response': data['NonEmp-response'],
            'contrast': data['response'],
            'label': 'non-nvc'
        })
    
    save_dir = 'dataset/PsyDTCorpus_train'
    with open(os.path.join(save_dir, 'non-nvc_train.json'), 'w', encoding='utf-8') as fp:
        json.dump(non_nvc_data, fp, indent=4)


def user_state_dataset():
    """
    This function should be run after the function ``reason_user_state()``.
    The processed data can be saved in the file 'dataset/scored_PsyDTCorpus_train/train.json'.

    :return: processed data is in the format {"conversation", "response", "user_state"}
    conversation: the conversation context,
    response: the response the assistant should give,
    user_state: the user state inferred from the conversation context that can be used to generate the response.
    """
    with open(USER_STATE_PATH, 'r') as fp:
        user_state_data = fp.readlines()
        user_state_data = [json.loads(line) for line in user_state_data]
    save_dir = 'dataset/PsyDTCorpus_train'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    training_data = []
    for idx, data in tqdm(enumerate(user_state_data), total=len(user_state_data)):
        messages = data['messages'][1:]
        user_states_list = data['user_states']
        for turn_id, turn_line in enumerate(messages):
            if turn_line['role'] == 'user' or turn_id == 0:
                continue
            training_line = {
                'conv_id': idx,
                'turn_id': turn_id,
                'conversation': messages[:turn_id],
                'response': turn_line['content']
            }
            if str(turn_id) in user_states_list[0].keys():
                user_state = user_states_list.pop(0)
                state_content = user_state[str(turn_id)]
                fact_aspect = [k for k, v in state_content['确认内容'].items() if v != []]
                inference_aspect = [k for k, v in state_content['猜测内容'].items() if v != []]
                state_fact = '\n'.join([f'{k}: {"".join(state_content["确认内容"][k])}' for k in fact_aspect])
                state_inference = '\n'.join(
                    [f'{k}: {"".join(state_content["猜测内容"][k])}' for k in inference_aspect])
                fact_str = f'根据来访者和倾听者的对话，来访者当下的{"、".join(fact_aspect)}如下：\n{state_fact}。' if state_fact != [] else ''
                inference_str = f'根据分析，来访者可能有以下的{"、".join(inference_aspect)}：\n{state_inference}。' if state_inference != [] else ''
                training_line['user_state'] = '\n'.join([fact_str, inference_str])
            training_data.append(training_line)
    with  open(os.path.join(save_dir, 'user_state_train.json'), 'w', encoding='utf-8') as fp:
        json.dump(training_data, fp, indent=4)


def cot_training_dataset():
    aspects = ["观察：", "感受：", "需求：", "请求："]
    
    with open(f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_cot/response_cot.jsonl', 'r') as f:
        dataset = [json.loads(line) for line in f.readlines()]
    user_state_dataset = json.load(
        open(f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_train/contrastive_train.json', 'r'))
    
    json_dataset = []
    for i, line in tqdm(enumerate(dataset), total=len(dataset), desc='Processing vanilla cot response data'):
        user_state = user_state_dataset.pop(0).get('user_state', '')
        if line['label'] == 'dataset':
            assert i > len(dataset) - 2 or dataset[i + 1]['label'] == 'rewritten'
            cot = line['cot'].replace('【倾听者思维链】为：', '').replace('```markdown', '').replace('```', '')
            if '我对来访者有如下判断' in cot:
                # extract the user state
                pattern = r"我对来访者有如下判断：(.*?)在接下来的回复中"
                match = re.search(pattern, cot, re.DOTALL)
                user_state = match.group(1).strip()
                # extract the strategy
                pattern = r"在接下来的回复中，(.*)"
                match = re.search(pattern, cot, re.DOTALL)
                strategy = "在接下来的回复中，" + match.group(1).strip()
            else:
                strategy = cot.strip()
            user_state = user_state.replace(":", "：")
            explicit, implicit = '', ''
            if "来访者对于问题显示出来的认知或感受：" in user_state and "对于问题隐藏的认知或感受：" in user_state:
                pattern = r"来访者对于问题显示出来的认知或感受：(.*?)(?=对于问题隐藏的认知或感受：)"
                match = re.search(pattern, user_state, re.DOTALL)
                explicit = match.group(1).strip()
                implicit = user_state.split("对于问题隐藏的认知或感受：")[1].strip()
            elif "来访者对于问题显示出来的认知或感受：" in user_state:
                explicit = user_state.split("来访者对于问题显示出来的认知或感受：")[1].strip()
            elif "对于问题隐藏的认知或感受：" in user_state:
                implicit = user_state.split("对于问题隐藏的认知或感受：")[1].strip()
            
            user_state = ''
            if explicit:
                pattern = r"(" + "|".join(re.escape(keyword) for keyword in aspects) + r")(.*?)(?=(" + "|".join(
                    re.escape(keyword) for keyword in aspects) + r"|$))"
                matches = re.finditer(pattern, explicit, re.DOTALL)
                result = {}
                for match in matches:
                    keyword = match.group(1).strip(punctuation)
                    content = match.group(2).strip(punctuation)
                    result[keyword] = content
                user_state += f'\t来访者显示出的{"、".join(list(result.keys()))}：\n'
                user_state += '\n'.join([f'\t\t{keyword}：{content}' for keyword, content in result.items()])
                user_state += '\n'
            if implicit:
                pattern = r"(" + "|".join(re.escape(keyword) for keyword in aspects) + r")(.*?)(?=" + "|".join(
                    re.escape(keyword) for keyword in aspects) + r"|$)"
                matches = re.finditer(pattern, implicit, re.DOTALL)
                result = {}
                for match in matches:
                    keyword = match.group(1).strip(punctuation)
                    content = match.group(2).strip(punctuation)
                    result[keyword] = content
                user_state += f'\t来访者可能有的{"、".join(list(result.keys()))}：\n'
                user_state += '\n'.join([f'\t\t{keyword}：{content}' for keyword, content in result.items()])
            
            if i + 2 > len(dataset) or dataset[i + 2]['conv_id'] != line['conv_id']:
                user_state += '\n\t来访者认为对话可以结束。'
                strategy = '在接下来的回复中，我将结束对话并且和用户告别。'
            
            if user_state:
                cot = f'我对来访者有如下判断：\n{user_state}\n\n{strategy}'
            else:
                cot = strategy
            conversation = '\n\t'.join([f'{msg["role"]}: {msg["content"]}' for msg in line['conversation']])
            
            json_dataset.append({
                'conv_id': line['conv_id'],
                'turn_id': line['turn_id'],
                'instruction': '【任务】：请根据倾听者和来访者的历史对话，生成倾听者的思维链和相应的倾听者回复。\n',
                'input': f'【历史对话】：\n\t{conversation}\n\n请判断倾听者思维链，并做出对来访者的回复。\n',
                'output': f'【倾听者思维链】：{cot}\n\n【倾听者回复】：{line["response"]}'
            })
    
    with open(f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_train/cot_vanilla_train.json', 'w') as fp:
        json.dump(json_dataset, fp, indent=4)


def contrastive_data():
    with open(USER_STATE_PATH, 'r') as fp:
        user_state_dataset = [json.loads(line) for line in fp.readlines()]
    logging.info(f'There are {len(user_state_dataset)} conversations in the dataset.')
    with open('dataset/PsyDTCorpus_rewritten/nonEmp_rewritten.jsonl', 'r', encoding='utf-8') as fp:
        non_empathetic_dataset = [json.loads(line) for line in fp.readlines()]
    
    training_dataset = []
    for conv_id, datapoint in tqdm(enumerate(user_state_dataset), total=len(user_state_dataset)):
        messages = datapoint['messages'][1:]
        user_states = datapoint['user_states']
        for idx, msg in enumerate(messages):
            if msg['role'] == 'user':
                continue
            response = msg['content']
            empathetic_line = {
                'conv_id': datapoint['id'],
                'turn_id': idx,
                'normalizedTag': datapoint['normalizedTag'],
                'conversation': messages[:idx],
                'response': response,
                'label': 'dataset'
            }
            non_empathetic_data = non_empathetic_dataset.pop(0)
            assert non_empathetic_data[
                       'response'] == response, f'{non_empathetic_data["response"]} != {response}\n{datapoint}\n{non_empathetic_data}'
            non_empathetic_line = {
                'conv_id': datapoint['id'],
                'turn_id': idx,
                'normalizedTag': datapoint['normalizedTag'],
                'conversation': messages[:idx],
                'response': non_empathetic_data['NonEmp-response'],
                'label': 'rewritten'
            }
            
            if idx > 2:
                reference_state = user_states.pop(0)
                assert idx == int(list(reference_state.keys())[0]), f'{idx} != {int(list(reference_state.keys())[0])}'
                user_state_dict = list(reference_state.values())[0]
                confirmed = '\n'.join(
                    [f"\t\t- {k}: {'。'.join(v)}。" for k, v in user_state_dict['确认内容'].items() if len(v) > 0])
                inferred = '\n'.join(
                    [f"\t\t- {k}: {'。'.join(v)}。" for k, v in user_state_dict['猜测内容'].items() if len(v) > 0])
                current_user_state = f'\t来访者对于问题显示出来的认知或感受：\n{confirmed}\n\t对于问题隐藏的认知或感受：\n{inferred}'
                empathetic_line['user_state'] = current_user_state.replace("。。", "。")
                if len(non_empathetic_dataset) > 0:
                    non_empathetic_line['user_state'] = current_user_state.replace("。。", "。")
            
            training_dataset.append(empathetic_line)
            if len(non_empathetic_dataset) > 0:
                training_dataset.append(non_empathetic_line)
    save_path = f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_train/contrastive_train.json'
    json.dump(training_dataset, open(save_path, 'w', encoding='utf-8'), indent=4)


def main(function: str):
    if function == 'process_user_simulator':
        user_simulator_dataset()
    elif function == 'process_vanilla_train':
        logging.info('The vanilla_train [Empathetic LLM training] includes the user states.')
        logging.info('Run at 4:10PM 12/12/2024')
        run = input('Input Yes if you want to run this code:\n')
        if run.strip() == 'Yes':
            user_state_dataset()
    elif function == 'process-neg':
        logging.info('Run at 4:10PM 12/12/2024')
        run = input('Input Yes if you want to run this code:\n')
        if run.strip() == 'Yes':
            non_nvc_dataset()
    elif function == 'process_vanilla_cot':
        cot_training_dataset()
    elif function == 'process_contrastive':
        logging.info('Run at 4:10PM 12/12/2024')
        run = input('Input Yes if you want to run this code:\n')
        if run.strip() == 'Yes':
            contrastive_data()


if __name__ == '__main__':
    # Fire(main)
    # user_simulator_dataset()
    cot_training_dataset()
