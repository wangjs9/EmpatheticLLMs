from tqdm import tqdm
from typing import List, Dict, Any, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
from utils.config_utils import *

logging.getLogger().setLevel(logging.INFO)


class Node:
    def __init__(
            self, listener: str = None, model_cot: str = None, user_react: str = None, turn_num: int = 1,
            parent: Optional['Node'] = None, children: List['Node'] = None
    ):
        """
        
        there are three types of nodes:
        - root: the first node in the dialogue.
            content is None, user_react is xxx.
        - mid node: the nodes that have both parent and children.
            both the content and user_react are xxx.
        - leave: the nodes that do not have children.
            content is xxx, user_react is None
            
        the model cot has the same type with the listener.
        """
        self.listener = listener.replace('\n', ' ') if type(listener) == str else None
        if self.listener and self.listener.startswith('】：'):
            self.listener = self.listener.split('】：')[-1]

        self.model_cot = model_cot
        assert type(self.model_cot) == type(self.listener), f'Listen: {self.listener}\nModel COT: {self.model_cot}'
        if user_react and 'EOC' in user_react:
            self.user_react = None
            self.success = 1.0
        else:
            self.user_react = user_react.replace('\n', ' ') if user_react else None
        self.turn_num = turn_num
        self.parent = parent
        self.children = children if children else []

    def __str__(self):
        """Returns: the dialogue from the root parent to this node."""
        if not self.listener:
            return f'来访者：{self.user_react}'
        elif not self.user_react:
            return str(self.parent) + f'\n倾听者：{self.listener}'
        else:
            return str(self.parent) + f'\n倾听者：{self.listener}\n来访者：{self.user_react}'

    def to_dict(self):
        if not self.listener:
            return [{'content': self.user_react, 'role': 'user'}]
        elif not self.user_react:
            return self.parent.to_dict() + [{'content': self.listener, 'role': 'assistant'}]
        else:
            return (self.parent.to_dict() +
                    [{'content': self.listener, 'role': 'assistant'}, {'content': self.user_react, 'role': 'user'}])

    def success_rate(self):
        if not hasattr(self, 'success'):
            if len(self.children) == 0:
                return 0
            self.success = sum([child.success_rate() for child in self.children]) / len(self.children)
        return self.success

    def add_child(self, new_utterance: 'Node'):
        if str(new_utterance.parent) != str(self):
            raise ValueError('The child is not the next turn.')
        if Node in self.children:
            raise ValueError('This Node already has this child.')
        self.children.append(new_utterance)

    def find_or_add_child(
            self, listener: str = None, model_cot: str = None, user_react: str = None, turn_num: int = None
    ):
        for child in self.children:
            if child.listener == listener and child.model_cot == model_cot and child.user_react == user_react and child.turn_num == turn_num:
                return child

        new_child = Node(
            listener=listener, model_cot=model_cot, user_react=user_react, turn_num=self.turn_num + 1, parent=self
        )

        self.add_child(new_child)
        return new_child


def build_tree_from_trajectories(root: Node, trajectories: List[List[Dict[str, str]]], cots: List[List[str]]):
    for turn_id, (cot_list, traj) in enumerate(zip(cots, trajectories)):
        current = root
        # the turn number is the index of the trajectory plus 2 (1: root).
        assert len(cot_list) == len(traj) // 2, f'{len(cot_list)} != {len(traj) // 2}'
        finished = False
        for i in range(len(traj[1:-1:2])):
            if traj[i * 2 + 1]['content'] == '':
                print('\n'.join([f'{line["content"]}\n' for line in traj]))
                print("***************")
                finished = True
                current.success = 0.0
                break
            current = current.find_or_add_child(
                traj[i * 2 + 1]['content'].replace('\n', ''),
                cot_list[i],
                traj[i * 2 + 2]['content'].replace('\n', ''),
                i + 2
            )
        if traj[-2]['content'] != 'EOC' and not finished:
            current.add_child(Node(
                listener=traj[-1]['content'].replace('\n', ''),
                model_cot=cot_list[-1],
                turn_num=len(traj) // 2 + 1,
                parent=current
            ))

    return root


def build_comparison_pair(root: Node) -> List[Dict[str, Any]]:
    """responses with the same context. Thus, we only compare the children of the root node."""
    children = root.children
    context = root.to_dict()
    if context[-1]['role'] == 'assistant':
        context.pop(-1)
    success_rate = {child.listener: child.success_rate() for child in children}
    children_cots = {child.listener: child.model_cot for child in children}
    sorted_responses = sorted(success_rate, key=success_rate.get, reverse=True)
    sorted_rate = [success_rate[response] for response in sorted_responses]
    sorted_cots = [children_cots[response] for response in sorted_responses]
    comparison_pair = []
    gap = 1 if len(context) > 3 and len(context) < 47 else 100
    for i in range(len(sorted_responses)):
        for j in range(i + gap, len(sorted_responses)):
            if sorted_rate[i] > sorted_rate[j]:
                if ('【倾听者思维链】：我对来访者有如下判断：' not in sorted_cots[i] or '在接下来的回复中' not in
                        sorted_cots[i]):
                    logging.info(f'At least one of the cots are not valid:\n\t{sorted_cots[i]}\n\t{sorted_cots[j]}')
                    continue
                comparison_pair.append({
                    'conversation': context,
                    'chosen': f'{sorted_cots[i].strip()}\n\n【倾听者回复】：{sorted_responses[i]}',
                    'rejected': f'{sorted_cots[j].strip()}\n\n【倾听者回复】：{sorted_responses[j]}'
                })

    return comparison_pair


def get_pair_from_tree(root: Node):
    comparison_pair = build_comparison_pair(root)
    if root.children:
        child_results = [get_pair_from_tree(child) for child in root.children]
        for result in child_results:
            comparison_pair.extend(result)
    return comparison_pair


def load_dialogues():
    data_dir = f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_rewards'
    path_list = sorted(os.listdir(data_dir))
    # path_list = [path for path in path_list if int(path.split('_')[1][:-5]) < 64]
    mcts_writer = open(f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_rewards/mcts_dialogues.jsonl', 'w')
    for path in tqdm(path_list, total=len(path_list)):
        if path.endswith('.jsonl'):
            continue
        abs_path = os.path.join(data_dir, path)
        instance = json.load(open(abs_path, 'r', encoding='utf-8'))

        # get the dialogues and cots
        completed_dialogue = [conv['dialogue'] for conv in instance['completed']]
        completed_cot = [conv['predicted_state'] for conv in instance['completed']]
        other_dialogue = [conv['dialogue'] for conv in instance['trajectories']]
        other_cot = [conv['predicted_state'] for conv in instance['trajectories']]
        discard_dialogue = [conv['dialogue'] for conv in instance['discard']]
        discard_cot = [conv['predicted_state'] for conv in instance['discard']]

        # build the tree from the trajectories
        first_user = other_dialogue[0][0]['content']
        dialogue_root = Node(turn_num=1, user_react=first_user)
        build_tree_from_trajectories(dialogue_root, completed_dialogue, completed_cot)
        build_tree_from_trajectories(dialogue_root, other_dialogue, other_cot)
        build_tree_from_trajectories(dialogue_root, discard_dialogue, discard_cot)

        # obtain the comparison pair
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(get_pair_from_tree, dialogue_root.children))
        mcts_dialogues = []
        for result in results:
            mcts_dialogues.extend(result)
        for line in mcts_dialogues:
            mcts_writer.write(json.dumps(line) + '\n')
    mcts_writer.close()


def format_training_data():
    with open(
            f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_rewards/mcts_dialogues.jsonl', 'r', encoding='utf-8'
    ) as fp:
        dataset = fp.readlines()

    formatted_data = []
    for line in tqdm(dataset, total=len(dataset)):
        line = json.loads(line)
        conversation = '\n\t'.join([f'{ROLE_MAP[turn["role"]]}: {turn["content"]}' for turn in line['conversation']])
        chosen = line['chosen']
        rejected = line['rejected']
        if "\n完整的\n" in chosen:
            chosen = chosen.replace("\n完整的\n", '')
        if "\n完整的\n" in rejected:
            rejected = rejected.replace("\n完整的\n", '')

        if '\n\n【倾听者回复】：' not in chosen:
            chosen_list = chosen.split('\n\n')
            chosen = (f'{chosen_list[0]}\n\n' +
                      ' '.join(chosen_list[1:-1]).replace('\n', '') + f'\n\n【倾听者回复】：{chosen_list[-1]}')

        if '\n\n【倾听者回复】：' not in rejected:
            rejected_list = rejected.split('\n\n')
            rejected = (f'{rejected_list[0]}\n\n' +
                        ' '.join(rejected_list[1:-1]).replace('\n', '') + '\n\n【倾听者回复】：' + \
                        rejected_list[-1])

        formatted_data.append({
            'instruction': '【任务】：请根据倾听者和来访者的历史对话，生成倾听者的思维链和相应的倾听者回复。\n',
            'input': f'【历史对话】：\n\t{conversation}\n\n请判断倾听者思维链，并作出对来访者的回复。\n',
            'chosen': chosen,
            'rejected': rejected
        })

    json.dump(
        formatted_data,
        open(f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_train/dpo_simulated_train.json', 'w', encoding='utf-8'),
        ensure_ascii=False, indent=4
    )


if __name__ == '__main__':
    # load_dialogues()
    format_training_data()
