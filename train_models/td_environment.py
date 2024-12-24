import random
from typing import Optional, Dict, Any
import time
import logging
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from llamafactory.hparams import get_train_args, get_eval_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from utils.template_utils import get_template
import concurrent.futures

logging.getLogger().setLevel(logging.CRITICAL)
# openai.util.logger.setLevel(logging.WARNING)
PROMPT_TEMPLATE = 'You are playing a game called twenty questions with me. The rule of twenty question is that you are given a hidden word, and I am guessing what the word is within twenty questions. For every question, if it is an invalid question, you should answer "Invalid Question.". For any valid question, you should answer either "Yes." or "No.". Now the hidden word given to you is "{word}", and the question for the current round is "{question}". Your response is:'
DEFAULT_OBJECT_DICT = {
    "Sports": ["Basketball", "Football", "Baseball", "Soccer ball", "Golf ball", "Tennis ball", "Volleyball",
               "Tennis racket", "Baseball bat", "Helmet"],
    "Animals": ["Cat", "Dog", "Horse", "Cow", "Sheep", "Rabbit", "Lion", "Tiger", "Bear", "Elephant"],
    "Fruits": ["Apple", "Banana", "Orange", "Strawberry", "Grape", "Watermelon", "Pineapple", "Mango", "Cantaloupe",
               "Peach"],
    "Vehicles": ["Car", "Truck", "Motorcycle", "Boat", "Airplane;Plane", "Train", "Bus", "Helicopter", "Scooter",
                 "Ship"],
    "Clothes": ["Shirt", "Pants;Pant;Pair of pants", "Jacket", "Dress", "Skirt", "Belt", "Shoes;Shoe;Pair of shoes",
                "Boots;Boot;Pair of boots", "Socks;Sock;Pair of socks", "Hat", "Scarf"],
    "Electronics": ["Computer", "Smartphone", "Television;TV", "Headphone;Headphones;Pair of headphones",
                    "Monitor;Computer monitor", "Camera", "Microwave;Microwave oven", "Refrigerator", "Blender",
                    "Computer keyboard;Keyboard"],
    "Musical Instruments": ["Piano", "Guitar", "Drum;Drums", "Violin", "Saxophone", "Flute", "Trumpet", "Clarinet",
                            "Harp", "Trombone"],
    "Furniture": ["Chair", "Table", "Bed", "Desk", "Couch", "Dresser", "Bookcase", "Nightstand", "Mattress", "Pillow"],
    "Office Supplies": ["Pen", "Paper;Piece of paper", "Stapler", "Printer", "Calculator",
                        "Battery;Battery pack;Pack of batteries", "Toothbrush", "Toothpaste", "Pencil", "Sharpie",
                        "Scissors;Pair of scissors", "Key", "Diary", "Calendar"],
    "Vegetables": ["Carrot", "Potato", "Broccoli", "Tomato", "Onion", "Spinach", "Corn", "Peas;Pea", "Celery",
                   "Cucumber"],
    "Art": ["Painting;Canvas painting;Oil painting;Watercolor painting", "Paintbrush", "Canvas;Painting canvas",
            "Eraser;Pencil eraser", "Marker", "Glue;Glue stick;Bottle of glue", "Sculpture"],
    "Kitchen Tools": ["Knife", "Spoon", "Fork", "Plate", "Bowl", "Cooking pot;Pot", "Pan;Saucepan;Frying pan", "Cup",
                      "Chopstick;Chopsticks;Pair of chopsticks", "Whisk"],
    "Nature": ["Rock", "Tree", "Bush", "Mountain", "Forest", "Ocean", "Sea", "Lake", "River", "Meteorite", "Cactus"],
    "Toys": ["Lego;Lego set", "Doll;Toy doll;Plush doll", "Kite", "Puzzle;Jigsaw puzzle"],
    "Jewelry": ["Earring;Earrings;Pair of earrings", "Necklace", "Bracelet", "Ring", "Brooch", "Hairclip", "Pendant",
                "Watch", "Locket"],
    "Garden Supplies": ["Gloves;Glove;Pair of gloves", "Shovel", "Rake", "Watering can", "Lawn mower"],
    "Tools": ["Hammer", "Screwdriver", "Wrench", "Saw", "Pliers;plier;Pair of pliers", "Drill"]
}

DEFAULT_OBJECT_LIST = sum([d for d in DEFAULT_OBJECT_DICT.values()], [])
# random.seed(42)
# DEFAULT_OBJECT_LIST = random.sample(DEFAULT_OBJECT_LIST, k=5)
# DEFAULT_OBJECT_LIST = [DEFAULT_OBJECT_LIST[i] for i in [1,11,21,31,41,51,61,71,81,91]]
INITIAL_STR = "Questions:\n"


class TwentyQuestionsEnv():
    def __init__(
            self,
            word_list: list = DEFAULT_OBJECT_LIST,
            max_conversation_length: int = 20,
    ):
        self.word_list = word_list
        self.word_list = [list(map(lambda x: x.lower(), word.split(";"))) for word in self.word_list]
        self.max_conversation_length = max_conversation_length
        self.random = random.Random(None)
        self.count = 0
        self.curr_word = None
        self.history = ''
        self.done = True
    
    def is_correct(self, question):
        # check for the last word
        # cut out punctuations at the end
        while len(question) > 0 and not question[-1].isalpha():
            question = question[:-1]
        
        if len(question) == 0:
            return False
        guess = question.split(" ")[-1].lower()
        return guess in self.curr_word
    
    def _step(self, question, answer):
        if 'yes' in answer.strip().lower():
            answer = 'Yes.'
        elif 'no' in answer.strip().lower():
            answer = 'No.'
        else:
            # print("question:  " + question)
            # print('answer: '+ answer)
            # import IPython; IPython.embed()
            answer = 'Invalid Question.'
        if self.done:
            return None
        self.count += 1
        self.history += question + ' ' + answer + '\n'
        
        # trajectory = create_trajectory_from_history(self.curr_word, text_history + (answer_text,), self.max_conversation_length)
        # if self.count == self.max_conversation_length:
        #     print("The word was", self.curr_word[0])
        done = (answer.replace('.', '').lower() == 'yes') and self.is_correct(question)
        reward = -1
        if done:
            reward = 0
        self.done = done or self.count == self.max_conversation_length
        return self.history, reward, self.done
    
    def reset(self, idx: Optional[int] = None):
        self.count = 0
        # if self.curr_word is not None:
        #     print("The word was ", self.curr_word)
        #     print("Next word...")
        if idx is not None:
            self.curr_word = self.word_list[idx]
        else:
            self.curr_word = self.random.choice(self.word_list)
        self.history = INITIAL_STR
        self.done = False
        return INITIAL_STR
        # return (Text(INITIAL_STR, is_action=False),)
    
    def copy(self):
        return TwentyQuestionsEnv(
            oracle=self.oracle,
            word_list=self.word_list,
            max_conversation_length=self.max_conversation_length,
        )


class BatchedTwentyQuestionsEnv():
    def __init__(
            self,
            env_load_path: str,
            cache_dir: str,
            device,
            max_conversation_length: int = 20,
            bsize: int = 32,
    ):
        self.env_list = [TwentyQuestionsEnv(max_conversation_length) for _ in range(bsize)]
        self.bsize = bsize
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", cache_dir=cache_dir).to(device)
        self.model.load_state_dict(torch.load(env_load_path)['model_state_dict'])
    
    def generate_answers(self, questions):
        curr_words = [env.curr_word[0].lower() for env in self.env_list]
        inputs = [f"The object is {curr_word}." + question for curr_word, question in zip(curr_words, questions)]
        encoder_ids = self.tokenizer(inputs, padding=True, return_tensors='pt').to(self.model.device)
        return self.tokenizer.batch_decode(
            self.model.generate(input_ids=encoder_ids['input_ids'], attention_mask=encoder_ids['attention_mask'], \
                                max_new_tokens=16, do_sample=False), skip_special_tokens=True)
    
    def reset(self, idx: Optional[int] = None):
        return [env.reset(idx) for env in self.env_list]
    
    def step(self, questions):
        answers = self.generate_answers(questions)
        # print("Step once!")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(env._step, q, a) for env, q, a in zip(self.env_list, questions, answers)]
            results = [job.result() for job in jobs]
        return results
