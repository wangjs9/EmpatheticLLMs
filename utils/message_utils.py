import time
import hashlib
from typing import List, Union
from dataclasses import dataclass
from uuid import uuid1


def _hash(input: str):
    """
    Helper function that generates a SHA256 hash of a given input string.

    Parameters:
        input (str): The input string to be hashed.

    Returns:
        str: The SHA256 hash of the input string.
    """
    hex_dig = hashlib.sha256(input.encode()).hexdigest()
    return hex_dig


@dataclass
class Message:
    content: str
    agent_name: str = ''
    turn: int = -1
    timestamp: int = time.time_ns()
    visible_to: Union[str, List[str]] = "all"
    msg_type: str = "text"
    logged: bool = False  # Whether the message is logged in the database

    @property
    def msg_hash(self):
        # Generate a unique message id given the content, timestamp and role
        return _hash(
            f"agent: {self.agent_name}\ncontent: {self.content}\ntimestamp: {str(self.timestamp)}\nturn: {self.turn}\nmsg_type: {self.msg_type}"
        )

    def __str__(self):
        """Print all the messages in the pool."""
        string = f'[{self.agent_name}->{self.visible_to}]: {self.content}'
        return string


class MessagePool:
    """
    A pool to manage the messages in the chatArena environment.

    The pool is essentially a list of messages, and it allows a unified treatment of the visibility of the messages.
    It supports two configurations for step definition: multiple players can act in the same turn (like in rock-paper-scissors).
    Agents can only see the messages that 1) were sent before the current turn, and 2) are visible to the current role.
    """

    def __init__(self):
        """Initialize the MessagePool with a unique conversation ID."""
        self.conversation_id = str(uuid1())
        self._messages: List[Message] = []
        self._last_message_idx = 0

    def reset(self):
        """Clear the message pool."""
        self._messages = []

    def append_message(self, message: Message):
        """
        Append a message to the pool.

        Parameters:
            message (Message): The message to be added to the pool.
        """
        self._messages.append(message)

    def __str__(self):
        """Print all the messages in the pool."""
        string = ''
        for message in self._messages:
            string += str(message) + '\n'
        return string

    @property
    def last_turn(self):
        """
        Get the turn of the last message in the pool.

        Returns:
            int: The turn of the last message.
        """
        if len(self._messages) == 0:
            return 0
        else:
            return self._messages[-1].turn

    @property
    def last_message(self):
        """
        Get the last message in the pool.

        Returns:
            Message: The last message.
        """
        if len(self._messages) == 0:
            return None
        else:
            return self._messages[-1]

    def get_all_messages(self) -> List[Message]:
        """
        Get all the messages in the pool.

        Returns:
            List[Message]: A list of all messages.
        """
        return self._messages

    def get_visible_messages(self, agent_name, turn: int) -> List[Message]:
        """
        Get all the messages that are visible to a given agent before a specified turn.

        Parameters:
            agent_name (str): The name of the agent.
            turn (int): The specified turn.

        Returns:
            List[Message]: A list of visible messages.
        """

        # Get the messages before the current turn
        prev_messages = [message for message in self._messages if message.turn < turn]

        visible_messages = []
        for message in prev_messages:
            if message.visible_to == "all" or agent_name in message.visible_to:
                visible_messages.append(message)
        return visible_messages
