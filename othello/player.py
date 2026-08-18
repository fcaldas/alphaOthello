import abc
from .board import Board, Color

class Player:

    def __init__(self, color: Color):
        self.color = color

    @abc.abstractmethod
    def play(self, board: Board) -> bool:
        raise NotImplementedError("Abstract class")
    
    def get_color(self) -> Color:
        return self.color
