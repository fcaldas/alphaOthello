import enum
import numpy as np


class Tile(enum.IntEnum):
    EMPTY = 0
    BLACK = -1
    WHITE = 1


class Board:

    def __init__(self) -> None:
        self.board = np.zeros([8, 8])
        self.board[3, 3] = 1
        self.board[4, 4] = 1
        self.board[3, 4] = -1
        self.board[4, 3] = -1

    def get_board(self) -> np.array:
        return self.board

    def get_positions_with(self, tile: Tile | int) -> list[tuple[int, int]]:
        pos_x, pos_y = np.where(self.board == int(tile))
        return list(zip(pos_x, pos_y))

    def play(self, x: int, y: int, tile: Tile) -> tuple[bool, int]:
        """
        Plays a tile in position x, y returns a boolean indicating if this was a valid
        move and an integer containing the number of pieces turned.
        """

        def in_bounds(px: int, py: int):
            return px >= 0 and px < 8 and py >= 0 and py < 8

        if not in_bounds(x, y) or self.board[x, y] != int(Tile.EMPTY):
            return False, 0

        turns = 0
        iters = np.array(
            [[0, -1], [0, 1], [-1, 0], [1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]
        )

        initial_pos = np.array([x, y])
        for direction in iters:
            iter_pos = initial_pos + direction
            # Go to the last piece we can turn
            while in_bounds(iter_pos[0], iter_pos[1]) and self.board[
                iter_pos[0], iter_pos[1]
            ] == -1 * int(tile):
                iter_pos += direction

            # Check the piece after that is the same color the piece we are playing
            if in_bounds(iter_pos[0], iter_pos[1]) and self.board[
                iter_pos[0], iter_pos[1]
            ] == int(tile):
                # Go back one so we can start flipping
                iter_pos -= direction
                while iter_pos != initial_pos:
                    self.board[iter_pos[0], iter_pos[1]] *= -1
                    turns += 1
                    iter_pos -= direction

        # In Othello every move has to turn pieces from your oponent
        # otherwhise it is not a valid move
        if turns == 0:
            return False, 0

        self.board[x, y] = int(tile)
        return True, turns

    def print(self):
        print("    0 1 2 3 4 5 6 7")

        def to_token(tile: Tile) -> str:
            match tile:
                case Tile.EMPTY:
                    return " "
                case Tile.WHITE:
                    return "O"
                case Tile.BLACK:
                    return "X"

        for ix in range(0, 8):
            print(f"{ix}:  {' '.join([to_token(v) for v in self.board[ix, :]])}")


if __name__ == "__main__":
    b = Board()
    b.print()
