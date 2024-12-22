import enum
import numpy as np


class Color(enum.IntEnum):
    BLACK = -1
    WHITE = 1


class Tile(enum.IntEnum):
    EMPTY = 0
    BLACK = Color.BLACK
    WHITE = Color.WHITE


class Board:

    def __init__(self) -> None:
        self.board = np.zeros([8, 8])
        self.board[3, 3] = 1
        self.board[4, 4] = 1
        self.board[3, 4] = -1
        self.board[4, 3] = -1

    def get_board(self) -> np.ndarray:
        return self.board

    def get_positions_with(self, tile: Tile | int) -> list[tuple[int, int]]:
        pos_x, pos_y = np.where(self.board == int(tile))
        return list(zip(pos_x, pos_y))

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < 8 and 0 <= y < 8

    def is_valid_move(self, x: int, y: int, color: Color) -> bool:
        if not self.in_bounds(x, y):
            return False
        if self.board[x][y] != 0:
            return False
        initial_pos = np.array([x, y])
        iters = np.array(
            [[0, -1], [0, 1], [-1, 0], [1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]
        )

        other_color = int(color) * -1
        is_valid = False
        for direction in iters:
            turned = 0
            iter_pos = initial_pos + direction
            # Go to the last piece we can turn
            while (
                self.in_bounds(iter_pos[0], iter_pos[1])
                and self.board[iter_pos[0], iter_pos[1]] == other_color
            ):
                turned += 1
                iter_pos += direction

            if all(iter_pos == initial_pos) or turned == 0:
                continue
            # Check the piece after that is the same color the piece we are playing
            if self.in_bounds(iter_pos[0], iter_pos[1]) and self.board[
                iter_pos[0], iter_pos[1]
            ] == int(color):
                return True
        return False

    def play(self, x: int, y: int, tile: Color) -> bool:
        """
        Plays a tile in position x, y returns a boolean indicating if this was a valid
        move
        """

        if not self.is_valid_move(x, y, tile):
            return False
        turns = 0
        iters = np.array(
            [[0, -1], [0, 1], [-1, 0], [1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]
        )

        initial_pos = np.array([x, y])
        for direction in iters:
            iter_pos = initial_pos + direction
            # Go to the last piece we can turn
            while self.in_bounds(iter_pos[0], iter_pos[1]) and self.board[
                iter_pos[0], iter_pos[1]
            ] == -1 * int(tile):
                iter_pos += direction

            # Check the piece after that is the same color the piece we are playing
            if self.in_bounds(iter_pos[0], iter_pos[1]) and self.board[
                iter_pos[0], iter_pos[1]
            ] == int(tile):
                # Go back one so we can start flipping
                iter_pos -= direction
                while not all(iter_pos == initial_pos):
                    self.board[iter_pos[0], iter_pos[1]] *= -1
                    turns += 1
                    iter_pos -= direction

        # In Othello every move has to turn pieces from your oponent
        # otherwhise it is not a valid move
        if turns == 0:
            return False

        self.board[x, y] = int(tile)
        return True

    def is_game_over(self):
        """
        Check if the game is over and returns true if that is the case.
        """
        return len(np.where(self.board == int(Tile.EMPTY))[0]) == 0

    def print(self):
        """
        Print the game board
        """
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
