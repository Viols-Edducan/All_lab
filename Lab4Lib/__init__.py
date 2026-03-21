import numpy as np

from .bezier import BezierCurve
from .BlackBox import BlackBox, word_bit

class LidError(Exception):

    def __init__(self, value):
        super().__init__(value)


def get_variant(word):
    try:
        x_line, y_line = word_bit(word)
        y_line[4] = BlackBox.line_point(x_line[2:4], y_line[2:4], x_line[4])
        y_line[7] = BlackBox.line_point(x_line[5:7], y_line[5:7], x_line[7])
        y_line[8] = BlackBox.line_point(x_line[0:2], y_line[0:2], x_line[8] - (x_line[9] - x_line[0]))
        y_line[9] = y_line[0]
    except Exception as ex:
        raise LidError(f'Original error: {ex}\nJust use another word')
    return BlackBox(np.stack([x_line, y_line], axis=0))
