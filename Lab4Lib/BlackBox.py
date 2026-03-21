import numpy as np
from .bezier import BezierCurve

class BlackBox:
    def __init__(self, builder: np.ndarray):
        x_line, y_line = builder
        self._curves = [BezierCurve(np.array([x_line[i:i + 4], y_line[i:i + 4]])) for i in (0, 3, 6)]

    def data_points(self, num, noise=0):
        x = np.random.rand(num) * 200 - 100
        return x, self.set_data_points(x, noise)


    def set_data_points(self, x, noise=0):
        norm_x = x.copy()
        while np.any((norm_x > 100) | (norm_x <= -100)):
            norm_x[norm_x > 100] -= 200
            norm_x[norm_x <= -100] += 200

        y = np.zeros_like(x)

        for c1 in self._curves:
            start_x = c1.control_points[0][0]
            end_x = c1.control_points[0][-1]
            x_mask = (norm_x >= start_x) & (norm_x <= end_x)
            if any(x_mask):
                y[x_mask] = c1.get_y_vector(norm_x[x_mask])

        return y + np.random.rand(x.size) * noise * 2 - noise

    @staticmethod
    def line_point(x, y, new_x):
        X = np.stack([x, np.ones(len(x))]).T
        kb = np.linalg.inv(X) @ y
        return np.array([new_x, 1]) @ kb


def word_bit(word):
    while len(word) < 10:
        word += word

    first_line = split_ordinal(word)
    second_line = split_ordinal(word[::-1])

    first_line = np.array(first_line).astype(np.float32)
    second_line = np.array(second_line).astype(np.float32)

    first_line = first_line + np.roll(second_line, -1) - np.roll(first_line, 1)
    second_line = second_line + np.roll(first_line, -1) - np.roll(second_line, 1)

    first_line = normalize(np.sort(first_line), -100, 100)
    second_line = normalize(second_line, -50, 50)

    return first_line, second_line


def split_ordinal(word):
    line = []
    for symbol in word[:10]:
        ordinal = ord(symbol) ** 2
        while not(0 <= ordinal < 500):
            if ordinal < 0:
                ordinal *= -1
            elif ordinal > 2000:
                ordinal *= (ordinal % 100 + 1) / 101
            elif ordinal > 500:
                ordinal = ordinal * 3.3 - 100

        line.append(int(ordinal))
    return line


def normalize(array, min_, max_):
    min_arr = np.min(array)
    max_arr = np.max(array)
    delta = max_arr - min_arr
    array = (array - min_arr) / delta
    return array * (max_ - min_) + min_

if __name__ == '__main__':
    b1 = BlackBox()

    print(BlackBox.line_point(np.array([-1, 1]), np.array([5, 6]), 2))