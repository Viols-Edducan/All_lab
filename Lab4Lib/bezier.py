import numpy as np
from .minilib import c_, bezier_rang_matrix

class BezierCurve(object):
    _p_vector: np.ndarray
    _b_vector: np.ndarray
    _a_vector: np.ndarray

    def __init__(self, vectors: np.ndarray):
        if vectors.shape[0] != 2:
            if vectors.shape[1] != 2:
                raise ValueError('broken vector')
            self._p_vector = vectors.T
        else:
            self._p_vector = vectors.copy()
        self._p_vector = self._p_vector.astype(np.float32)
        if self.power < 1:
            raise ValueError('curve is compressed to point')
        self._b_vector = np.array([c_(self.power, i) for i in range(self.power + 1)]).reshape(1, -1)
        self._a_vector = (self._p_vector @ bezier_rang_matrix(self.power + 1))

    def for_t_vector(self, t_vector: np.ndarray):
        coef_x = self._b_vector * self._p_vector[0]
        coef_y = self._b_vector * self._p_vector[1]
        big_t_vector = np.array([np.power(t_vector, i) * np.power(1 - t_vector, self.power - i)
                                 for i in range(self.power + 1)])
        return np.concatenate((coef_x @ big_t_vector, coef_y @ big_t_vector))

    def get_y_vector(self, x: np.ndarray):
        t = self._transform(self._a_vector[0], x)
        return self.for_t_vector(t)[1]

    def get_x_vector(self, y: np.ndarray):
        t = self._transform(self._a_vector[1], y)
        return self.for_t_vector(t)[0]

    def _transform(self, coeff_vector: np.ndarray, arr: np.ndarray):
        result = np.tile(coeff_vector[::-1], (len(arr), 1))
        result[:, -1] -= arr
        roots_array = np.apply_along_axis(np.roots, axis=1, arr=result)
        ans = self.cut_for(roots_array)
        return ans

    @staticmethod
    def cut_for(elements):
        def pick_root(roots_row, tol=1e-8):
            # Вибираємо дійсні корені (з малою уявною частиною)
            real_roots = [r.real for r in roots_row if abs(r.imag) < tol]
            if not real_roots:
                return np.nan  # якщо нема дійсних

            # Відстань до [0,1]
            def dist_to_unit_interval(x):
                if x < 0:
                    return -x
                elif x > 1:
                    return x - 1
                else:
                    return 0

            # Обираємо корінь з мінімальною відстанню
            return min(real_roots, key=dist_to_unit_interval)

        return np.array([pick_root(row) for row in elements])

    def for_point_amount(self, amount):
        return self.for_t_vector(np.linspace(0, 1, amount))

    @property
    def power(self):
        return self._p_vector.shape[1] - 1

    @property
    def control_points(self):
        return self._p_vector.copy()

    @property
    def polynomial(self):
        return self._a_vector.copy()


