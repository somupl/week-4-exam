import unittest as unittest
import sqlite3 as sql
import numpy as np
import pandas as pd
from src import assessment as a


class TestAssessment1(unittest.TestCase):

    def test_01_count_characters(self):
        string = "abafdcggfaabe"
        answer = {"a": 4, "b": 2, "c": 1, "d": 1, "e": 1, "f": 2, "g": 2}
        result = a.count_characters(string)
        self.assertEqual(result, answer)

    def test_02_invert_dictionary(self):
        d = {"a": 4, "b": 2, "c": 1, "d": 1, "e": 1, "f": 2, "g": 2}
        result = {4: {'a'}, 2: {'b', 'f', 'g'}, 1: {'c', 'd', 'e'}}
        self.assertEqual(a.invert_dictionary(d), result)

    def test_04_matrix_multiplication(self):
        A = [[2, 3, 4], [6, 4, 2], [-1, 2, 0]]
        B = [[8, -3, 1], [-7, 3, 2], [0, 3, 3]]
        answer = [[-5, 15, 20], [20, 0, 20], [-22, 9, 3]]
        self.assertEqual(a.matrix_multiplication(A, B), answer)

    def test_05_cookie_jar(self):
        self.assertEqual(a.cookie_jar(0.2, 0.6), 0.25)

    def test_06_array_work(self):
        matrixA = np.array([[-4, -2],
                            [0, -3],
                            [-4, -1],
                            [-1, 1],
                            [-3, 0]])
        answer1 = np.array([[-24, -24, -24],
                            [-12, -12, -12],
                            [-20, -20, -20],
                            [0, 0, 0],
                            [-12, -12, -12]])
        result1 = a.array_work(2, 3, 4, matrixA)
        self.assertTrue(np.all(answer1 == result1))

        answer2 = np.array([[-36, -36],
                            [-18, -18],
                            [-30, -30],
                            [0, 0],
                            [-18, -18]])
        result2 = a.array_work(2, 2, 6, matrixA)
        self.assertTrue(np.all(answer2 == result2))

    def test_07_boolean_indexing(self):
        arr = np.array([[-4, -4, -3],
                        [-1, 16, -4],
                        [-3, 6, 4]])
        result1 = a.boolean_indexing(arr, 0)
        answer1 = np.array([16, 6, 4])
        self.assertTrue(np.all(result1 == answer1))
        result2 = a.boolean_indexing(arr, 10)
        answer2 = np.array([16])
        self.assertTrue(np.all(result2 == answer2))

    def test_08_make_series(self):
        result = a.make_series(7, 4, ['a', 'b', 'c', 'd'])
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(result['a'], 7)
        self.assertEqual(result['d'], 10)

        result = a.make_series(22, 5, ['a', 'b', 'c', 'd', 'hi'])
        self.assertEqual(result['a'], 22)
        self.assertEqual(result['d'], 25)
        self.assertEqual(result['hi'], 26)

    def test_09_data_frame_work(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        colA, colB, colC = ('a', 'b', 'c')
        a.data_frame_work(df, colA, colB, colC)
        self.assertTrue(colC in df.columns.tolist())
        self.assertEqual(df[colC].tolist(), [5, 7, 9])


if __name__ == '__main__':
    unittest.main()
