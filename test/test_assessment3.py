from __future__ import division
import unittest as unittest
import numpy as np
import pandas as pd
import scipy.stats as st
import sqlite3 as sql
from src import assessment3 as a


class TestAssessment3(unittest.TestCase):

    def test_roll_the_dice(self):
        result = a.roll_the_dice(10000)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 2 / 3, places=1)

#     def test_pandas_query(self):
#         answer = [934.38, 1129.41, 2399.89, 3504.50, 6889.37, 11478.91]
#         result = a.pandas_query(pd.read_csv('data/universities.csv'))
#         if isinstance(result, pd.DataFrame):
#             self.assertIn('Size', result.columns)
#             series = result['Size']
#         else:
#             series = result
#         self.assertIsInstance(series, pd.Series)
#         lst = series.values.tolist()
#         self.assertEqual(len(lst), 6)
#         for x, y in zip(lst, answer):
#             self.assertIsInstance(x, float)
#             self.assertAlmostEqual(x, y, places=2)

#     def test_df_to_numpy(self):
#         df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8],
#                            'c': [9, 10, 11, 12], 'd': [13, 14, 15, 16]})
#         results = a.df_to_numpy(df, 'c')
#         self.assertIsNotNone(results)
#         self.assertEqual(len(results), 2)
#         X, y = results
#         self.assertIsInstance(X, np.ndarray)
#         self.assertIsInstance(y, np.ndarray)
#         self.assertEqual(X.tolist(), [[1, 5, 13], [2, 6, 14], [3, 7, 15], [4, 8, 16]])
#         self.assertEqual(y.tolist(), [9, 10, 11, 12])

#     def test_only_positive(self):
#         arr = np.array([[1, 2, 3], [4, -5, -6], [-7, 8, 9], [10, 11, 12]])
#         result = a.only_positive(arr)
#         self.assertIsInstance(result, np.ndarray)
#         self.assertEqual(result.tolist(), [[1, 2, 3], [10, 11, 12]])

#     def test_add_column(self):
#         arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#         col = np.array([10, 11, 12])
#         result = a.add_column(arr, col)
#         answer = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
#         self.assertIsInstance(result, np.ndarray)
#         self.assertEqual(result.tolist(), answer)

#     def test_size_of_multiply(self):
#         A = np.array([[1, 2]])
#         B = np.array([[3, 4]])
#         self.assertIsNone(a.size_of_multiply(A, B))
#         A = np.array([[1, 2], [3, 4], [5, 6]])
#         B = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
#         self.assertEqual(a.size_of_multiply(A, B), (3, 4))
#         self.assertIsNone(a.size_of_multiply(B, A))

#     def test_linear_regression(self):
#         df = pd.read_csv('data/lin_reg.csv')
#         X = df[['A', 'B']].values
#         y = df['C'].values
#         X_train = X[:25]
#         y_train = y[:25]
#         X_test = X[25:]
#         y_test = y[25:]
#         results = a.linear_regression(X_train, y_train, X_test, y_test)
#         self.assertIsNotNone(results)
#         self.assertEqual(len(results), 2)
#         coeffs, r2 = results
#         self.assertAlmostEqual(r2, 0.773895, 6)
#         self.assertAlmostEqual(coeffs[0], 13.815259, 6)
#         self.assertAlmostEqual(coeffs[1], 85.435835, 6)


if __name__ == '__main__':
    unittest.main()
