from collections import defaultdict
import numpy as np
import pandas as pd


def count_characters(string):
    '''
    INPUT: STRING
    OUTPUT: DICT (with counts of each character in input string)

    Return a dictionary which contains
    a count of the number of times each character appears in the string.
    Characters with a count of 0 should not be included in the
    output dictionary.
    '''
    return {letter: string.count(letter) for letter in set(string)}


def invert_dictionary(d):
    '''
    INPUT: DICT
    OUTPUT: DICT (of sets of input keys indexing the same input values
                  indexed by the input values)

    Given a dictionary d, return a new dictionary with d's values
    as keys and the value for a given key being
    the set of d's keys which shared the same value.
    e.g. {'a': 2, 'b': 4, 'c': 2} => {2: {'a', 'c'}, 4: {'b'}}
    '''
    my_dict = defaultdict(list)
    for x, y in d.items():
        my_dict[y].append(x)

    ans_dict = {}
    for x, y in dict(my_dict).items():
        ans_dict[x] = set(y)
    return ans_dict


def matrix_multiplication(A, B):
    '''
    INPUT: LIST (of length n) OF LIST (of length n) OF INTEGERS,
            LIST (of length n) OF LIST (of length n) OF INTEGERS
    OUTPUT: LIST OF LIST OF INTEGERS
            (storing the product of a matrix multiplication operation)

    Return the matrix which is the product of matrix A and matrix B
    where A and B will be (a) integer valued (b) square matrices
    (c) of size n-by-n (d) encoded as lists of lists.

    For example:
    A = [[2, 3, 4], [6, 4, 2], [-1, 2, 0]] corresponds to the matrix

        | 2  3  4 |
        | 6  4  2 |
        |-1  2  0 |

    Please do not use numpy. Write your solution in straight python.
    '''
    a_shape = (len(A), len(A[0]))
    b_shape = (len(B), len(B[0]))
    if(a_shape == b_shape[:: -1]):
        b_t = []
        for col in range(b_shape[1]):
            v = []
            for row in range(b_shape[0]):
                v.append(B[row][col])
            b_t.append(v)

        mat_mul = []
        for row in range(a_shape[0]):
            vec = []
            vec = [sum([c1 * c2 for c1, c2 in zip(A[row], b_t[x])]) for x in range(a_shape[1])]
            # vec.append(sum([c1 * c2 for c1, c2 in zip(A[row], b_t[0])]))
            # vec.append(sum([c1 * c2 for c1, c2 in zip(A[row], b_t[1])]))
            # vec.append(sum([c1 * c2 for c1, c2 in zip(A[row], b_t[2])]))
            mat_mul.append(vec)
        return(mat_mul)
    else:
        return


def cookie_jar(a, b):
    '''
    INPUT: FLOAT (probability of drawing a chocolate cooking from Jar A),
            FLOAT (probability of drawing a chocolate cooking from Jar B)
    OUTPUT: FLOAT (conditional probability that cookie was drawn from Jar A
                   given that a chocolate cookie was drawn)

    There are two jars of cookies.
    Each has chocolate and peanut butter cookies.
    INPUT 'a' is the fraction of cookies in Jar A which are chocolate
    INPUT 'b' is the fraction of cookies in Jar B which are chocolate
    A jar is chosen at random and a cookie is drawn.
    The cookie is chocolate.
    Return the probability that the cookie came from Jar A.
    '''
    return ((a * 0.5) / ((a * 0.5) + (b * 0.5)))


def array_work(rows, cols, scalar, matrixA):
    '''
    INPUT: INT, INT, INT, NUMPY ARRAY
    OUTPUT: NUMPY ARRAY
    (of matrix product of r-by-c matrix of "scalar"'s time matrixA)

    Create matrix of size (rows, cols) with elements initialized to the scalar
    value. Right multiply that matrix with the passed matrixA (i.e. AB, not
    BA).  Return the result of the multiplication.  You needn't check for
    matrix compatibililty, but you accomplish this in a single line.

    E.g., array_work(2, 3, 5, [[3, 4], [5, 6], [7, 8]])
           [[3, 4],      [[5, 5, 5],
            [5, 6],   *   [5, 5, 5]]
            [7, 8]]
    '''
    return matrixA @ np.array([scalar] * (rows * cols)).reshape(rows, cols)


def boolean_indexing(arr, minimum):
    '''
    INPUT: NUMPY ARRAY, INT
    OUTPUT: NUMPY ARRAY
    (of just elements in "arr" greater or equal to "minimum")

    Return an array of only the elements of "arr" that are greater than or
    equal to "minimum"

    Ex:
    In [1]: boolean_indexing([[3, 4, 5], [6, 7, 8]], 7)
    Out[1]: array([7, 8])
    '''
    return arr[arr >= minimum]


def make_series(start, length, index):
    '''
    INPUTS: INT, INT, LIST (of length "length")
    OUTPUT: PANDAS SERIES (of "length" sequential integers
             beginning with "start" and with index "index")

    Create a pandas Series of length "length" with index "index"
    and with elements that are sequential integers starting from "start".
    You may assume the length of index will be "length".

    E.g.,
    In [1]: make_series(5, 3, ['a', 'b', 'c'])
    Out[1]:
    a    5
    b    6
    c    7
    dtype: int64
    '''

    return pd.Series(np.arange(start, start + length), index=index)


def data_frame_work(df, colA, colB, colC):
    '''
    INPUT: DATAFRAME, STR, STR, STR
    OUTPUT: None

    Insert a column (colC) into the dataframe that is the sum of colA and colB.
    Assume that df contains columns colA and colB and that these are numeric.
    '''
    df[colC] = df[colA] + df[colB]
    return df
