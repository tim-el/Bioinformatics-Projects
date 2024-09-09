import sys
import numpy

"""The following uses Python to challenge you to create an algorithm for finding
matches between a set of aligned strings. Minimal familiarity with Python is 
necessary, notably list and Numpy array slicing. 

Do read the accompanying Durbin et al. paper before attempting the challenge!
"""

"""Problem 1.

Let X be a list of M binary strings (over the alphabet { 0, 1 }) each of length 
N. 

For integer 0<=i<=N we define an ith prefix sort as a lexicographic sort 
(here 0 precedes 1) of the set of ith prefixes: { x[:i] | x in X }.
Similarly an ith reverse prefix sort is a lexicographic sort of the set of
ith prefixes after each prefix is reversed.

Let A be an Mx(N+1) matrix such that for all 0<=i<M, 0<=j<=N, A[i,j] is the 
index in X of the ith string ordered by jth reverse prefix. To break ties 
(equal prefixes) the ordering of the strings in X is used. 

Complete code for the following function that computes A for a given X.

Here X is a Python list of Python strings. 
To represent A we use a 2D Numpy integer array.

Example:

>>> X = getRandomX() #This is in the challenge1UnitTest.py file
>>> X
['110', '000', '001', '010', '100', '001', '100'] #Binary strings, M=7 and N=3
>>> A = constructReversePrefixSortMatrix(X)
>>> A
array([[0, 1, 1, 1],
       [1, 2, 2, 4],
       [2, 3, 5, 6],
       [3, 5, 4, 3],
       [4, 0, 6, 0],
       [5, 4, 3, 2],
       [6, 6, 0, 5]])
>>> 

Hint:
Column j (0 < j <= N) of the matrix can be constructed from column j-1 and the 
symbol in each sequence at index j-1.  

Question 1: In terms of M and N what is the asymptotic cost of your algorithm?
    
    Answer: Considering each sort operation takes O(M log M) time and there are N such operations, the asymptotic cost will be O(NM log M).

"""


def constructReversePrefixSortMatrix(X):
    """
    Constructs the reverse prefix sort matrix for a given list of binary strings.

    Args:
        X (list): A list of M binary strings, each of length N.

    Returns:
        numpy.ndarray: An M*(N+1) matrix A, where A[i,j] is the index in X of the ith string
        ordered by the jth reverse prefix.
    """
    # Creates the Mx(N+1) matrix
    A = numpy.empty(shape=[len(X), 1 if len(X) == 0 else len(X[0]) + 1], dtype=int)

    # Code to write:

    # Initialize the first column of A
    A[:, 0] = numpy.arange(len(X))

    # Fill in each column of A based on the sorted order of reverse prefixes.
    for j in range(1, len(X[0]) + 1):
        # Sort the indices of X based on the jth reverse prefix of each string.
        A[:, j] = sorted(range(len(X)), key=lambda i: X[i][:j][::-1])

    return A


"""Problem 2: 

Following on from the previous problem, let Y be the MxN matrix such that for 
all 0 <= i < M, 0 <= j < N, Y[i,j] = X[A[i,j]][j].

Complete the following to construct Y for X. 

Hint: You can either use your solution to constructReversePrefixSortMatrix() 
or adapt the code from that algorithm to create Y without using 
constructReversePrefixSortMatrix().

Question 2: In terms of M and N what is the asymptotic cost of your algorithm?
    Answer: The asymptotic cost is O(MN), as it involves iterating over each element in an MxN matrix.
"""
# Function to construct Y from X
def constructYFromX(X):
    """
    Constructs the Y matrix from X using the reverse prefix sort matrix.

    Args:
        X (list): A list of M binary strings, each of length N.

    Returns:
        numpy.ndarray: An MxN matrix Y, where Y[i,j] = X[A[i,j]][j].
    """

    # Creates the MxN matrix
    Y = numpy.empty(shape=[len(X), 0 if len(X) == 0 else len(X[0])], dtype=int)

    # Utilize the reverse prefix sort matrix A to construct Y.
    A = constructReversePrefixSortMatrix(X)  
    for i in range(len(X)):
        for j in range(len(X[0])):
            Y[i, j] = int(X[A[i, j]][j])  # Assign the corresponding value from X to Y

    return Y


"""Problem 3.

Y is a transformation of X. Complete the following to construct X from Y, 
returning X as a list of strings as defined in problem 1.
Hint: This is the inverse of X to Y, but the code may look very similar.

Question 3a: In terms of M and N what is the asymptotic cost of your algorithm?
    Answer: The asymptotic cost is O(MN), as it involves concatenating characters for each of the MN positions.

Question 3b: What could you use Y for? 
Hint: consider the BWT.
    Answer: Y can be used for efficient haplotype matching and storage, as it transforms the data into a form 
    that facilitates compression and pattern matching, similar to BWT in text compression and search.

Question 3c: Can you come up with a more efficient data structure for storing Y?
    Answer: A more efficient data structure for storing Y could be a run-length encoded structure, 
    especially if Y contains many repeated elements. 
    This can significantly reduce space if Y has redundancy, which is common in genetic data.
"""

import numpy as np

def constructXFromY(Y):
    """
    Constructs X from Y.

    Args:
        Y (numpy.ndarray): Input array of shape (M, N), where M is the number of sequences and N is the number of positions.

    Returns:
        list: List of strings representing the constructed X.

    """

    M, N = Y.shape  # M sequences, N positions

    # Initialize array A for tracking indices during reconstruction.
    A = np.arange(M)
    # Initialize the array X for constructing the binary strings.
    X = numpy.empty(shape=[len(Y), 0 if len(Y) == 0 else len(Y[0]) ], dtype=int)
    
    # Reconstruct each position of X from Y using the sorted indices.
    for j in range(N):
        if j > 0:
            # Update A via sorting by the previous column of Y.
            A = A[np.argsort(Y[:, j-1])]
        
        for i in range(M):
            X[A[i], j] = Y[i, j]    
    # Convert the integer array X back into a list of binary strings.
    return ["".join([str(char) for char in seq]) for seq in X]




"""Problem 4.

Define the common suffix of two strings to be the maximum length suffix shared 
by both strings, e.g. for "10110" and "10010" the common suffix is "10" because 
both end with "10" but not both "110" or both "010". 

Let D be a Mx(N+1) Numpy integer array such that for all 1<=i<M, 1<=j<=N, 
D[i,j] is the length of the common suffix between the substrings X[A[i,j]][:j] 
and X[A[i-1,j]][:j].  

Complete code for the following function that computes D for a given A.

Example:

>>> X = getRandomX()
>>> X
['110', '000', '001', '010', '100', '001', '100']
>>> A = constructReversePrefixSortMatrix(X)
>>> A
array([[0, 1, 1, 1],
       [1, 2, 2, 4],
       [2, 3, 5, 6],
       [3, 5, 4, 3],
       [4, 0, 6, 0],
       [5, 4, 3, 2],
       [6, 6, 0, 5]])
>>> D = constructCommonSuffixMatrix(A, X)
>>> D
array([[0, 0, 0, 0],
       [0, 1, 2, 2],
       [0, 1, 2, 3],
       [0, 1, 1, 1],
       [0, 0, 2, 2],
       [0, 1, 0, 0],
       [0, 1, 1, 3]])

Hints: 

As before, column j (0 < j <= N) of the matrix can be constructed from column j-1 
and thesymbol in each sequence at index j-1.

For an efficient algorithm consider that the length of the common suffix 
between X[A[i,j]][:j] and X[A[i-k,j]][:j], for all 0<k<=i is 
min(D[i-k+1,j], D[i-k+2,j], ..., D[i,j]).

Question 4: In terms of M and N what is the asymptotic cost of your algorithm?
    Answer: The asymptotic cost is O(MN), as it involves iterating over each element in an MxN matrix.
            The worst-case asymptotic cost is O(MN^2), as it involves iterating over each element in an MxN matrix and performing a comparison for each element in the matrix.
"""

def constructCommonSuffixMatrix(A, X):
    """
    Computes the common suffix matrix D for given A and X.

    Args:
        A (numpy.ndarray): The reverse prefix sort matrix.
        X (list): The original list of binary strings.

    Returns:
        numpy.ndarray: The common suffix matrix D.
    """

    D = numpy.zeros(shape=A.shape, dtype=int)  # Keeps the Mx(N+1) D matrix 

    # Define a helper function to compute the length of the common suffix.
    def common_suffix_len(str1, str2):
        i = 0 # Initialize suffix length counter.
        while i < len(str1) and i < len(str2) and str1[-(i + 1)] == str2[-(i + 1)]:
            i += 1
        return i

    # Compute the common suffix lengths for each position in D.
    for j in range(1, A.shape[1]):
        for i in range(1, A.shape[0]):
            suffix_len = common_suffix_len(X[A[i, j]][:j], X[A[i - 1, j]][:j])
            D[i, j] = suffix_len

    return D



"""Problem 5.
    
For a pair of strings X[x], X[y], a long match ending at j is a common substring
of X[x] and X[y] that ends at j (so that X[x][j] != X[y][j] or j == N) that is longer
than a threshold 'minLength'. E.g. for strings "0010100" and "1110111" and length
threshold 2 (or 3) there is a long match "101" ending at 5.
    
The following algorithm enumerates for all long matches between all substrings of
X, except for simplicity those long matches that are not terminated at
the end of the strings.
    

Question 5a: What is the asymptotic cost of the algorithm in terms of M, N and the
number of long matches?

# The asymptotic cost of the algorithm in terms of M, N, and the number of long matches
 is primarily O(MN + K), where O(MN) accounts for the iteration over the sequences and their positions, 
 and O(K) represents the cost of processing and yielding the long matches. 


Question 5b: Can you see any major time efficiencies that could be gained by
refactoring?

# Refactoring the algorithm to incorporate the PBWT approach directly for identifying 
 long matches could yield major time efficiencies. By organizing data in a manner that 
 allows for rapid comparison of haplotype sequences based on their prefixes, the algorithm
 can avoid redundant comparisons and quickly identify potential matches.



Question 5c: Can you see any major space efficiencies that could be gained by
refactoring?

# Space efficiencies can be gained by refactoring the algorithm to utilize the 
 compact representation capabilities of PBWT. The PBWT compresses haplotype data
 by exploiting the redundancy and similarity within haplotype sequences, a strategy
 which could be applied to the storage of X, A, and D matrices, significantly reducing memory footprint.



Question 5d: Can you imagine alternative algorithms to compute such matches?,
if so, what would be the asymptotic cost and space usage?

# An alternative approach to computing long matches could involve directly applying the PBWT's more
 efficient preprocessing and organization methods. The PBWT offer a structured way to preprocess
 and organize data that can drastically speed up the search for matching sequences, compressing data 
 and finding maximal haplotype matches within a set in linear time, O(NM). The space usage would also 
 benefit from the PBWT's compression capabilities, potentially reducing the storage requirements for large genomic datasets. 
    
"""
def getLongMatches(X, minLength):
    assert minLength > 0
    
    A = constructReversePrefixSortMatrix(X)
    D = constructCommonSuffixMatrix(A, X)
    
    #For each column, in ascending order of column index
    for j in range(1, 0 if len(X) == 0 else len(X[0])):
        #Working arrays used to store indices of strings containing long matches
        #b is an array of strings that have a '0' at position j
        #c is an array of strings that have a '1' at position j
        #When reporting long matches we'll report all pairs of indices in b X c,
        #as these are the long matches that end at j.
        b, c = [], []
        
        #Iterate over the aligned symbols in column j in reverse prefix order
        for i in range(len(X)):
            #For each string in the order check if there is a long match between
            #it and the previous string.
            #If there isn't a long match then this implies that there can
            #be no long matches ending at j between sequences indices in A[:i,j]
            #and sequence indices in A[i:,j], thus we report all long matches
            #found so far and empty the arrays storing long matches.
            if D[i,j] < minLength:
                for x in b:
                    for y in c:
                        #The yield keyword converts the function into a
                        #generator - alternatively we could just to append to
                        #a list and return the list
                        
                        #We return the match as tuple of two sequence
                        #indices (ordered by order in X) and coordinate at which
                        #the match ends
                        yield (x, y, j) if x < y else (y, x, j)
                b, c = [], []
            
            #Partition the sequences by if they have '0' or '1' at position j.
            if X[A[i,j]][j] == '0':
                b.append(A[i,j])
            else:
                c.append(A[i,j])
        
        #Report any leftover long matches for the column
        for x in b:
            for y in c:
                yield (x, y, j) if x < y else (y, x, j)


