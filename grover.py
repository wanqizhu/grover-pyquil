import numpy as np
import math
from collections import Counter
import itertools
import random

from pyquil import quil
from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import *

qvm = QVMConnection()

T_inv_mat = np.array([[1,0],[0,(1-1j)/math.sqrt(2)]])


"""
Singular qubits are lowercase and often surronded in |a> in comments
Multi-qubit states are represented by uppercase letter arrays A[m]
"""


TMP_MAX = 63
tmp_in_use = {63: False}

def get_tmp_qubit():
    i = TMP_MAX
    while i in tmp_in_use and tmp_in_use[i]:
        i-=1

    tmp_in_use[i] = True
    return i


def Toffoli(q0, q1, q2, p=None):
    """ Toffoli gate: a, b, c --> a, b, (a AND b) XOR c

    https://quantumexperience.ng.bluemix.net/proxy/tutorial/full-user-guide/004-Quantum_Algorithms/061-Basic_Circuit_Identities_and_Larger_Circuits.html

    """

    if p is None: p = Program()

    p += Program(H(q2), CNOT(q1, q2), ("T_inv", q2), 
           CNOT(q0, q2), T(q2), CNOT(q1, q2),
           ("T_inv", q2), CNOT(q0, q2), T(q1),
           T(q2), H(q2), CNOT(q0, q1), T(q0),
           ("T_inv", q1), CNOT(q0, q1))

    return p


def Flip(A, B):
    """ Flip acts on an m-qubit register A and an m-1 qubit register B
    |a_(m-1), a_(m-2), ..., a_0> |b_(m-2), b_(m-3), ..., b_0>

    We negate b_i exactly when the AND conjuction of a_0 ^ a_1 ^ ... ^ a_(i+1) is true
    """

    m = len(A)
    assert len(B) == m-1, "Must pass in m qubit state 'A' followed by m-1 qubit state 'B'"
    assert m > 1

    if m == 2:
        return Toffoli(A[0], A[1], B[0])

    state = Toffoli(A[0], B[1], B[0])

    state += Flip(A[1:], B[1:])

    state += Toffoli(A[0], B[1], B[0])

    return state


# testing flip
# p += Program(X(4), X(1), X(2), X(3)) + Flip([0, 1, 2, 3, 4])



def AndTemp(A, b, C):
    """

    A[m] |b> C[m-2]  --> applies AND across 'A' onto the single qubit 'B', using C temporarily
    """

    m = len(A)
    assert len(C) == m-2, "Must pass in m qubit state 'A' followed by 1 qubit state |b> followed by m-2 state 'C'"
    assert m > 1

    if m == 2:
        return Toffoli(A[0], A[1], b)

    return Flip(A, [b]+C) + Flip(A[1:], C)


def And(A, b, p=None):
    """ A[m] |b> |t> --> |b> is applied C-AND over all qubits in A,
    using temporary qubit |t> """
    if p is None: p = Program()
    t = get_tmp_qubit()


    m = len(A)
    if m == 1:
        return p.inst(CNOT(A[0], b))

    if m == 2:
        return Toffoli(A[0], A[1], b, p=p)

    k = (m+1)//2

    p += AndTemp(A[:k], t, A[m-(k-2):])  # AND first k qubits into |t>
    p += AndTemp([t] + A[k:], b, A[:m-k-1])  # AND t and all qubits after k into |b>
    p += AndTemp(A[:k], t, A[m-(k-2):])
    tmp_in_use[t] = False
    
    return p


def Conditional(bitstring, gate, Qubits, target_bit, p=None):
    """ Conditionally apply 'gate' to |target_bit>
    iff bitstring is equal to the state passed in Qubits


    """
    assert len(Qubits) == len(bitstring)

    if p is None: p = Program()
    t = get_tmp_qubit()

    for cnt, bit in enumerate(bitstring):
        if bit == '0':
            p.inst(X(Qubits[cnt]))
    
    p = And(Qubits, t, p=p)

    if gate == 'Z':
        p.inst(CZ(t, target_bit))
    elif gate == 'NOT':
        p.inst(CNOT(t, target_bit))
    # TODO: other gates

    p = And(Qubits, t, p=p)
    tmp_in_use[t] = False
    
    for cnt, bit in enumerate(bitstring):
        if bit == '0':
            p.inst(X(Qubits[cnt]))

    return p



def Oracle(Qubits, aux_bit, bitstring, p=None):
    """ constuct an oracle which flips aux_bit if the input qubits
    correspond with the bitstring

    bitstring: a string of 0 and 1's
    """
    if p is None: p=Program()

    assert len(Qubits) == len(bitstring)

    # controlled-NOT on every bit
    p = Conditional(bitstring, 'NOT', Qubits, aux_bit, p=p)

    return p


def Grover(Qubits, bitstring, p=None, num_iters=1):
    assert len(Qubits) == len(bitstring)

    if p is None: p = Program()

    t = get_tmp_qubit()
    n = len(Qubits)

    p.inst(X(t))
    for bit in Qubits: p.inst(H(bit))
    p.inst(H(t))

    for i in range(num_iters):
        """ repeat oracle + amplification """
        
        p = Oracle(Qubits, t, bitstring, p)

        p.inst(H(t))
        for bit in Qubits: p.inst(H(bit))

        for bit in Qubits: p.inst(X(bit))

        # apply control-Z across qubits
        p = Conditional('1'*(n-1), 'Z', Qubits[:-1], Qubits[-1], p)

        for bit in Qubits: p.inst(X(bit))
        for bit in Qubits: p.inst(H(bit))

    p.inst(X(t))
    tmp_in_use[t] = False

    return p



def testGrover(n, bitstring, num_iters=-1):
    if num_iters == -1: num_iters = round(math.pi / 4 * math.sqrt(n))

    start_qubits = list(range(n))
    print("Searching for {} with {} iterations...\n".format(bitstring, num_iters))

    p = Program().defgate("T_inv", T_inv_mat)
    p = Grover(start_qubits, bitstring, p, num_iters)
    
    # print(len(p))

    #sorted(list(p.get_qubits()))
    results = qvm.run_and_measure(p, start_qubits, trials=num_trials)
    # p.measure_all(*zip(start_qubits, start_qubits))
    # results = qvm.run(p, start_qubits, trials=num_trials)

    # get a distribution of results
    print("Result: ")
    print(Counter([tuple(a) for a in results]))
    print()
    return p

# p.inst(H(0), X(1), H(2), X(3))
# p = And([0, 1, 2, 3], 4, p=p)


n = 3
num_trials = 1000


# for bitstring in ["".join(seq) for seq in itertools.product("01", repeat=n)]:
#     testGrover(n, bitstring)

# testGrover(4, '0101')


# for n in range(2, 20):
#     print(len(Grover(list(range(n)), '1'*n)))

testGrover(6, '110011', 1)
testGrover(6, '110011')

testGrover(8, '11001101', 1)
testGrover(8, '11001101')

testGrover(11, '11001101000', 1)
testGrover(11, '11001101000')



# for n in range(4, 6):
#     p = testGrover(n, random.choice(["".join(seq) for seq in itertools.product("01", repeat=n)]))
#     print(len(p))

# measures all qubits
# this was resolved here: https://github.com/rigetticomputing/pyquil/issues/223
# so latest version we can simply call p.measure_all()
#*zip(p.get_qubits(), p.get_qubits()))

# print(p.out())


