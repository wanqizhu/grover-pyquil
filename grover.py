import numpy as np
import math
from collections import Counter
import itertools
import random

from pyquil import quil
from pyquil.quil import Program
from pyquil.api import QVMConnection
from pyquil.gates import *
import matplotlib.pyplot as plt

qvm = QVMConnection(random_seed=137)
qvm_queue = QVMConnection(random_seed=137, use_queue=True)

T_inv_mat = np.array([[1,0],[0,(1-1j)/math.sqrt(2)]])


"""
Singular qubits are lowercase and often surronded in |a> in comments
Multi-qubit states are represented by uppercase letter arrays A[m]
"""


TMP_MAX = 63
tmp_in_use = {63: False}

def get_tmp_qubit():
    """ Helper function for allocating and deallocating temporary qubits
    It is assumed that each qubit starts off and ends in the |0> ground state

    We use our own alloc functionality because
    1) latest version of Pyquil disallows mixing p.alloc() with
        indexed qubits like passing in [0, 1] explicitly;
    2) We want to be space efficient, and so far pyquil doesn't have a way
        to dealloc qubits and allow re-using temporary indices
    3) Track globally which qubits are currently in use so that
        multiple sub-parts of a program don't use the same temporary index

    For simplicity, we use high-indexed qubits (e.g. 63, 62, ...) as temp qubits

    We call this function to get a temporary qubit index, and
    when we're done, we mark it free using 'tmp_in_use[t]=True'
    """
    i = TMP_MAX
    while i in tmp_in_use and tmp_in_use[i]:  # get next free qubit
        i-=1

    tmp_in_use[i] = True
    return i


def Toffoli(q0, q1, q2, p=None):
    """ Toffoli gate: a, b, c --> a, b, (a AND b) XOR c

    An implementating using basic gates available on Pyquil

    If c starts out in the |0> state, e.g. with a temporary qubit,
    then this computes a AND b

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

    We negate b_i exactly when the AND conjuction of a_0 ^ a_1 ^ ... 
        ^ a_(i+1)is true

    This is used as a helper function for implementing AndTemp
    """

    m = len(A)
    assert len(B) == m-1, "Must pass in m qubit state 'A' followed by m-1 qubit state 'B'"
    assert m > 1

    if m == 2:  # base case
        return Toffoli(A[0], A[1], B[0])

    # recursive case: 
    state = Toffoli(A[0], B[1], B[0])  # uncomputes step 3
    state += Flip(A[1:], B[1:])  # computes b_(m-3) = a_(m-2) AND ... AND a_0
    state += Toffoli(A[0], B[1], B[0])  # computes b_(m-2) = a_(m-1) AND b_(m-3)

    return state


# testing flip
# p += Program(X(4), X(1), X(2), X(3)) + Flip([0, 1, 2, 3, 4])



def AndTemp(A, b, C):
    """

    A[m] |b> C[m-2]  --> applies AND across 'A' onto the single qubit 'B',
    using C temporarily
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
    p += AndTemp(A[:k], t, A[m-(k-2):])  # undos first step
    tmp_in_use[t] = False
    
    return p

# print((lambda p: (str(qvm.wavefunction(p)), p.get_qubits()))(And([0, 1, 2, 3], 4, Program(H(0), H(1), H(2), H(3)).defgate("T_inv", T_inv_mat))))


def Conditional(bitstring, gate, Qubits, target_bit, p=None):
    """ Conditionally apply 'gate' to |target_bit>
    iff bitstring is equal to the state 'Qubits'
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
    
    for cnt, bit in enumerate(bitstring):
        if bit == '0':
            p.inst(X(Qubits[cnt]))

    tmp_in_use[t] = False

    return p


# print((lambda p: (str(qvm.wavefunction(p)), p.get_qubits()))
#       (Conditional('1001', 'Z', [0, 1, 2, 3], 4, Program(H(0), H(1), H(2), H(3), X(4)).defgate("T_inv", T_inv_mat))))



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


# print((lambda p: (str(qvm.wavefunction(p)), p.get_qubits()))
#       (Oracle([0, 1, 2, 3], 4, '1011', Program(H(0), H(1), H(2), H(3), X(4)).defgate("T_inv", T_inv_mat))))



def Grover(Qubits, bitstring, p=None, num_iters=1):
    assert len(Qubits) == len(bitstring)

    if p is None: p = Program().defgate("T_inv", T_inv_mat)

    t = get_tmp_qubit()
    n = len(Qubits)

    # initialize to a superposition state, with auxiliary qubit in state |->
    p.inst(X(t))
    for bit in Qubits: p.inst(H(bit))
    p.inst(H(t))


    for i in range(num_iters):
        """ repeat oracle + amplification """
        p = Oracle(Qubits, t, bitstring, p)  # negates the state encoded in 'bitstring'

        for bit in Qubits: p.inst(H(bit))
        for bit in Qubits: p.inst(X(bit))

        # apply control-Z across qubits
        p = Conditional('1'*(n-1), 'Z', Qubits[:-1], Qubits[-1], p)

        for bit in Qubits: p.inst(X(bit))
        for bit in Qubits: p.inst(H(bit))

    # resetting the auxiliary qubit to state |0>
    p.inst(H(t))
    p.inst(X(t))
    tmp_in_use[t] = False

    return p



def testGrover(n, bitstring, num_iters=-1, measure=True, printOut=True):
    if num_iters == -1: num_iters = round(math.pi / 4 * math.sqrt(2**n))

    start_qubits = list(range(n))
    if printOut: print("Searching for {} with {} iterations...\n".format(bitstring, num_iters))

    p = Program().defgate("T_inv", T_inv_mat)
    p = Grover(start_qubits, bitstring, p, num_iters)
    
    # print(len(p))

    if measure:
        #sorted(list(p.get_qubits()))
        if n < 8:
            results = qvm.run_and_measure(p, start_qubits, trials=num_trials)
        else:
            results = qvm_queue.run_and_measure(p, start_qubits, trials=num_trials)
        # p.measure_all(*zip(start_qubits, start_qubits))
        # results = qvm.run(p, start_qubits, trials=num_trials)

        if printOut:
            # get a distribution of results
            print("Result: ", end="")
            print(Counter([tuple(a) for a in results]))
            print()

        return Counter([tuple(a) for a in results])

    return p

##########################################################
""" Testing AND circuit

We should see: a) auxiliary qubit always in state |0>
               b) all states in equal probability
               c) only when all inputs are 1 does the target qubit flip
"""
# p.inst(H(0), X(1), H(2), X(3))
# p = And([0, 1, 2, 3], 4, p=p)



n = 3 
num_trials = 1000




##########################################################
"""Graphing the number of gates as a function of n"""
# num_gates = []
# max_n = 8  # graph is made with 15, but that runs for a few minutes

# for n in range(2, max_n+1):
#     num_gates.append(len(testGrover(n, '1'*n, measure=False, printOut=False)))

# print(num_gates)
# c = num_gates[-1] / (max_n*math.sqrt(2**max_n))  # constant in big(O)
# plt.plot(list(range(2, max_n+1)), [c*n*math.sqrt(2**n) for n in range(2, max_n+1)], label='O(n * sqrt(N)), c={}'.format(c))
# plt.plot(list(range(2, max_n+1)), num_gates, label='number of gates')
# plt.title("Number of gates needed")
# plt.legend()
# plt.show()


##########################################################
""" Graphing probability as a function of number of iterations of oracle + amplification"""
def graph_probability_by_iterations(bitstring, max_iter, num_trials=1000, fig_name='plot.png'):
    n = len(bitstring)
    prob = []

    for num_iters in range(max_iter):
        p = testGrover(n, bitstring, num_iters, measure=False, printOut=False)
        results = qvm.run_and_measure(p, list(range(n)), trials=num_trials)
        prob.append(results.count(list(map(int, bitstring)))/num_trials)

    plt.plot(list(range(max_iter)), prob)
    plt.title("Grover's search for {}".format(bitstring))
    plt.xlabel("number of iterations of oracle + amplification")
    plt.ylabel("probability of measuring the desired state")
    plt.savefig(fig_name)
    # plt.show()
    

# graph_probability_by_iterations('101100', 15)



##########################################################
""" Graph outcome probabilities of a single run """

# res = testGrover(4, '1010', measure=True, printOut=False)
# plt.figure(figsize=(20, 10))
# plt.scatter([''.join(map(str, i)) for i in res], [res[i]/num_trials for i in res])
# plt.title('Grover Algorithm Outcomes')
# plt.savefig('Grover_Algorithm_Outcomes_searching_{}.png'.format('1010'))
# # plt.show()
# print(res)





##########################################################
""" Various basic testing suites for Grover """
# testGrover(3, '010', num_iters=2)


# print(qvm.wavefunction(testGrover(3, '010', measure=False)))


# for bitstring in ["".join(seq) for seq in itertools.product("01", repeat=n)]:
#     testGrover(n, bitstring)


p = testGrover(3, '010')#, measure=False)
# print(p.out())

# testGrover(8, '11001101', 1)  # seeing the difference in # of iterations
# testGrover(8, '11001101')

# testGrover(11, '11001101000', 1)
# testGrover(11, '11001101000')








# measures all qubits
# this was resolved here: https://github.com/rigetticomputing/pyquil/issues/223
# so latest version we can simply call p.measure_all()
#*zip(p.get_qubits(), p.get_qubits()))


