# Grover's Algorithm

An implementation in Pyquil, using only basic gates T, T_inverse, H, X, CNOT, and CZ.

Various test cases and graphings are commented out at the bottom of the [code](grover.py). More robust test suite and user interface coming!

`Grover(Qubits, bitstring)` gives you a program which you can run and measure using `qvm.run_and_measure()` or `qvm.wavefunction()`. `testGrover` is a helper function for running tests and summarizing results.


```python
>>> testGrover(3, '010')


"""Searching for 010 with 2 iterations...

Result: Counter({(0, 1, 0): 947, (0, 0, 0): 13, (1, 1, 1): 9, (0, 1, 1): 9, (0, 0, 1): 6, (1, 1, 0): 6, (1, 0, 1): 5, (1, 0, 0): 5})"""
```


## Resources

[Pyquil documentation](http://docs.rigetti.com/en/latest/index.html)

[Constructing Toffoli Gate](https://quantumexperience.ng.bluemix.net/proxy/tutorial/full-user-guide/004-Quantum_Algorithms/061-Basic_Circuit_Identities_and_Larger_Circuits.html)

[Grover Circuit Outline](https://www.nature.com/articles/s41467-017-01904-7)

[Building multi-qubit gates from simple gates](https://www.amazon.com/Quantum-Computing-Introduction-Engineering-Computation/dp/0262526670) Rieffel and Polak, Quantum Computing: A Gentle Introduction, Ch 6.4

[Proving Correctness of Amplification Circuit](https://www.cs.cmu.edu/~odonnell/quantum15/lecture04.pdf)


---

[Presentation of the circuit building blocks](https://docs.google.com/presentation/d/1F_vSeDQtoX7h4MHu0wlFe094_97QFWFNxOvz74-Wlx4/edit?usp=sharing)

Thanks to Monica Schleier-Smith and Jordan Cotler!