OPENQASM 2.0;
include "qelib1.inc";

// Smallest hardware validation of the controlled periodic x -> x+1 mod 4
// primitive used by quantum_aero.quantum.controlled_periodic_streaming_circuit.
qreg q[3];
creg c[3];
h q[0];
x q[1];
ccx q[0],q[1],q[2];
cx q[0],q[1];
measure q -> c;
