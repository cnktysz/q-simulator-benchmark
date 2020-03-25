import sys, os, time, datetime, csv
import pennylane as qml 
from pennylane import numpy as np
import pennylane_qulacs

dev1 = qml.device("default.qubit", wires=6)
@qml.qnode(dev1)
def pennylane_circuit0(input_array,learning_variables):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	# STATE PREPARATION
	for i in range(len(input_array)):
		qml.RY(input_array[i],wires=i)
	# APPLY forward sequence
	qml.RY(learning_variables[0],wires=0)
	qml.RY(learning_variables[1],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(learning_variables[2],wires=2)
	qml.RY(learning_variables[3],wires=3)
	qml.CNOT(wires=[2,3])
	qml.RY(learning_variables[4],wires=4)
	qml.RY(learning_variables[5],wires=5)
	qml.CNOT(wires=[5,4])
	qml.RY(learning_variables[6],wires=1)
	qml.RY(learning_variables[7],wires=3)
	qml.CNOT(wires=[1,3])
	qml.RY(learning_variables[8],wires=3)
	qml.RY(learning_variables[9],wires=4)
	qml.CNOT(wires=[3,4])
	qml.RY(learning_variables[10],wires=4)
	return(qml.expval(qml.PauliZ(wires=4)))

dev2 = qml.device("default.qubit", wires=8)
@qml.qnode(dev2)
def pennylane_circuit1(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	
	# STATE PREPARATION
	for i in range(8):
		qml.RY(edge[i],wires=i)
	# APPLY forward sequence
	# First Layer
	qml.RY(theta_learn[0],wires=0)
	qml.RY(theta_learn[1],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(theta_learn[2],wires=2)
	qml.RY(theta_learn[3],wires=3)
	qml.CNOT(wires=[3,2])
	qml.RY(theta_learn[4],wires=4)
	qml.RY(theta_learn[5],wires=5)
	qml.CNOT(wires=[4,5])
	qml.RY(theta_learn[6],wires=6)
	qml.RY(theta_learn[7],wires=7)
	qml.CNOT(wires=[7,6])
	# Second Layer
	qml.RY(theta_learn[8],wires=1)
	qml.RY(theta_learn[9],wires=2)
	qml.CNOT(wires=[1,2])
	qml.RY(theta_learn[10],wires=5)
	qml.RY(theta_learn[11],wires=6)
	qml.CNOT(wires=[6,5])
	# Third Layer
	qml.RY(theta_learn[12],wires=2)
	qml.RY(theta_learn[13],wires=5)
	qml.CNOT(wires=[2,5])
	#Last Layer
	qml.RY(theta_learn[14],wires=5)		
	return qml.expval(qml.PauliZ(wires=5))

dev3 = qml.device("default.qubit", wires=10)
@qml.qnode(dev3)
def pennylane_circuit2(edge,theta_learn):
	# STATE PREPARATION
	for i in range(10):
		qml.RY(edge[i],wires=i)
	# APPLY forward sequence
	# First Layer
	qml.RY(theta_learn[0],wires=0)
	qml.RY(theta_learn[1],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(theta_learn[2],wires=2)
	qml.RY(theta_learn[3],wires=3)
	qml.CNOT(wires=[3,2])
	qml.RY(theta_learn[4],wires=4)
	qml.RY(theta_learn[5],wires=5)
	qml.CNOT(wires=[5,4])
	qml.RY(theta_learn[6],wires=6)
	qml.RY(theta_learn[7],wires=7)
	qml.CNOT(wires=[6,7])
	qml.RY(theta_learn[8],wires=8)
	qml.RY(theta_learn[9],wires=9)
	qml.CNOT(wires=[8,9])
	# Second Layer
	qml.RY(theta_learn[10],wires=1)
	qml.RY(theta_learn[11],wires=2)
	qml.CNOT(wires=[1,2])
	qml.RY(theta_learn[12],wires=7)
	qml.RY(theta_learn[13],wires=9)
	qml.CNOT(wires=[9,7])
	# Third Layer
	qml.RY(theta_learn[14],wires=2)
	qml.RY(theta_learn[15],wires=4)
	qml.CNOT(wires=[2,4])
	# Forth Layer
	qml.RY(theta_learn[16],wires=4)
	qml.RY(theta_learn[17],wires=7)
	qml.CNOT(wires=[4,7])
	#Last Layer
	qml.RY(theta_learn[18],wires=7)		
	return qml.expval(qml.PauliZ(wires=7))

def pennylane_grad0(input_array,learning_variables):
	dcircuit = qml.grad(pennylane_circuit0, argnum=1)
	return (1-dcircuit(input_array,learning_variables))/2

def pennylane_grad1(input_array,learning_variables):
	dcircuit = qml.grad(pennylane_circuit1, argnum=1)
	return (1-dcircuit(input_array,learning_variables))/2

def pennylane_grad2(input_array,learning_variables):
	dcircuit = qml.grad(pennylane_circuit2, argnum=1)
	return (1-dcircuit(input_array,learning_variables))/2
	
pennylane_grads = []
input_array0 = np.random.rand(100,6)
input_array1 = np.random.rand(100,8)
input_array2 = np.random.rand(100,10)


learning_variables0 = np.random.rand(11) * np.pi * 2
learning_variables1 = np.random.rand(15) * np.pi * 2
learning_variables2 = np.random.rand(19) * np.pi * 2



t0 = time.time()

for i in range(len(input_array0)):
	pennylane_grads.append(pennylane_grad0(input_array0[i],learning_variables0))

t1 = time.time()

print('Hid0 run: Elapsed time: %.3f seconds' %(t1-t0))

t2 = time.time()

for i in range(len(input_array1)):
	pennylane_grads.append(pennylane_grad1(input_array1[i],learning_variables1))

t3 = time.time()

print('Hid1 run: Elapsed time: %.3f seconds' %(t3-t2))

t4 = time.time()

for i in range(len(input_array1)):
	pennylane_grads.append(pennylane_grad2(input_array2[i],learning_variables2))

t5 = time.time()

print('Hid2 run: Elapsed time: %.3f seconds' %(t5-t4))

with open('logs/log_pennylane.csv', 'a') as f:
		f.write('%.4f, %.4f, %.4f' %(t1-t0,t3-t2,t5-t4))