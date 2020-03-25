import csv
import numpy as np
import matplotlib.pyplot as plt 

log_loc = 'logs/'
pdf_location = 'pdf/'

with open(log_loc+'log_pennylane.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	duration0 = np.array(list(reader)).astype(float) 
with open(log_loc+'log_pennylane_qulacs.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	duration1 = np.array(list(reader)).astype(float) 

x = [6,8,10]
label1 = 'Pennylane'
label2 = 'Pennylane + Qulacs'

plt.plot(x,duration0[0,:],label=label1,c='darkorange')
plt.plot(x,duration1[0,:],label=label2,c='navy')
plt.title('Quantum Circuit Simulation Benchmark')
plt.xlabel(r'$N_{qubits}$')
plt.ylabel('Time [seconds]')
plt.legend()
plt.savefig(pdf_location+'benchmark.pdf')