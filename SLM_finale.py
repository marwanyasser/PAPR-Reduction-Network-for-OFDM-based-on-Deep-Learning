from re import I
from turtle import left, position, shapesize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy.core.fromnumeric import cumsum, mean
from numpy.lib.type_check import real
#from numpy.fft import fft, ifft
from ModulationPy import QAMModem,PSKModem
#import scipy
#from scipy import signal


#FUNCTION TO MODULATE
def Modulation(random_Data, M, o):   #62
    data_non_modulated = random_Data
    data = np.zeros((M,o),dtype=complex)
    modem = PSKModem(8,                  #QPSK modulation
                 gray_map=True, 
                 bin_input=True)
    for i in range(o):
        data[:,i] = modem.modulate(data_non_modulated[:,i])
    return data
#print(Modulation(random_Data,M,o))  
#print(len(Modulation(random_Data,M,o))) #31

def Alamouti_DATA(data):
    z=np.zeros((1,o), dtype=complex)   
    data_conj = np.zeros((M,o), dtype=complex)
    for i in range (o):
        data_conj[:,i] = np.conj(data[:,i]) #take conjugate to data
    data_reverseConj = data_conj[-1: :-1]
    data_beforeIfft = np.vstack((z,data_reverseConj))   #add zeros on top of reverseconj
    data_beforeIfft = np.vstack((data,data_beforeIfft)) #add data on top of databefore
    data_beforeIfft = np.vstack((z,data_beforeIfft))    #add zeros on top of databefore

    return(data_beforeIfft)          #amplitude?

#print(len(Alamouti_DATA(Modulation(random_Data,M,o)))) #64


def IFFT (data, o): 
    ifft_data = np.zeros((N,o), dtype= complex)
    for i in range(o):
        ifft_data[:,i] = np.fft.ifft(data[:,i], N)
    return(ifft_data)

def IFFT_Symbol (data):  
    temp = np.fft.ifft(data, N)
    return(temp)
#print(len(IFFT(Alamouti_DATA(Modulation(random_Data,M,o))))) #64
#data_IFFT = IFFT(Alamouti_DATA(Modulation(random_Data,M,o)), o)
#print(data_IFFT)
#print(len(data_IFFT))


def PAPR_of_each_symbol(data, o):
    PAPR = np.zeros(o, dtype=complex)
    for i in range(o):
        PAPR[i] = np.max(data[:,i]*np.conj(data[:,i])) / np.mean(data[:,i]*np.conj(data[:,i]))
    PAPR_db = 10*np.log10(PAPR)
    print(max(PAPR_db))
    return PAPR_db
  
def PAPR_of_Symbol(symbol):
    PAPR = np.max(symbol*np.conj(symbol)) / np.mean(symbol*np.conj(symbol))
    PAPR_db = 10*np.log10(PAPR)
    return PAPR_db


def Generate_Phase_Sequences(U,N):
    phases = np.array([1, -1, 1j, -1j])   
    phase_sequence = np.zeros((U, N) ,dtype=complex)    #Column vector
    for l in range(U):
        phase_sequence[l, :] = np.random.choice(phases, N)
    return phase_sequence



def PhaseShifter(symbol, phase_sequence, N):  #multiply each symbol with one of the phase sequences 
    return np.multiply(phase_sequence, symbol) 
    #DEBUGGED

def OFDM_Simulation_SLM(modulated_data, N, o, U):     #where U is the number of phase shift trials 
    Phase_Sequence_patterns = Generate_Phase_Sequences(U,N)
    phase_index_optimum = np.zeros(o, dtype=int)
    papr_of_shifted_symbol = np.zeros(U, dtype=complex)
    PAPR_SLM = np.zeros(o, dtype=complex)
    for j in range(o):
        for x in range(U):          #number of phase sequence trials
            papr_of_shifted_symbol[x] = PAPR_of_Symbol(IFFT_Symbol(PhaseShifter(modulated_data[:, j], Phase_Sequence_patterns[x, :], N)))
        phase_index_optimum[j] = np.argmin(papr_of_shifted_symbol)
    PAPR_noSLM = PAPR_of_each_symbol(IFFT(modulated_data, o), o)  
    shifted_data = np.zeros((N,o),dtype=complex)
    for j in range(o):
        phase = Phase_Sequence_patterns[phase_index_optimum[j],:]
        shifted_data[:, j] = PhaseShifter(modulated_data[:,j], phase, N)
    PAPR_SLM = PAPR_of_each_symbol(IFFT(shifted_data, o), o)
    return PAPR_noSLM, PAPR_SLM


def CCDF_Plotter(PAPR_SLM, PAPR_noSLM, U):
    x, n = np.histogram(PAPR_noSLM, bins= 1000)
    x2, n2 = np.histogram(PAPR_SLM, bins = 1000)
    plt_label = "Conventional SLM U = " + str(U)
    if U == 2:
        plt.plot(n[1:], 1 - np.cumsum(x)/o, 'k', label = 'Normal OFDM')
        color = 'b'
    elif U == 4:
        color = 'g'
    elif U == 8:
        color = 'r'
    elif U == 16:
        color = '#FF7F00'
    elif U == 32:
        color = '#FFD700'
    elif U == 64:
        color = 'b--'
    plt.plot(n2[1:], 1 - np.cumsum(x2)/o, color, label = plt_label)



N = 64   #SUBCARRIERS
o = 30000  #Number of Symbols
U = {2, 4, 8, 16, 32, 64}

noBitsPerSymbol = 2       #MODULATION DEPENDANT 
#print(M)
K = N*noBitsPerSymbol     # size for modulation 62
# GENERATE DATA
random_Data = np.random.randint(2, size=(K, o)) #62
#REAL DATA
Modulated_Data = Modulation(random_Data,N,o)
fig = plt.figure()
for i in U:
    PAPR_noSLM, PAPR_SLM = OFDM_Simulation_SLM(Modulated_Data, N, o, i)
    CCDF_Plotter(PAPR_SLM, PAPR_noSLM, i)
plt.ylabel('Probability of <= x')
plt.xlabel('PAPR in db')
plt.title("OFDM PAPR Simulation at N= "+ str(N) + " U= "+ str(U))
plt.yscale('log')
plt.ylim(10e-5, 1.2)
plt.grid(True, which="both")
plt.legend(loc='upper right', fontsize='small')
plt.show()