import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input , Lambda
import matplotlib.pyplot as plt
from ModulationPy import QAMModem
import time


noBitsPerSymbol = 2       #MODULATION DEPENDANT 

SNR = 25
N = 64
learning_Rate = 0.0001
opt = keras.optimizers.Adam(learning_Rate)
initializer = tf.keras.initializers.GlorotUniform()
batchsize = 400
lamba = 0.002
do = 0

def Modulation(random_Data, N, o):
    data_non_modulated = random_Data
    data = np.zeros((o,N),dtype=complex)
    modem = QAMModem(4,                  #QPSK modulation
                 gray_map=True, 
                 bin_input=True)
    for i in range(o):
        data[i,:] = modem.modulate(data_non_modulated[i,:])
    return data

def IFFT(tensor):
    return tf.signal.ifft(tf.complex(tensor[:, 0:N], tensor[:, N:]))  #non trainable parameters 

def Channel_and_Noise(tensor):
    signal_avg = tf.reduce_mean(tf.square(tf.abs(tensor)))
    noise_avg = signal_avg / 10**(SNR/10)
    rayleigh = (1.0/2.0)*tfp.random.rayleigh((batchsize, N))
    noise_real = (tf.sqrt(noise_avg / 2.0))*tf.random.normal((batchsize , N), dtype=tf.float32)
    noise_imag = (tf.sqrt(noise_avg / 2.0))*tf.random.normal((batchsize , N), dtype=tf.float32)

    #Corruption = Channel Effect + AWGN
    corruption =  tf.complex(tf.divide(noise_real, rayleigh) , tf.divide(noise_imag, rayleigh))
    #Equalizing done on Signal
    x = tensor + corruption
    #Recieving 
    x_fft = tf.signal.fft(x)

    return tf.keras.layers.Concatenate(axis = -1) ([tf.math.real(x_fft), tf.math.imag(x_fft)])

def PAPR_of_Data(data, o):
    PAPR = np.zeros(o, dtype=complex)
    for i in range(o):
        x = np.square(np.abs(data[i,:]))
        PAPR[i] = np.max(x) / np.mean(x)
    PAPR_db = 10*np.log10(PAPR)
    return PAPR_db 

def PAPR_of_Data_IFFT(data, o):
    PAPR = np.zeros(o, dtype=complex)
    for i in range(o):
        x = np.square(np.abs(np.fft.ifft(data[i,:])))
        PAPR[i] = np.max(x) / np.mean(x)
    PAPR_db = 10*np.log10(PAPR)
    return PAPR_db 

def papr(y_true, y_pred):
    x = tf.square(tf.abs(y_pred))
    return 10*tf.experimental.numpy.log10(tf.reduce_mean(tf.reduce_max(x, axis = 1) / tf.reduce_mean(x, axis = 1)))


inputs = Input(shape = (N*2,))
#ENCODER
x1 = Dense(2048, kernel_initializer=initializer, bias_initializer='random_normal') (inputs)
x2 = BatchNormalization() (x1)
x2 = Dropout(do) (x2)
x3 = Activation('relu') (x2)

x4 = Dense(2048, kernel_initializer=initializer, bias_initializer='random_normal') (x3)
x5 = BatchNormalization() (x4)
x5 = Dropout(do) (x5)
x6 = Activation('relu')(x5)

x7 = Dense(2048, kernel_initializer=initializer, bias_initializer='random_normal') (x6)
x8 = BatchNormalization() (x7)
x8 = Dropout(do) (x8)
x9 = Activation('relu')(x8)

x10 = Dense(2048, kernel_initializer=initializer, bias_initializer='random_normal') (x9)
x11 = BatchNormalization() (x10)
x11 = Dropout(do) (x11)
x12 = Activation('relu')(x11)

x13 = Dense(N*2, kernel_initializer=initializer, bias_initializer='random_normal', activation= 'tanh') (x12)

encoder = Lambda(IFFT, name='encoder') (x13)

#RECEIVER
decoder = Lambda(Channel_and_Noise) (encoder)

#DECODER
y1 = Dense(2048, kernel_initializer=initializer, bias_initializer='random_normal') (decoder)
y2 = BatchNormalization() (y1)
y2 = Dropout(do) (y2)
y3 = Activation('relu') (y2)

y4 = Dense(2048, kernel_initializer=initializer, bias_initializer='random_normal') (y3)
y5 = BatchNormalization() (y4)
y5 = Dropout(do) (y5)
y6 = Activation('relu')(y5)

y7 = Dense(2048, kernel_initializer=initializer, bias_initializer='random_normal') (y6)
y8 = BatchNormalization() (y7)
y8 = Dropout(do) (y8)
y9 = Activation('relu')(y8)

y10 = Dense(2048, kernel_initializer=initializer, bias_initializer='random_normal') (y9)
y11 = BatchNormalization() (y10)
y11 = Dropout(do) (y11)
y12 = Activation('relu')(y11)

outputs = Dense(N*2, kernel_initializer=initializer, bias_initializer='random_normal', activation= 'sigmoid', name= 'decoder') (y12)

model = keras.Model(inputs=inputs, outputs=[encoder, outputs])


dataset = np.random.randint(2, size=(500000, N*2))
valset = np.random.randint(2, size=(50000, N*2))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(optimizer= opt, loss= {'encoder' : papr, 'decoder' : 'binary_crossentropy'}, loss_weights=[lamba, 1], metrics={'decoder': tf.keras.metrics.BinaryAccuracy()})
print(model.summary())
history = model.fit(dataset, [dataset, dataset], epochs= 400, batch_size = 400, shuffle= True, callbacks=[callback], validation_data = [valset, valset])


size = 100000
dataset = np.random.randint(2, size=(size, N*2))
st = time.time()
NET, blabla = model.predict(dataset, batch_size = batchsize)  # adapt size
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
Modulated_Data = Modulation(dataset, N, size)


PRNET = PAPR_of_Data(NET, size)
OFDM = PAPR_of_Data_IFFT(Modulated_Data, size)


plt.figure(1)
plt.xlabel('PAPR in db')
plt.ylabel('Probability (%)')
plt.title("PRNET vs OFDM PAPR Simulation")
plt.yscale('log')
plt.ylim(0.001, 1.2)
plt.grid(True, which="both")
m, n = np.histogram(PRNET)
plt.plot(n[1:], 1 - np.cumsum(m)/size, 'r', label = 'PRNET')

m1, n1 = np.histogram(OFDM)
plt.plot(n1[1:], 1 - np.cumsum(m1)/size, 'b', label = 'Classical OFDM')

plt.show()
