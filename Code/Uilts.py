from matplotlib.colors import ListedColormap
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import concatenate, Flatten, Add, Reshape, Dense, GlobalAvgPool1D, multiply
from keras.layers import Lambda
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model

def cseis():
    seis=np.concatenate(
(np.concatenate((0.5*np.ones([1,40]),np.expand_dims(np.linspace(0.5,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((0.25*np.ones([1,40]),np.expand_dims(np.linspace(0.25,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((np.zeros([1,40]),np.expand_dims(np.linspace(0,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose()),axis=1)
    return ListedColormap(seis)

def patch(A,l1,l2,o1,o2):

    n1,n2=np.shape(A);
    tmp=np.mod(n1-l1,o1)
    if tmp!=0:
        ##print(np.shape(A), o1-tmp, n2)
        A=np.concatenate([A,np.zeros((o1-tmp,n2))],axis=0)

    tmp=np.mod(n2-l2,o2);
    if tmp!=0:
        A=np.concatenate([A,np.zeros((A.shape[0],o2-tmp))],axis=-1);


    N1,N2 = np.shape(A)
    X=[]
    for i1 in range (0,N1-l1+1, o1):
        for i2 in range (0,N2-l2+1,o2):
            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1));
            X.append(tmp);
    X = np.array(X)
    #print(np.shape(X))
    return X[:,:,0]


def patch_inv(X1, n1, n2, l1, l2, o1, o2):
    tmp1 = np.mod(n1 - l1, o1)
    tmp2 = np.mod(n2 - l2, o2)
    if (tmp1 != 0) and (tmp2 != 0):
        A = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))
        mask = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))

    if (tmp1 != 0) and (tmp2 == 0):
        A = np.zeros((n1 + o1 - tmp1, n2))
        mask = np.zeros((n1 + o1 - tmp1, n2))

    if (tmp1 == 0) and (tmp2 != 0):
        A = np.zeros((n1, n2 + o2 - tmp2))
        mask = np.zeros((n1, n2 + o2 - tmp2))

    if (tmp1 == 0) and (tmp2 == 0):
        A = np.zeros((n1, n2))
        mask = np.zeros((n1, n2))

    N1, N2 = np.shape(A)
    ids = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            ##print(i1,i2)
            #       [i1,i2,ids]
            A[i1:i1 + l1, i2:i2 + l2] = A[i1:i1 + l1, i2:i2 + l2] + np.reshape(X1[:, ids], (l1, l2))
            mask[i1:i1 + l1, i2:i2 + l2] = mask[i1:i1 + l1, i2:i2 + l2] + np.ones((l1, l2))
            ids = ids + 1

    A = A / mask;
    A = A[0:n1, 0:n2]

    return A

def logcosh_loss(y_true, y_pred):
    return K.mean(K.log(tf.cosh(y_pred - y_true)))


def frequency_domain_constraint(
        y_true, y_pred,
        low_freq_cutoff=40, high_freq_cutoff=80,
        low_freq_weight=2, mid_freq_weight=5.0, high_freq_weight=1.5,
        dynamic_weights=True):

    # Fourier Transform
    y_true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
    y_pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))

    # Normalized fft
    y_true_fft /= tf.cast(tf.size(y_true), tf.complex64)
    y_pred_fft /= tf.cast(tf.size(y_pred), tf.complex64)

    # Frequency masks
    low_freq_mask = tf.concat([
        tf.ones_like(y_true_fft[:, :low_freq_cutoff]),
        tf.zeros_like(y_true_fft[:, low_freq_cutoff:])
    ], axis=1)

    mid_freq_mask = tf.concat([
        tf.zeros_like(y_true_fft[:, :low_freq_cutoff]),
        tf.ones_like(y_true_fft[:, low_freq_cutoff:high_freq_cutoff]),
        tf.zeros_like(y_true_fft[:, high_freq_cutoff:])
    ], axis=1)

    high_freq_mask = tf.concat([
        tf.zeros_like(y_true_fft[:, :high_freq_cutoff]),
        tf.ones_like(y_true_fft[:, high_freq_cutoff:])
    ], axis=1)

    y_true_low_freq = tf.multiply(y_true_fft, low_freq_mask)
    y_pred_low_freq = tf.multiply(y_pred_fft, low_freq_mask)
    y_true_mid_freq = tf.multiply(y_true_fft, mid_freq_mask)
    y_pred_mid_freq = tf.multiply(y_pred_fft, mid_freq_mask)
    y_true_high_freq = tf.multiply(y_true_fft, high_freq_mask)
    y_pred_high_freq = tf.multiply(y_pred_fft, high_freq_mask)

    # Dynamically adjust frequency band weights
    if dynamic_weights:
        low_freq_energy = tf.reduce_mean(tf.abs(y_true_low_freq))
        mid_freq_energy = tf.reduce_mean(tf.abs(y_true_mid_freq))
        high_freq_energy = tf.reduce_mean(tf.abs(y_true_high_freq))

        low_freq_weight = low_freq_weight / (low_freq_energy + 1e-6)
        mid_freq_weight = mid_freq_weight / (mid_freq_energy + 1e-6)
        high_freq_weight = high_freq_weight / (high_freq_energy + 1e-6)

    def logcosh(x):
        return tf.reduce_mean(tf.math.log(tf.math.cosh(x)))

    low_freq_loss = logcosh(tf.abs(y_true_low_freq - y_pred_low_freq)) / tf.cast((low_freq_cutoff + 1), tf.float32)
    mid_freq_loss = logcosh(tf.abs(y_true_mid_freq - y_pred_mid_freq)) / tf.cast(
        (high_freq_cutoff - low_freq_cutoff + 1), tf.float32)
    high_freq_loss = logcosh(tf.abs(y_true_high_freq - y_pred_high_freq)) / tf.cast(
        (tf.shape(y_true_fft)[1] - high_freq_cutoff + 1), tf.float32)

    total_loss = (low_freq_weight * low_freq_loss +
                  mid_freq_weight * mid_freq_loss +
                  high_freq_weight * high_freq_loss)

    return total_loss

def amplitude_weighted_loss(y_true, y_pred):
    weights = tf.reduce_mean(tf.abs(y_true), axis=-1)
    weights = tf.expand_dims(weights, axis=-1)
    loss = weights * tf.keras.losses.mean_squared_error(y_true, y_pred)
    return tf.reduce_mean(loss)

def combined_loss(y_true, y_pred):
    return logcosh_loss(y_true, y_pred) + 0.2 * amplitude_weighted_loss(y_true, y_pred) + 0.4*frequency_domain_constraint(y_true, y_pred)


def Block(inp, D):
    x = Dense(D, activation='relu')(inp)
    x = Reshape((x.shape[-1], 1))(x)
    return x


def FAblock(y, D):
    # Split the data into two channels
    s0 = Lambda(lambda x: x[:, :, 0])(y)
    s1 = Lambda(lambda x: x[:, :, 1])(y)

   # noisy data channel
    B1 = Block(s0, D)
    # cwt data channel
    B2 = Block(s1, D)

    # Connect two channels
    B = concatenate([B1, B2], axis=-1)

    filters = B.shape[-1]

    sa_shape = (1, filters)
    avg_pool = GlobalAvgPool1D()(B)
    reshape = Reshape(sa_shape)(avg_pool)

    # Generating weights for the selective attention mechanism
    x1 = Dense(filters // 2, activation='relu')(reshape)
    x1 = Dense(filters * 2, activation='relu')(x1)

    attention_weights = Dense(filters, activation='softmax')(x1)

    # Combining input features and attention weights
    scaled = multiply([B, attention_weights])
    output = Add()([B, scaled])

    return output


def PDDNet(CWT, NOISY, EPOCHNO, BATCHSIZE, w1, w2, s1z, s2z, D1):
    # Normalize the CWT Scale

    ma = np.max(np.abs(CWT))
    dataInput = CWT / ma


    # Patching the CWT SCALE.
    dataInputP = patch(dataInput, w1, w2, s1z, s2z)
    dataInput2 = np.reshape(dataInputP, (dataInputP.shape[0], w1 * w2, 1))
    print(dataInputP.shape)
    # Patching the Band-pass Filtered Data.
    dataInputF = patch(NOISY, w1, w2, s1z, s2z)
    dataInputF2 = np.reshape(dataInputF, (dataInputF.shape[0], w1 * w2, 1))

    # Setting the Inputs to the DL Model.
    input_shape = (w1, w1, 1)
    inp1 = layers.Input(shape=(w1 * w2, 1), name='input_layer1')
    inp2 = layers.Input(shape=(w1 * w2, 1), name='input_layer2')

    # Setting the Number of Neurons for the Encoder/Decoder.
    D2 = int(D1 / 2)
    D3 = int(D2 / 2)
    D4 = int(D3 / 2)
    D5 = int(D4 / 2)

    # Concatenate the Patches.
    inp4 = concatenate([inp1, inp2])

    e1 = FAblock(inp4, D1)
    e2 = FAblock(e1, D2)
    e3 = FAblock(e2, D3)
    e4 = FAblock(e3, D4)
    e5 = FAblock(e4, D5)

    d2 = FAblock(e5, D4)
    d3 = FAblock(d2, D3)
    d4 = FAblock(d3, D2)
    d5 = FAblock(d4, D1)
    d6 = FAblock(d5, D1)

    # Output Layer
    d6 = Flatten()(d6)
    y = Dense(w1 * w2, activation='linear')(d6)

    # Creating the Model
    model = Model(inputs=[inp1, inp2], outputs=[y])
    model.compile(optimizer='adam', loss=combined_loss)


    # Schedule the Learning Rate.
    A = 40
    def lr_schedule(epoch):
        initial_lr = 1e-3

        if epoch <= A:
            lr = initial_lr
        elif epoch <= A + 10:
            lr = initial_lr / 2
        elif epoch <= A + 30:
            lr = initial_lr / 10
        else:
            lr = initial_lr / 20
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=0.5,
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6,
                                   monitor='loss')
    early_stopping_monitor = EarlyStopping(monitor='loss', patience=5)
    checkpoint = ModelCheckpoint('./Epoch/best_model_2dfield.h5',
                                 monitor='loss',
                                 mode='auto',
                                 verbose=0,
                                 save_best_only=True)
    callbacks = [lr_scheduler, early_stopping_monitor, checkpoint]

    # Print the Model Summary.
    model.summary()
    # Training the DL Model
    history = model.fit([dataInputF2, dataInput2], [dataInputF], epochs=EPOCHNO, batch_size=BATCHSIZE, shuffle=True,
                        callbacks=callbacks, verbose=1)

    # Loading the optimal Model.
    #model.load_weights('./Epoch/best_model_2dfield.h5')
    base_model = load_model('Epoch/best_model_2dfield.h5', custom_objects={'combined_loss': combined_loss})

    predict = base_model.predict([dataInputF2, dataInput2], batch_size=BATCHSIZE)
    print('The shape of ouput data is:',predict.shape)

    # Unpatching to Reconstruct the Input Data.
    predict_ori = np.reshape(predict, (predict.shape[0], w1 * w2))
    predict_ori = np.transpose(predict_ori)
    n1, n2 = np.shape(NOISY)
    predict_data = patch_inv(predict_ori, n1, n2, w1, w2, s1z, s2z)
    predict_data = np.array(predict_data)

    K.clear_session()

    print('The training and testing stages are completed')
    return predict_data