from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K

class Attention(layers.Layer):
    
    def __init__(self, units):
        super(Attention, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1], self.units), initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1], self.units), initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
               
        return K.sum(output, axis=1)
    
def stem(x, num_blocks=8, num_filters=16, strides=2):
    kernels = [(11, 9), (9, 7), (7, 5), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)]
    filters = [num_filters, num_filters, num_filters*2, num_filters*2, num_filters*4, num_filters*4, num_filters*6, num_filters*6]
    for i in range(num_blocks):
        x = layers.Conv1D(filters=filters[i], kernel_size=kernels[i][0], padding="same")(x)
        x = layers.Conv1D(filters=filters[i], kernel_size=kernels[i][1], padding="same")(x)
        x = layers.Activation(activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=strides+1, strides=strides, padding="same")(x)

    return layers.LayerNormalization(epsilon=1e5)(x)

def ModelCLA(input_shape):
    inp = layers.Input(input_shape)
    x = stem(inp, num_blocks=8, num_filters=32, strides=2)

    x = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(x)
    x = layers.add([x, Attention(256)(x)])
    
    # arousals
    # x1 = layers.Dropout(0.5)(x)
    out1 = layers.Conv1D(filters=1, kernel_size=1, activation="sigmoid", name="arousal")(x)

    # stages
    # x2 = layers.Dropout(0.5)(x)
    out2 = layers.Conv1D(filters=5, kernel_size=1, activation="softmax", name="stage")(x)

    # create the model
    model = models.Model(inputs=inp, outputs=[out1, out2])

    return model

def ModelCL(input_shape):
    inp = layers.Input(input_shape)
    x = stem(inp, num_blocks=8, num_filters=32, strides=2)

    x = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(x)
    
    # arousals
    #x1 = layers.Dropout(0.5)(x)
    out1 = layers.Conv1D(filters=1, kernel_size=1, activation="sigmoid", name="arousal")(x)

    # stages
    #x2 = layers.Dropout(0.5)(x)
    out2 = layers.Conv1D(filters=5, kernel_size=1, activation="softmax", name="stage")(x)

    # create the model
    model = models.Model(inputs=inp, outputs=[out1, out2])

    return model

def ModelC(input_shape):
    inp = layers.Input(input_shape)
    x = stem(inp, num_blocks=8, num_filters=32, strides=2)
    
    # arousals
    #x1 = layers.Dropout(0.5)(x)
    out1 = layers.Conv1D(filters=1, kernel_size=1, activation="sigmoid", name="arousal")(x)

    # stages
    #x2 = layers.Dropout(0.5)(x)
    out2 = layers.Conv1D(filters=5, kernel_size=1, activation="softmax", name="stage")(x)

    # create the model
    model = models.Model(inputs=inp, outputs=[out1, out2])

    return model

def ModelCA(input_shape):
    inp = layers.Input(input_shape)
    x = stem(inp, num_blocks=8, num_filters=32, strides=2)

    x = layers.add([x, Attention(192)(x)])
    
    # arousals
    #x1 = layers.Dropout(0.5)(x)
    out1 = layers.Conv1D(filters=1, kernel_size=1, activation="sigmoid", name="arousal")(x)

    # stages
    #x2 = layers.Dropout(0.5)(x)
    out2 = layers.Conv1D(filters=5, kernel_size=1, activation="softmax", name="stage")(x)

    # create the model
    model = models.Model(inputs=inp, outputs=[out1, out2])

    return model