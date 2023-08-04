from tensorflow.keras import layers
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
               
        return output