from keras.models import Model
from keras.layers import Input, Activation, Dropout, merge, TimeDistributed, Masking, Dense
from keras.layers.embeddings import Embedding

from keras.regularizers import l2
from keras.optimizers import Adam

from keras import backend as K

class TNN:

    def __init__(self, d_value = 0.2, act_f = "relu", batch_size = 16, input_size = 1600):
        self.d_value = d_value
        self.act_f = act_f
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_size2 = 1
        self.input_size3 = 2

    def buildModel(self):
        inputs = Input(shape = (self.input_size,))
        lay_0 = Dense(output_dim = 512, activation = self.act_f)(inputs)
        d_lay_0 = Dropout(self.d_value)(lay_0)
        lay_1 = Dense(output_dim = 256, activation = self.act_f)(d_lay_0)
        d_lay_1 = Dropout(self.d_value)(lay_1)
        lay_2 = Dense(output_dim = 128, activation = self.act_f)(d_lay_1)
        d_lay_2 = Dropout(self.d_value)(lay_2)
        lay_3 = Dense(output_dim = 64, activation = self.act_f)(d_lay_2)
        d_lay_3 = Dropout(self.d_value)(lay_3)
        lay_4 = Dense(output_dim = 32, activation = self.act_f)(d_lay_3)
        d_lay_4 = Dropout(self.d_value)(lay_4)
        lay_5 = Dense(output_dim = 1, activation = "sigmoid")(d_lay_4)

        model = Model(input = [inputs], output = lay_5)
        model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

        return model

    def buildModel2(self):
        inputs = Input(shape = (self.input_size,))
        lay_0 = Dense(output_dim = 512, activation = self.act_f)(inputs)
        d_lay_0 = Dropout(self.d_value)(lay_0)
        lay_1 = Dense(output_dim = 256, activation = self.act_f)(d_lay_0)
        d_lay_1 = Dropout(self.d_value)(lay_1)

        inputs3 = Input(shape = (self.input_size3,))
        aux_layer1 = Dense(output_dim = 128)
        aux_lay1 = aux_layer1(inputs3)
        aux_drop_l1 = Dropout(self.d_value)
        aux_d_l1 = aux_drop_l1(aux_lay1)
        merged = merge([d_lay_1, aux_d_l1], mode = 'concat')

        lay_2 = Dense(output_dim = 128, activation = self.act_f)(d_lay_1)
        d_lay_2 = Dropout(self.d_value)(lay_2)
        lay_3 = Dense(output_dim = 64, activation = self.act_f)(d_lay_2)
        d_lay_3 = Dropout(self.d_value)(lay_3)
        lay_4 = Dense(output_dim = 32, activation = self.act_f)(d_lay_3)
        d_lay_4 = Dropout(self.d_value)(lay_4)
        lay_5 = Dense(output_dim = 1, activation = "sigmoid")(d_lay_4)

        model = Model(input = [inputs, inputs3], output = lay_5)
        model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

        return model
