from keras.models import Model
from keras.layers import Input, Activation, Dropout, merge, TimeDistributed, Masking, Dense
from keras.layers.embeddings import Embedding

from keras.regularizers import l2
from keras.optimizers import Adam

from keras import backend as K

class TNN:

    def __init__(self, d_value = 0.2, act_f = "relu", batch_size = 16, input_size1 = 1600, input_size2 = 1200):
        self.d_value = d_value
        self.act_f = act_f
        self.batch_size = batch_size
        self.input_size = input_size1
        self.input_size2 = input_size2
        self.input_size3 = 2

    def buildModels(self):
        inputs = Input(shape = (self.input_size,))
        m1_layer0 = Dense(output_dim = 512, activation = self.act_f)
        m1_lay0 = m1_layer0(inputs)
        m1_drop_l0 = Dropout(self.d_value)
        m1_d_l0 = m1_drop_l0(m1_lay0)
        m1_layer1 = Dense(output_dim = 256, activation = self.act_f)
        m1_lay1 = m1_layer1(m1_d_l0)
        m1_drop_l1 = Dropout(self.d_value)
        m1_d_l1 = m1_drop_l1(m1_lay1)
        m1_layer2 = Dense(output_dim = 128, activation = self.act_f)
        m1_lay2 = m1_layer2(m1_d_l1)
        m1_drop_l2 = Dropout(self.d_value)
        m1_d_l2 = m1_drop_l2(m1_lay2)

        inputs2 = Input(shape = (self.input_size2,))
        m2_layer0 = Dense(output_dim = 512, activation = self.act_f)
        m2_lay0 = m2_layer0(inputs2)
        m2_drop_l0 = Dropout(self.d_value)
        m2_d_l0 = m2_drop_l0(m2_lay0)
        m2_layer1 = Dense(output_dim = 256, activation = self.act_f)
        m2_lay1 = m2_layer1(m2_d_l0)
        m2_drop_l1 = Dropout(self.d_value)
        m2_d_l1 = m2_drop_l1(m2_lay1)


        inputs3 = Input(shape = (self.input_size3,))
        aux_layer1 = Dense(output_dim = 128)
        aux_lay1 = aux_layer1(inputs3)
        aux_drop_l1 = Dropout(self.d_value)
        aux_d_l1 = aux_drop_l1(aux_lay1)
        merged = merge([m2_d_l1, aux_d_l1], mode = 'concat')
#        merged = merge([m2_d_l1, inputs3], mode = 'concat')
        m2_layer2 = Dense(output_dim = 128, activation = self.act_f)
        m2_lay2 = m2_layer2(merged)
        m2_drop_l2 = Dropout(self.d_value)
        m2_d_l2 = m2_drop_l2(m2_lay2)

        layer_3 = Dense(output_dim = 64, activation = self.act_f)
        m1_lay3 = layer_3(m1_d_l2)
        m2_lay3 = layer_3(m2_d_l2)
        d_lay_3 = Dropout(self.d_value)
        m1_d_l3 = d_lay_3(m1_lay3)
        m2_d_l3 = d_lay_3(m2_lay3)
        layer_4 = Dense(output_dim = 32, activation = self.act_f)
        m1_lay4 = layer_4(m1_d_l3)
        m2_lay4 = layer_4(m2_d_l3)
        d_lay_4 = Dropout(self.d_value)
        m1_d_l4 = d_lay_4(m1_lay4)
        m2_d_l4 = d_lay_4(m2_lay4)
        layer_5 = Dense(output_dim = 1, activation = "sigmoid")
        m1_lay5 = layer_5(m1_d_l4)
        m2_lay5 = layer_5(m2_d_l4)
        model1 = Model(input = inputs, output = m1_lay5)
        model1.compile(optimizer = 'adam', loss = 'mean_absolute_error')
        model2 = Model(input = [inputs2, inputs3], output = m2_lay5)
        model2.compile(optimizer = 'adam', loss = 'mean_absolute_error')
        return model1, model2
