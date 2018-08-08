from keras.models import Model
from keras.engine.topology import Input
from keras.layers import Dense, Conv2DTranspose, Conv2D, Reshape, Activation, LeakyReLU, BatchNormalization, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K

class factory:

    @staticmethod
    def create_generator(in_layer):
        #conv_filters = [512, 256, 128, 3]
        conv_filters = [128, 64, 32, 3]
        conv_strides = [2, 2, 2, 1]
        
        x = in_layer
        x = Dense(8 * 8 * 1024)(x)
        x = Reshape((8, 8, 1024))(x)

        for filters, strides in zip(conv_filters, conv_strides):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2DTranspose(filters = filters,
                                kernel_size = 5,
                                strides = strides,
                                padding = 'same')(x)

        out_layer = Activation('tanh')(x)

        model = Model(in_layer, out_layer, name = 'gen')
        return model

    @staticmethod
    def create_discriminator(in_layer):
        conv_filters = [32, 64, 128, 256]
        conv_strides = [2, 2, 2, 1]

        x = in_layer

        for filters, strides in zip(conv_filters, conv_strides):
            x = LeakyReLU(alpha = 0.2)(x)
            x = Conv2D(filters = filters,
                       kernel_size = 5,
                       strides = strides,
                       padding = 'same')(x)

        x = Flatten()(x)
        x = Dense(1)(x)
        out_layer = Activation('sigmoid')(x)

        model = Model(in_layer, out_layer, name = 'disc')
        return model

    @staticmethod
    def create_adversarial_pair():
        in_layer_gen = Input(shape = (100,))
        gen = factory.create_generator(in_layer_gen)
        
        in_layer_disc = Input(shape = (64, 64, 3))
        disc = factory.create_discriminator(in_layer_disc)
        #opt = Adam(lr = 0.0001, beta_1 = 0.5, beta_2 = 0.999, epsilon = K.epsilon(), decay = 0.0)
        opt = RMSprop(lr = 2e-4, decay = 6e-8)
        disc.compile(loss = 'binary_crossentropy',
                     optimizer = opt,
                     metrics = ['accuracy'])

        disc.trainable = False
        adv = Model(in_layer_gen, disc(gen(in_layer_gen)))
        #opt = Adam(lr = 0.0001, beta_1 = 0.5, beta_2 = 0.999, epsilon = K.epsilon(), decay = 0.0)
        opt = RMSprop(lr = 1e-4, decay = 1e-8)
        adv.compile(loss = 'binary_crossentropy',
                    optimizer = opt,
                    metrics = ['accuracy'])

        return gen, disc, adv
