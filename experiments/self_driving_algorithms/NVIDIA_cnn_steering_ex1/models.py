"""
Models Developed
-------------------------
Created on Mon Apr 19 19:39:37 2021
@author: kevin machado gamboa
"""
# -----------------------------------------------------------------------------
#                                Libraries
# -----------------------------------------------------------------------------
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Conv2DTranspose, UpSampling2D

# %%
# -----------------------------------------------------------------------------
#                               Model Structure
# -----------------------------------------------------------------------------
class multibranch_car_control_model:
    def __init__(self, input_shape):
        # Initialise the model
        self.input_shape = input_shape
        print('building architecture .. ')
        self.build_multivariable_model()
        print('compiling model .. ')
        self.compile()
        print('finished .. ')

    def trunk_branch(self, x_in):
        base = Conv2D(24, 5, strides=(2, 2), activation='elu', padding='same')(x_in)
        base = BatchNormalization()(base)
        base = Conv2D(36, 5, strides=(2, 2), activation='elu', padding='same')(base)
        base = BatchNormalization()(base)
        base = Conv2D(48, 5, strides=(2, 2), activation='elu', padding='same')(base)
        base = BatchNormalization()(base)
        base = Conv2D(64, 5, strides=(2, 2), activation='elu', padding='same')(base)
        base = BatchNormalization()(base)
        base = Conv2D(48, 5, strides=(2, 2), activation='elu', padding='same')(base)
        base = BatchNormalization()(base)
        base = Conv2D(36, 5, strides=(2, 2), activation='elu', padding='same')(base)
        base = BatchNormalization()(base)
        base = Conv2D(24, 5, strides=(2, 2), activation='elu', padding='same')(base)
        base = Dropout(0.5)(base)
        base = Flatten()(base)
        # Steering output
        #steer = Dense(100, activation='relu')(base)
        steer = Dense(1, activation='linear', name='steering_output')(base)
        # Throttle output
        throttle = Dense(1, activation='sigmoid', name='throttle_output')(base)
        # Break output
        brake = Dense(1, activation='sigmoid', name='break_output')(base)

        return steer, throttle, brake

    def build_multivariable_model(self):
        x_in = tf.keras.Input(self.input_shape)
        out = self.trunk_branch(x_in)
        self.model = tf.keras.models.Model(
            inputs=x_in,
            outputs=[out[0], out[1], out[2]],
            name='multibranch_car_control_model')

    def compile(self):
        # Model Parameters
        optimizer_f = tf.keras.optimizers.Adam(lr=1.0e-3)
        loss_f = tf.keras.losses.MeanSquaredError()
        # Compiling model
        self.model.compile(optimizer=optimizer_f,
                           loss=loss_f,
                           metrics='mse'
                            )

# %%
# -----------------------------------------------------------------------------
#                               Model Structure
# -----------------------------------------------------------------------------
pool_size = (2, 2)

class capstone_car_control_model:
    def __init__(self, input_shape):
        # Initialise the model
        self.input_shape = input_shape
        print('building architecture .. ')
        self.build_multivariable_model()
        print('compiling model .. ')
        self.compile()
        print('finished .. ')


    def trunk_branch(self, x_in):
        base = BatchNormalization()(x_in)
        base = Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1')(base)
        base = Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2')(base)

        # Pooling 1
        base = MaxPooling2D(pool_size=pool_size)(base)

        # Conv Layer 3
        base = Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv3')(base)
        base = Dropout(0.2)(base)

        # Conv Layer 4
        base = Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv4')(base)
        base = Dropout(0.2)(base)

        # Conv Layer 5
        base = Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv5')(base)
        base = Dropout(0.2)(base)

        # Pooling 2
        base = MaxPooling2D(pool_size=pool_size)(base)

        # Conv Layer 6
        base = Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv6')(base)
        base = Dropout(0.2)(base)

        # Conv Layer 7
        base = Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv7')(base)
        base = Dropout(0.2)(base)

        # Pooling 3
        base = MaxPooling2D(pool_size=pool_size)(base)

        base = Flatten()(base)
        # Steering output
        #steer = Dense(100, activation='relu')(base)
        steer = Dense(1, activation='linear', name='steering_output')(base)
        # Throttle output
        throttle = Dense(1, activation='sigmoid', name='throttle_output')(base)
        # Break output
        brake = Dense(1, activation='sigmoid', name='break_output')(base)

        return steer, throttle, brake

    def build_multivariable_model(self):
        x_in = tf.keras.Input(self.input_shape)
        out = self.trunk_branch(x_in)
        self.model = tf.keras.models.Model(
            inputs=x_in,
            outputs=[out[0], out[1], out[2]],
            name='multibranch_car_control_model')

    def compile(self):
        # Model Parameters
        optimizer_f = tf.keras.optimizers.Adam(lr=1.0e-3)
        loss_f = tf.keras.losses.MeanSquaredError()
        # Compiling model
        self.model.compile(optimizer=optimizer_f,
                           loss=loss_f,
                           metrics='mse'
                            )

# %%
# -----------------------------------------------------------------------------
#                              Model Structure 2
# -----------------------------------------------------------------------------
class multivariable_car_control_model:
    def __init__(self, input_shape):
        # Initialise the model
        self.input_shape = input_shape
        self.build_multivariable_model()

    def steering_branch(self, x_in):
        x = Conv2D(24, 5, strides=(2, 2), activation='elu', padding='same')(x_in)
        x = Conv2D(36, 5, strides=(2, 2), activation='elu', padding='same')(x)
        x = Conv2D(48, 5, strides=(2, 2), activation='elu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        x = Dense(100, activation='relu')(x)
        x = Dense(1, activation='sigmoid', name='steering_output')(x)

        return x

    def throttle_branch(self, x_in):
        x = Conv2D(24, 5, strides=(2, 2), activation='elu', padding='same')(x_in)
        x = Conv2D(36, 5, strides=(2, 2), activation='elu', padding='same')(x)
        x = Conv2D(48, 5, strides=(2, 2), activation='elu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        x = Dense(100, activation='relu')(x)
        x = Dense(1, activation='sigmoid', name='throttle_output')(x)

        return x

    def break_branch(self, x_in):
        x = Conv2D(24, 5, strides=(2, 2), activation='elu', padding='same')(x_in)
        x = Conv2D(36, 5, strides=(2, 2), activation='elu', padding='same')(x)
        x = Conv2D(48, 5, strides=(2, 2), activation='elu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        x = Dense(100, activation='relu')(x)
        x = Dense(1, activation='sigmoid', name='break_output')(x)

        return x

    def build_multivariable_model(self):
        x_in = tf.keras.Input(self.input_shape)
        self.model = tf.keras.models.Model(
            inputs=x_in,
            outputs=[self.steering_branch(x_in), self.throttle_branch(x_in), self.break_branch(x_in)],
            name='multivariable_car_control_model')
