import numpy as np
import tensorflow as tf


def build_actor(action_space, state_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='tanh', input_shape=state_shape),
        tf.keras.layers.Dense(units=256, activation='tanh'),
        tf.keras.layers.Dense(units=action_space, activation='linear')
    ])

    # custom loss function with 1e-6 so log(0) = -inf will not happen
    def loss(y_true, y_pred):
        action_true = y_true[:, :action_space]
        advantage = y_true[:, action_space:]
        return -np.log(y_pred.prob(action_true) + 1e-6) * advantage

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss=loss)

    return model


def build_critic(state_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation='tanh', input_shape=state_shape),
        tf.keras.layers.Dense(units=512, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

    return model
