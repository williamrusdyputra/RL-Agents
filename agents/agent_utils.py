import numpy as np
import tensorflow as tf


def build_actor():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu', input_shape=self.state_shape),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=self.action_space)
    ])

    # custom loss function with 1e-6 so log(0) = -inf will not happen
    def loss(y_true, y_pred):
        action_true = y_true[:, :self.action_space]
        advantage = y_true[:, self.action_space:]
        return -np.log(y_pred.prob(action_true) + 1e-6) * advantage

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss=loss)

    return model


def build_critic():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu', input_shape=self.state_shape),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='mse')

    return model
