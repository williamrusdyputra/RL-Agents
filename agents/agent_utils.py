import tensorflow as tf


def build_network(mode, action_space, state_shape, path):
    if mode == 'actor':
        model = build_actor(action_space, state_shape)
        try:
            model.load_weights(path)
            print('Model loaded')
        except ValueError:
            print('Model initialized')
        return model
    elif mode == 'critic':
        model = build_critic(state_shape)
        try:
            model.load_weights(path)
            print('Model loaded')
        except ValueError:
            print('Model initialized')
        return model


def build_actor(action_space, state_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation='tanh', input_shape=state_shape),
        tf.keras.layers.Dense(units=64, activation='tanh'),
        tf.keras.layers.Dense(units=action_space, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4))

    return model


def build_critic(state_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation='tanh', input_shape=state_shape),
        tf.keras.layers.Dense(units=512, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

    return model
