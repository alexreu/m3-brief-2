import tensorflow as tf

Sequential = tf.keras.Sequential
Dense = tf.keras.layers.Dense
Input = tf.keras.layers.Input
keras = tf.keras


def create_nn_model(input_dim: int) -> keras.Model:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_model(
    model: keras.Model,
    X,
    y,
    X_val=None,
    y_val=None,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 1,
    callbacks=None,
):
    history = model.fit(
        X,
        y,
        validation_data=(
            X_val, y_val) if X_val is not None and y_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )
    return model, history


def model_predict(model: keras.Model, X):
    return model.predict(X).flatten()
