import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score



def dl_model(x_train, y_train, x_test, y_test):
    # model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')  
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Train the model
    model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,  # This is just an example, for the final report the model is trained for multiple epochs                          
    batch_size=256,
    verbose=1
    )

    loss, accuracy, auc = model.evaluate(x_test, y_test, verbose=0)

    y_pred_proba = model.predict(x_test).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred)
    return {'F1' : f1, 'AUC' : auc}

