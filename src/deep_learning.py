from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def deep_learning_dense(X_train, y_train, X_test, y_test):
    """
    Cette fonction construit, compile et entraîne un modèle de réseau de neurones pour la classification,
    puis évalue les performances du modèle sur un jeu de test. Elle utilise la bibliothèque Keras pour 
    construire le modèle et scikit-learn pour l'évaluation des performances.

    Paramètres:
    - X_train : numpy array
        Les caractéristiques du jeu d'entraînement.
    - y_train : numpy array
        Les étiquettes du jeu d'entraînement.
    - X_test : numpy array
        Les caractéristiques du jeu de test.
    - y_test : numpy array
        Les étiquettes du jeu de test.

    Retourne:
    - model : keras.models.Model
        Le modèle entraîné.
    - history : keras.callbacks.History
        L'historique de l'entraînement du modèle.
    - loss : float
        La perte sur le jeu de test.
    - accuracy : float
        La précision sur le jeu de test.
    - cnf_matrix : numpy array
        La matrice de confusion du jeu de test.
    """
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    num_features = X_train.shape[1]
    num_classes = 4
    
    # Construction du modèle
    inputs = Input(shape = (num_features,), name = "Input")
    
    dense1 = Dense(units = 32, activation = "tanh", name = 'dense_1')
    dense2 = Dense(units = 16, activation = "tanh", name = 'dense_2')
    dense3 = Dense(units = 8, activation = "tanh", name = 'dense_3')
    dense4 = Dense(units = num_classes, activation = "softmax", name = 'dense_4')
    
    x = dense1(inputs)
    x = dense2(x)
    x = dense3(x)
    outputs = dense4(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()

    # Compilation du modèle
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

    # Early stopping pour éviter l'overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entraînement du modèle
    history = model.fit(X_train, y_train,
                         validation_data = (X_test, y_test),
                         epochs = 100, 
                         batch_size = 32,
                         validation_split = 0.2,
                         callbacks=[early_stopping])
   # Évaluation du modèle
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Validation Loss: {loss}')
    print(f'Validation Accuracy: {accuracy}')
    test_pred = model.predict(X_test)
    
    test_pred_class = test_pred.argmax(axis=1)
    y_test_class = y_test
    cnf_matrix = confusion_matrix(y_test_class, test_pred_class)
    
    print(cnf_matrix)
    print(classification_report(test_pred_class,y_test_class))

    return (model, history, loss, accuracy, cnf_matrix)

def deep_learning_improved(X_train, y_train, X_test, y_test):

    """
    Cette fonction construit, compile et entraîne un modèle de réseau de neurones pour la classification,
    puis évalue les performances du modèle sur un jeu de test. Elle utilise la bibliothèque Keras pour 
    construire le modèle et scikit-learn pour l'évaluation des performances.

    Paramètres:
    - X_train : numpy array
        Les caractéristiques du jeu d'entraînement.
    - y_train : numpy array
        Les étiquettes du jeu d'entraînement.
    - X_test : numpy array
        Les caractéristiques du jeu de test.
    - y_test : numpy array
        Les étiquettes du jeu de test.

    Retourne:
    - model : keras.models.Model
        Le modèle entraîné.
    - history : keras.callbacks.History
        L'historique de l'entraînement du modèle.
    - loss : float
        La perte sur le jeu de test.
    - accuracy : float
        La précision sur le jeu de test.
    - cnf_matrix : numpy array
        La matrice de confusion du jeu de test.
    """
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    num_features = X_train.shape[1]
    num_classes = 4

    inputs = Input(shape=(num_features,), name="Input")

    x = Dense(units=32, activation="relu", name='dense_1', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(units=16, activation="relu", name='dense_2', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(units=8, activation="relu", name='dense_3', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.3)(x)
    outputs = Dense(units=num_classes, activation="softmax", name='dense_4')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=200,
                        callbacks=[early_stopping, reduce_lr])
    
    # Évaluation du modèle
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Validation Loss: {loss}')
    print(f'Validation Accuracy: {accuracy}')
    test_pred = model.predict(X_test)
    
    test_pred_class = test_pred.argmax(axis=1)
    y_test_class = y_test
    cnf_matrix = confusion_matrix(y_test_class, test_pred_class)
    
    print(cnf_matrix)
    print(classification_report(test_pred_class,y_test_class))

    return (model, history, loss, accuracy, cnf_matrix)

