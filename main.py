import pandas as pd
import numpy as np
from numpy import array
import keras

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score



from data import orders_process, order_products_process, products_process, departments_process, merging, clients_sequences, padding_categories



"""=========================================================================="""
"""                             I - PREPROCESSING                            """
"""=========================================================================="""

NbClients = 100

if __name__ == '__main__':

    # ORDERS ----------------------------------------------------------------- #

    orders = orders_process("D:\GitHub\Kaggle\instacart-market-basket-analysis\Data\orders.csv", 'order_id', 'user_id', NbClients)

    # ORDERSPRODUCTSPRIOR ---------------------------------------------------- #
    order_products = order_products_process("D:\GitHub\Kaggle\instacart-market-basket-analysis\Data\order_products__prior.csv", 'order_id', 'product_id')

    # PRODUCTS --------------------------------------------------------------- #
    products = products_process("D:\GitHub\Kaggle\instacart-market-basket-analysis\Data\products.csv", 'product_id', 'department_id')

    # DEPARTMENTS ------------------------------------------------------------ #
    departments = departments_process("D:\GitHub\Kaggle\instacart-market-basket-analysis\Data\departments.csv")

    # MERGING ---------------------------------------------------------------- #
    all_basket = merging(orders, order_products, products, 'order_id', 'product_id')


    clients_seq = clients_sequences(all_basket)


    sortie = padding_categories(clients_seq, NbClients, 3, 21)


    X = sortie[:,:-1,:]
    Y = sortie[:,-1,:]


print(X)
print('\n')
print(Y)


"""=========================================================================="""
"""                             II - MODELING                                """
"""=========================================================================="""


X_train, X_test, y_train, y_test = train_test_split(X, Y)


callback_1 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5)
callback_2 = keras.callbacks.ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)


model = Sequential()
model.add(LSTM(21,return_sequences=True, input_shape=(2,21)))
model.add(LSTM(64,activation='relu'))
model.add(Dense(21,activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X, Y, epochs=50, batch_size=1, verbose=1, callbacks=[callback_1, callback_2])



"""=========================================================================="""
"""                             III - MODEL EVALUATION                       """
"""=========================================================================="""

accuracy = model.evaluate(X_test,y_test)
print(' Loss: {:0.3f} et Accuracy: {:0.3f}'.format(accuracy[0],accuracy[1]))
