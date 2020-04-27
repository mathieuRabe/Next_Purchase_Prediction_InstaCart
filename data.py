"""=========================================================================="""
"""                       PREPROCESSING FUNCTIONS                            """
"""=========================================================================="""

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences


""" ORDERS ------------------------------------------------------------------"""
#N=100
def orders_process(path, var1, var2, NbClients):
    orders = pd.read_csv(path, delimiter = ',')
    #N = range(1, NbClients+1)
    orders = orders[orders['user_id'].isin(range(NbClients+1))]
    return orders[[var1, var2]]


""" ORDER_PRODUCT_PRIOR -----------------------------------------------------"""

def order_products_process(path, var1, var2):
    order_products = pd.read_csv(path, delimiter = ',')
    return order_products[[var1, var2]]


""" PRODUCTS ----------------------------------------------------------------"""

def products_process(path, var1, var2):
    products = pd.read_csv(path, delimiter = ',')
    return products[[var1, var2]]


""" DEPARTMENTS -------------------------------------------------------------"""

def departments_process(path):
    departments = pd.read_csv(path, delimiter = ',')
    return departments


""" MERGING DATA ------------------------------------------------------------"""

def merging(tab1, tab2, tab3, key1, key2):
    join1 = pd.merge(tab1, tab2, on = key1)
    all_basket = pd.merge(join1, tab3, on = key2)
    return all_basket


""" SEQUENCING CLIENTS ------------------------------------------------------"""

def clients_sequences(all_basket):
    L = []

    for client in all_basket.groupby('user_id'):
        client_basket = client[1]
        sequences_client = []

        for order in client_basket.groupby("order_id"):
            #order_id = order[0]
            order_basket = order[1]
            department_ids = list(order_basket['department_id'])
            sequences_client.append(department_ids)
        L.append(sequences_client)

    return L


"""--------------------------------------------------------------------------"""


def padding_categories(clients_seq, NbClients, NbBasketsMin, NbCategories):
    X = np.zeros((NbClients, NbBasketsMin, NbCategories))
    for i in range(len(clients_seq)):
        xi = pad_sequences(clients_seq[i], maxlen = NbCategories,
                            dtype = 'int32', padding = 'post',
                            truncating = 'pre', value = 0.0)
        panier = xi[:NbBasketsMin]
        X[i] = panier
    return X
