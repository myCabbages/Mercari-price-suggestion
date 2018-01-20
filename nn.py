import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(np.log(y_pred[i]+1) - np.log(y[i]+1))**2 for i, pred in enumerate(y_pred)]
    return np.sqrt(sum(to_sum) * (1.0/len(y)))

def handle_missing(data):
    data.category_name.fillna(value='missing', inplace=True)
    data.brand_name.fillna(value='missing', inplace=True)
    data.item_description.fillna(value='missing', inplace=True)
    return data

def get_keras_data(data):
    X = {
        'name': pad_sequences(data.seq_name, maxlen = MAX_NAME_SEQ),
        'item_desc': pad_sequences(data.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(data.brand_name),
        'category_name': np.array(data.category_name),
        'item_condition': np.array(data.item_condition_id),
        'num_vars': np.array(data[['shipping']])
    }
    return X

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def get_model():
    #params
    dr_r = 0.1

    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    #Embeddings layers
    emb_name = Embedding(MAX_TEXT, 50)(name)
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    #rnn layer
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)

    #main layer
    main_l = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_category_name)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , num_vars
    ])
    main_l = Dropout(dr_r) (Dense(128) (main_l))
    main_l = Dropout(dr_r) (Dense(64) (main_l))

    #output
    output = Dense(1, activation="linear") (main_l)

    #model
    model = Model([name, item_desc, brand_name
                   , category_name, item_condition, num_vars], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])

    return model


if __name__ == '__main__':
    print('Reading Data......')
    train = pd.read_table('../data/train.tsv')
    test = pd.read_table('../data/test.tsv')

    train = handle_missing(train)
    test = handle_missing(test)

    # Process brand_name and category_name to be numerical features
    le = LabelEncoder()

    le.fit(np.hstack([train.category_name, test.category_name]))
    train.category_name = le.transform(train.category_name)
    test.category_name = le.transform(test.category_name)

    le.fit(np.hstack([train.brand_name, test.brand_name]))
    train.brand_name = le.transform(train.brand_name)
    test.brand_name = le.transform(test.brand_name)

    # Process item_description into tokens and generate new columns with lists of
    # words in numerical form
    raw_txt = np.hstack([train.item_description.str.lower(),train.name.str.lower()])
    print('Fitting tokenizer......')
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_txt)

    train['seq_item_description'] = tok_raw.texts_to_sequences(train.item_description.str.lower())
    test['seq_item_description'] = tok_raw.texts_to_sequences(test.item_description.str.lower())

    train['seq_name'] = tok_raw.texts_to_sequences(train.name.str.lower())
    test['seq_name'] = tok_raw.texts_to_sequences(test.name.str.lower())

    MAX_NAME_SEQ = np.max([np.max(train.seq_name.apply(lambda x: len(x))),\
                    np.max(test.seq_name.apply(lambda x: len(x)))])

    MAX_ITEM_DESC_SEQ = np.max([np.max(train.seq_item_description.apply(lambda x: len(x))),\
                    np.max(test.seq_item_description.apply(lambda x: len(x)))])

    MAX_TEXT = np.max([np.max(train.seq_name.max()),\
                        np.max(test.seq_name.max()),\
                        np.max(train.seq_item_description.max()),\
                        np.max(test.seq_item_description.max())]) + 2

    MAX_CATEGORY = np.max([train.category_name.max(),\
                            test.category_name.max()]) + 1
    MAX_BRAND = np.max([train.brand_name.max(), test.brand_name.max()]) + 1
    MAX_CONDITION = np.max([train.item_condition_id.max(),\
                            test.item_condition_id.max()]) + 1

    train['target'] = np.log(train.price + 1)
    target_scaler = MinMaxScaler(feature_range=(-1,1))
    train['target'] = target_scaler.fit_transform(train.target.reshape(-1,1))


    dtrain, dvalid = train_test_split(train, random_state=123, train_size=.99)
    X_train = get_keras_data(dtrain)
    X_valid = get_keras_data(dvalid)
    X_test = get_keras_data(test)

    BATCH_SIZE = 20000
    epochs = 5

    model = get_model()
    model.fit(X_train, dtrain.target, epochs=epochs, batch_size=BATCH_SIZE,\
                validation_data=(X_valid, dvalid.target), verbose=1)

    #EVLUEATE THE MODEL ON DEV TEST: What is it doing?
    val_preds = model.predict(X_valid)
    val_preds = target_scaler.inverse_transform(val_preds)
    val_preds = np.exp(val_preds)-1

    #mean_absolute_error, mean_squared_log_error
    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:,0]
    v_rmsle = rmsle(y_true, y_pred)
    print(" RMSLE error on dev test: "+str(v_rmsle))

    #CREATE PREDICTIONS
    preds = model.predict(X_test, batch_size=BATCH_SIZE)
    preds = target_scaler.inverse_transform(preds)
    preds = np.exp(preds)-1

    submission = test[["test_id"]]
    submission["price"] = preds

    submission.to_csv("./myNNsubmission.csv", index=False)
