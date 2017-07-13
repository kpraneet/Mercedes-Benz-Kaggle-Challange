import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as backend
from sklearn import preprocessing
import os


model_path = 'keras_model.h5'


def process_data(train_data, len_val):
    # print(train_data[0:, len_val:len_val + 1])
    tmp = []
    for x in train_data[0:, len_val:len_val + 1]:
        tmp.append(x[0])
    chk = []
    for x in set(tmp):
        chk.append(x)
    chk = sorted(chk)
    # print(chk)
    count = 0
    for x in chk:
        for y in train_data[0:, len_val:len_val + 1]:
            if x == y[0]:
                y[0] = count
        count += 1
    # print(train_data[0:, len_val:len_val + 1])


def r2(y_true, y_pred):
    res = backend.sum(backend.square(y_true - y_pred))
    tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return 1 - res/(tot + backend.epsilon())


def new_model():
    model = Sequential()
    model.add(Dense(376, input_dim=376, kernel_initializer='normal', activation='tanh'))
    model.add(Activation('linear'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(376, kernel_initializer='normal', activation='tanh'))
    model.add(Activation('linear'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(188, kernel_initializer='normal', activation='tanh'))
    model.add(Activation('linear'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(94, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2])
    print(model.summary())
    return model


def main():
    df = pd.read_csv('train.csv', header=None, dtype=object)
    train_data = df.as_matrix()[1:, 2:]
    train_label = df.as_matrix()[1:, 1:2]
    print(train_data.shape)
    print(train_label.shape)
    df = pd.read_csv('test.csv', header=None, dtype=object)
    eval_numbers = df.as_matrix()[1:, 0:1]
    eval_data = df.as_matrix()[1:, 1:]
    print(eval_data.shape)
    # Process data from csv
    # print(train_data)
    for x in range(0, train_data.shape[1]):
        process_data(train_data, x)
    # print(train_data)
    # print(train_label)
    for x in range(train_label.shape[0]):
        train_label[x][0] = float(train_label[x][0])
    # print(train_label)
    # print(eval_data)
    for x in range(0, eval_data.shape[1]):
        process_data(eval_data, x)
    # print(eval_data)
    check_data = []
    check_label = []
    for x in range(train_data.shape[0]):
        if x % 8 == 0:
            check_data.append(train_data[x])
            check_label.append(train_label[x])
    check_data = np.array(check_data)
    check_label = np.array(check_label)
    # Normalize data
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(np.concatenate((train_data, eval_data)))
    train_data = scaler.transform(train_data)
    eval_data = scaler.transform(eval_data)
    # New approach
    estimators = KerasRegressor(build_fn=new_model, epochs=300, batch_size=20, verbose=1,
                                validation_data=(check_data, check_label))
    callbacks = [EarlyStopping(monitor='val_loss', patience=30, verbose=1),
                 ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)]
    estimators.fit(train_data, train_label, epochs=100, verbose=2, callbacks=callbacks, shuffle=True)
    if os.path.isfile(model_path):
        estimators = load_model(model_path, custom_objects={'r2': r2})
    predict_list = estimators.predict(eval_data, batch_size=5)
    # print(predict_list)
    # Store values into a csv file
    final_list = list()
    final_list.append(['ID', 'y'])
    for x in range(eval_numbers.shape[0]):
        temp = list()
        temp.append(eval_numbers[x][0])
        temp.append(str(predict_list[x][0]))
        final_list.append(temp)
    df = pd.DataFrame(final_list)
    df.to_csv('output.csv', index=False, header=False)


if __name__ == '__main__':
    main()
