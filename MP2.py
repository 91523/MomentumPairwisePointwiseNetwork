# -*- coding: utf-8 -*-

"""
@Time ： 2021/2/7 12:26 PM
@Author ： Anonymity
@File ： MP2.py

"""

from keras.models import Model
from keras.layers import Input, Embedding, BatchNormalization, Dense, Lambda, Multiply, Flatten
from keras import backend as K
from keras import losses as KL
from keras.optimizers import Adam
from keras.callbacks import Callback
import numpy as np


class MomentumRankNetwork:
    def __init__(self, processed_input, user_embedding_size=50, item_embedding_size=50, beta=0.99):
        self.datasets = processed_input.datasets
        self.user_embedding_size = user_embedding_size
        self.item_embedding_size = item_embedding_size
        self.beta = beta
        self.user_input_dim = processed_input.user_vocab_size
        self.item_input_dim = processed_input.item_vocab_size
        self.model = self.set_model()

    def set_model(self):
        user_network = self.create_user_network()
        user_input_left = Input(shape=(1,), name='user_input_left')
        user_input_right = Input(shape=(1,), name='user_input_right')
        user_left = user_network(user_input_left)
        user_right = user_network(user_input_right)

        item_vanilla_network = self.create_item_network()
        item_input_left = Input(shape=(1,), name='item_input_left')
        item_input_right = Input(shape=(1,), name='item_input_right')
        item_vanilla_left = item_vanilla_network([item_input_left])
        item_vanilla_right = item_vanilla_network([item_input_right])

        momentum_network = self.create_item_network(initializer=item_vanilla_network.get_weights())

        item_momentum_left = momentum_network(item_input_left)
        momentum_part_left = Multiply()([user_left, item_momentum_left])
        vanilla_part_left = Multiply()([user_left, item_vanilla_left])

        item_momentum_right = momentum_network(item_input_right)
        momentum_part_right = Multiply()([user_right, item_momentum_right])
        vanilla_part_right = Multiply()([user_right, item_vanilla_right])

        preds_y1 = Dense(1, activation='sigmoid', name='item1')(momentum_part_left)
        preds_y2 = Dense(1, activation='sigmoid', name='item2')(momentum_part_right)

        pair_preds = Lambda(self.difference, output_shape=self.dist_output_shape)([vanilla_part_left, vanilla_part_right])
        pair_preds = Dense(1, activation='sigmoid', name='pair')(pair_preds)

        # label layer
        y_layer = Input((1,), name='y_true')
        y1_layer = Input((1,), name='y1_true')
        y2_layer = Input((1,), name='y2_true')

        # calculate uncertainty
        dist_item_part_left = K.abs(item_vanilla_left - momentum_part_left)
        dist_item_part_right = K.abs(item_vanilla_right - momentum_part_right)

        discrepancy_item_left = K.exp(K.mean(dist_item_part_left, axis=1, keepdims=True))
        discrepancy_item_right = K.exp(K.mean(dist_item_part_right, axis=1, keepdims=True))

        discrepancy_weights = Multiply(trainable=False, name="discrepancy_weights")([discrepancy_item_left, discrepancy_item_right])
        discrepancy_weights = Lambda(lambda x: 1.0 / x)(discrepancy_weights)

        bce = KL.BinaryCrossentropy()
        loss = 0.6 * bce(y_layer, pair_preds) + 0.2 * bce(y1_layer, preds_y1, discrepancy_weights) + 0.2 * bce(y2_layer, preds_y2, discrepancy_weights)

        # pass label layers as inputs
        model = Model(
            inputs=[user_input_left, item_input_left, user_input_right, item_input_right, y_layer, y1_layer, y2_layer],
            outputs=[pair_preds, preds_y1, preds_y2]
        )
        model.add_loss(loss)
        return model

    @staticmethod
    def difference(vectors):
        x, y = vectors
        return x - y

    @staticmethod
    def dist_output_shape(shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

    def create_user_network(self):
        user_input = Input(shape=(1,))
        user_embedding = Embedding(input_dim=self.user_input_dim, output_dim=self.user_embedding_size, input_length=1)(user_input)
        user_embedding = Flatten()(user_embedding)
        user_part = Dense(48, activation='linear', kernel_initializer='uniform', kernel_regularizer='l2')(user_embedding)
        user_part = Dense(32, activation='relu')(user_part)
        user_part = BatchNormalization()(user_part)
        model = Model(inputs=user_input, outputs=user_part, name='user_network')

        return model

    def create_item_network(self, initializer=None):
        item_input = Input(shape=(1,))
        if not initializer:
            item_embedding = Embedding(input_dim=self.item_input_dim, output_dim=self.item_embedding_size, input_length=1)(item_input)
            item_embedding = Flatten(name='vanilla_embedding')(item_embedding)
            item_part = Dense(48, activation='linear', kernel_initializer='uniform', kernel_regularizer='l1')(item_embedding)
            item_part = Dense(32, activation='relu')(item_part)
        else:
            def my_init(initializer):
                def _init(shape, dtype=None):
                    return initializer
                return _init
            item_embedding = Embedding(input_dim=self.item_input_dim, output_dim=self.item_embedding_size, input_length=1, embeddings_initializer=my_init(initializer[0]))(item_input)
            item_embedding = Flatten(name='momentum_embedding')(item_embedding)
            item_part = Dense(48, activation='linear', kernel_initializer=my_init(initializer[1]), trainable=False, name='momentum_dense_1')(item_embedding)
            item_part = Dense(32, activation='relu', kernel_initializer=my_init(initializer[3]), trainable=False, name='momentum_dense_2')(item_part)

        item_part = BatchNormalization()(item_part)
        name = "item_vanilla_network" if not initializer else "item_momentum_network"
        model = Model(inputs=item_input, outputs=item_part, name=name)

        # model.summary()
        return model

    def fit(self, train, epochs=10, batch_size=32):
        self.datasets["train"] = train
        train_input = self.datasets["train"]["left_input"] + self.datasets["train"]["right_input"]
        train_y = [self.datasets["train"]["y"], self.datasets["train"]["y1"], self.datasets["train"]["y2"]]
        train_input = train_input + train_y
        self.model.compile(optimizer=Adam(lr=0.0001), metrics=['acc'])
        # self.model.summary()
        print(f"Training on {len(self.datasets['train']['left_input'][0])} samples")

        weight_callback = WeightsCallback(beta=self.beta)

        if "validation" in self.datasets:
            print("Use specific validation set")
            valid_input = self.datasets["validation"]["left_input"] + self.datasets["validation"]["right_input"]
            valid_y = [self.datasets["validation"]["y"], self.datasets["validation"]["y1"], self.datasets["validation"]["y2"]]
            valid_input = valid_input + valid_y
            self.model.fit(train_input, train_y,
                           verbose=1, epochs=epochs, batch_size=batch_size, validation_data=(valid_input, valid_y),
                           callbacks=weight_callback)
        else:
            self.model.fit(train_input, self.datasets["train"]["y"],
                           verbose=1, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=weight_callback)
        return self.model


class WeightsCallback(Callback):
    def __init__(self, beta=0.99):
        super(WeightsCallback, self).__init__()
        self.beta = beta

    def on_batch_end(self, batch, logs=None):
        raw_weights = zip(self.model.get_layer("item_vanilla_network").get_weights(), self.model.get_layer("item_momentum_network").get_weights())
        new_weights = [w[1] * self.beta + w[0] * (1 - self.beta) for w in raw_weights]
        self.model.get_layer("item_momentum_network").set_weights(new_weights)
