# -*- coding: utf-8 -*-

"""
@Time ： 2021/2/7 2:14 PM
@Author ： Anonymity
@File ： train_and_evaluate.py

"""

import os
import numpy as np
from MP2 import MomentumRankNetwork
from utils import DataProcessor, InputProducer, logger, Dataset
from utils import build_score_dataframe, ndcg_score_at_k, hit_rate_at_k

if __name__ == '__main__':
    # generate datasets and produce model input by chunk
    if not os.path.exists('data/ml-100k'):
        os.system(f'cd data && unzip ml-100k.zip')

    file_set = {'train': 'data/u1.base.data', 'validation': 'data/u1.valid.data', 'test': 'data/u1.test.data'}
    processor = DataProcessor(base_path='data/ml-100k',
                              key='u1',
                              n=20,  # choose n samples randomly from a user to make pairs
                              seed=123,  # to reproduce the result
                              extra_negative_sampling=True,  # add unseen items as negative samples
                              sampling_rate=0.3,  # sampling rate for implicit negative samples
                              threshold=5)
    if not os.path.exists(file_set['train']) or not os.path.exists(file_set['test']):
        processor.split_data()  # generate encoded data
        processor.process(processor.base_file)  # generate train
        processor.process(processor.test_file)  # generate valid
        processor.process_test()  # generate test

    ip = InputProducer(user_vocab_size=944,
                       item_vocab_size=1683,
                       chunksize=1000000)
    ip.produce_chunk_input(file_set, sep='\t')

    # train model
    model = MomentumRankNetwork(ip,
                                user_embedding_size=64,
                                item_embedding_size=64,
                                beta=0.999)

    testset = model.datasets['test']
    test_y = [testset[Dataset.y], testset[Dataset.y1], testset[Dataset.y2]]
    test_input = testset['left_input'] + testset['right_input'] + test_y

    test_y = [testset[Dataset.y], testset[Dataset.y1], testset[Dataset.y2]]
    test_input = test_input + test_y

    trained_model = None
    chunked_train_dfs = ip.datasets['train']
    counter = 1
    for train_df in chunked_train_dfs:
        logger.info(f'chunk - [{counter}]')
        train_df = train_df.sample(frac=1.0, random_state=42)
        train_input = ip.produce_input_from_dataframe('train', train_df)
        trained_model = model.fit(train=train_input, batch_size=64, epochs=5)
        counter += 1
    if not os.path.exists('cache'):
        os.mkdir('cache')
    trained_model.save_weights(f'cache/mp2.h5')

    preds_proba = trained_model.predict(test_input, batch_size=64)  # shape (n_samples, 3)
    flatten_preds = list(np.concatenate(preds_proba[2]))

    score_df = build_score_dataframe(group_key=testset[Dataset.uid],
                                     y_true=testset[Dataset.y1],
                                     y_preds=flatten_preds,
                                     remove_dupes=True,
                                     dup_indicator=testset[Dataset.iid1])
    print(score_df)

    k_list = [5, 20]
    for k in k_list:
        ndcg_score = ndcg_score_at_k(score_df=score_df, k=k)
        hit_rate = hit_rate_at_k(score_df=score_df, k=k)
        logger.info(f'ndcg score at {k} (mean): {ndcg_score}')
        logger.info(f'hit rate at {k} (mean): {hit_rate}')
