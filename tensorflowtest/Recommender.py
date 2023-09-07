"""
TF 推荐系统 基础版本

本期收获：
    使用模型进行预测时，要求传递的特征张量维度必须与模型要求一致
    modelX(xxx) 调用的是class的call方法，所以call方法必须定义。
"""
import os.path
import pprint
import tempfile
import typing

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from pathlib import Path
from typing import Dict, Text


def recommender():

    input = {
        'user_id': tf.constant(np.array(['138'])),
        'movie_title': tf.constant(np.array(["One Flew Over the Cuckoo's Nest (1975)"])),
        'timestamp': tf.constant(np.array([879024327]))
    }
    input_rank = {
        'user_id': tf.constant(np.array(['138'])),
        'movie_title': tf.constant(np.array(["One Flew Over the Cuckoo's Nest (1975)"]))
    }

    save_retrieval_model_path = "saveModel/recommend_step1_retrieval"
    save_rank_model_path = "saveModel/recommend_step2_rank"
    exist_retrieval = Path(save_retrieval_model_path).exists()
    exist_rank = Path(save_rank_model_path).exists()
    if exist_retrieval and exist_rank:
        index = tf.saved_model.load(save_retrieval_model_path)
        rank = tf.saved_model.load(save_rank_model_path)
        #print(rank(input_rank))
        _, titles = index(input)
        #print(titles)
        title_rate = {}
        for title in titles.numpy()[0]:
            #print(title)
            rank_input_temp = {
                'user_id': tf.constant(np.array(['138'])),
                'movie_title': tf.constant(np.array([title])),
            }
            rank_result_temp: tf.Tensor = rank(rank_input_temp)
            #print(rank_result_temp.numpy())
            title_rate[title] = rank_result_temp.numpy()[0][0]
        print("sort before : ", title_rate)
        sorted_result = sorted(title_rate.items(), key=lambda x: x[1], reverse=True)
        print("sort after : ", sorted_result)
        return

    # 由于load没有指明返回类型，所以这里指定变量类型，以便后续可以.出行为名
    ratings: tf.data.Dataset = tfds.load("movielens/100k-ratings", split='train')
    movies = tfds.load("movielens/100k-movies", split="train")
    # 从dataset中取2份数据，查看数据的结构和内容
    #for x in rating.take(2).as_numpy_iterator():
    #    pprint.pprint(x)
    # {'bucketized_user_age': 45.0,
    # 题材
    #  'movie_genres': array([7], dtype=int64),
    #  'movie_id': b'357',
    # 标题
    #  'movie_title': b"One Flew Over the Cuckoo's Nest (1975)",
    #  'raw_user_age': 46.0,
    # 时间戳
    #  'timestamp': 879024327,
    # 性别
    #  'user_gender': True,
    #  'user_id': b'138',
    # 工作类型
    #  'user_occupation_label': 4,
    # 工作类型文本描述
    #  'user_occupation_text': b'doctor',
    # 评分
    #  'user_rating': 4.0,
    # 邮政编码
    #  'user_zip_code': b'53211'}

    rating = ratings.map(lambda x: {
        'user_id': x['user_id'],
        'timestamp': x['timestamp'],
        'movie_title': x['movie_title'],
        'user_rating': x['user_rating']
    })
    movie = movies.map(lambda x: {
        'movie_title': x['movie_title']
    })

    tf.random.set_seed(42)
    shuffled = rating.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)
    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    #for i in cached_train.take(2):
    #    print(i)

    # ============明确性特征==============
    # 处理方式一： 每个标题作为一个元素，构成词汇表
    #movie_title_lookup = tf.keras.layers.StringLookup()
    # 学习词汇表，学习过程会花些时间
    #movie_title_lookup.adapt(rating.map(lambda x: x['movie_title']))
    #print(f'{movie_title_lookup.get_vocabulary()[:10]}')
    # 词汇表中前十个元素 ['[UNK]', 'Star Wars (1977)', 'Contact (1997)', 'Fargo (1996)', 'Return of the Jedi (1983)', 'Liar Liar (1997)', 'English Patient, The (1996)', 'Scream (1996)', 'Toy Story (1995)', 'Air Force One (1997)']
    #movie_title_lookup_result: tf.Tensor = movie_title_lookup(['Star Wars (1977)', 'Fargo (1996)'])
    #print(movie_title_lookup_result, movie_title_lookup_result.numpy())
    # 两个标题经过词汇表映射后，转换为数字表示形式，这里是词汇表中的下标值。tf.Tensor([1 3], shape=(2,), dtype=int64)

    # 明确性特征，处理方式二： 对传入的元素进行hash
    num_hashing_bins = 200_000
    movie_title_hashing = tf.keras.layers.Hashing(num_bins=num_hashing_bins)
    #movie_title_hashing_result: tf.Tensor = movie_title_hashing(['Star Wars (1977)', 'Fargo (1996)'])
    #print(movie_title_hashing_result)
    # 标题hash结果 tf.Tensor([101016  91073], shape=(2,), dtype=int64)

    # 嵌入向量，指定输入维度，输出维度
    movie_title_embedding = tf.keras.layers.Embedding(
        input_dim=num_hashing_bins,
        #input_dim=movie_title_lookup.vocabulary_size(),
        output_dim=32
    )

    # 两个步骤合并在一起
    movie_title_model = tf.keras.Sequential([movie_title_hashing, movie_title_embedding])
    #movie_title_model = tf.keras.Sequential([movie_title_lookup, movie_title_embedding])
    # 测试 得到一个标题的嵌入向量
    #movie_title_model_result: tf.Tensor = movie_title_model(['Contact (1997)'])
    #print(movie_title_model_result)

    # 对用户ID使用词汇表，然后嵌入，得到对用户ID的处理模型
    user_id_lookup = tf.keras.layers.StringLookup()
    user_id_lookup.adapt(rating.map(lambda x: x['user_id']))
    user_id_embedding = tf.keras.layers.Embedding(user_id_lookup.vocabulary_size(), 32)
    user_id_model = tf.keras.Sequential([user_id_lookup, user_id_embedding])
    #for x in cached_train.take(1).as_numpy_iterator():
    #    print(user_id_model(x['user_id']))

    # ==============连续性特征===============
    # 方式一：标准化
    timestamp_normalization = tf.keras.layers.Normalization(
        axis=None
    )
    # 学习标准化
    timestamp_normalization.adapt(rating.map(lambda x: x['timestamp']).batch(1024))
    # 测试 输出缩放之后的结果
    #for x in rating.take(3).as_numpy_iterator():
    #    print(f"{x['timestamp']} 标准化后 {timestamp_normalization([x['timestamp']])}")
    # +嵌入
    #timestamp_normalization_model = tf.keras.Sequential([
    #    timestamp_normalization,
    #    tf.keras.layers.Embedding(1, 32)
    #])
    #for timestamp in rating.take(1).map(lambda x: x['timestamp']).batch(1).as_numpy_iterator():
    #    print(f"{timestamp} 标准化+嵌入后 {timestamp_normalization_model(timestamp)}")


    # 方式二：离散法  找到最大值 最小值，然后根据要求的数量去分桶
    max_timestamp = rating.map(lambda x: x['timestamp']).reduce(np.int64(1e9), tf.maximum).numpy().max()
    min_timestamp = rating.map(lambda x: x['timestamp']).reduce(np.int64(1e9), tf.minimum).numpy().min()
    timestamp_buckets = np.linspace(min_timestamp, max_timestamp, num=1000)
    print(f'取前10个分段 {timestamp_buckets[:10]}')

    # 离散法+嵌入
    timestamp_model = tf.keras.Sequential([
        tf.keras.layers.Discretization(timestamp_buckets.tolist()),
        tf.keras.layers.Embedding(len(timestamp_buckets)+1, 32)
    ])
    for timestamp in rating.take(1).map(lambda x: x['timestamp']).batch(1).as_numpy_iterator():
        print(f'{timestamp} 嵌入向量为 {timestamp_model(timestamp)}')

    # ==========原始文本特征===============
    # ==========原始文本特征===============
    # 分词，字典表，嵌入
    max_tokens = 10_000
    title_text = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
    title_text.adapt(movie.map(lambda x: x['movie_title']))
    # 看一个标题 分词后 有字典表位置构成的向量
    for title in rating.batch(1).map(lambda x: x['movie_title']).take(1):
        print(f'{title} 分词后，各个分词在字典表中的位置为 {title_text(title)}')
    # 查看字典表
    print(f'取几个分词看看 {title_text.get_vocabulary()[40:50]}')

    title_text_model = tf.keras.Sequential([
        title_text,
        # 标题会有多个分词，每个分词产生一个嵌入向量，因此每个标题对应多个嵌入向量
        tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
        # 标题的多个嵌入向量 合并为 一个向量
        tf.keras.layers.GlobalAveragePooling1D()
    ])
    print(f"标题 'Contact (1997)' 进行分词、嵌入、合并向量后， {title_text_model(['Contact (1997)'])}")

    # 用户模型 query模型
    class UserModel(tf.keras.Model):
        def __init__(self):
            super().__init__()

            self.user_embedding = user_id_model
            self.timestamp_embedding = timestamp_model
            self.timestamp_normalization = timestamp_normalization

        def call(self, inputs):
            #print("UserModel call : ", inputs, inputs['user_id'])
            user_embedding_result = self.user_embedding(inputs['user_id'])
            #print("UserModel user_embedding_result : ", user_embedding_result)
            # 得到几个元素各自的嵌入向量，然后连接在一起，[shape 32] + [shape 32] + [shape 1] = [shape 65]
            return tf.concat([
                user_embedding_result,
                self.timestamp_embedding(inputs['timestamp']),
                tf.reshape(self.timestamp_normalization(inputs['timestamp']), (-1, 1))
            ], axis=1)

    #user_model = UserModel()
    #for row in rating.batch(1).take(1):
    #    print(f'query模型 输出 {user_model(row)}')

    class MovieModel(tf.keras.Model):
        def __init__(self):
            super().__init__()

            self.title_model = movie_title_model
            self.title_text_model = title_text_model

        def call(self, inputs):
            return tf.concat([
                self.title_model(inputs['movie_title']),
                self.title_text_model(inputs['movie_title'])
            ], axis=1)

    movie_model = MovieModel()
    #for row in rating.batch(1).take(1):
    #    print(f'候选模型 输出 {movie_model(row)}')

    # 通过训练下述模型，从而训练出具有关系的查询模型和候选者模型
    class MovielesModel(tfrs.models.Model):
        def __init__(self):
            super(MovielesModel, self).__init__()
            self.query_model = tf.keras.Sequential([
                UserModel(),
                tf.keras.layers.Dense(32)
            ])
            self.candidate_model = tf.keras.Sequential([
                MovieModel(),
                tf.keras.layers.Dense(32)
            ])
            # 因子检索任务
            self.task = tfrs.tasks.Retrieval(
                # 在候选资料库中评估出前K个候选。 该衡量模型具有从所有可能候选者中挑选出真正候选者的能力。
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=movie.batch(128).map(self.candidate_model)
                )
            )

        def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
            query_embeddings = self.query_model({
                'user_id': inputs['user_id'],
                'timestamp': inputs['timestamp']
            })
            movie_embedding = self.candidate_model({'movie_title': inputs['movie_title']})
            # FactorizedTopK#call 两个主要参数 查询嵌入向量和特征嵌入向量
            # 表示为用户做的选择，即该查询嵌入向量 查询到 该特征嵌入向量
            # 目标：最大化 查询和候选对 的相关性，最小化 查询与其他候选 的相关性
            return self.task(query_embeddings, movie_embedding)

    model = MovielesModel()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    # 训练模型
    model.fit(cached_train, epochs=10)

    # 检测模型
    train_accuracy = model.evaluate(
        cached_train, return_dict=True
    )["factorized_top_k/top_100_categorical_accuracy"]
    # 检测模型
    test_accuracy = model.evaluate(
        cached_test, return_dict=True
    )["factorized_top_k/top_100_categorical_accuracy"]
    # 输出 top-100 准确性
    print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
    print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")

    # 创建一个检索模型，需使用训练后的查询模型和候选者模型
    index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
    # 生成 候选者标识 和 候选者嵌入 关系
    index.index_from_dataset(
        tf.data.Dataset.zip((movie.batch(100).map(lambda x: x['movie_title']), movie.batch(100).map(model.candidate_model)))
    )

    #########################Rank 阶段###############################
    # 排名模型
    class RatingModel(tf.keras.Model):
        '''
        训练 特征（用户id + 电影名称） 与 标签(用户评分) 的关系模型
        '''
        def __init__(self):
            super().__init__()

            self.user_embeddings = user_id_model
            self.movie_embeddings = movie_title_model
            self.rating = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

        def call(self, inputs):
            #print("RatingModel inputs : ", inputs)
            user_embedding = self.user_embeddings(inputs['user_id'])
            #print("RatingModel user_embedding : ", user_embedding)
            movie_embedding = self.movie_embeddings(inputs['movie_title'])
            rating_result = self.rating(tf.concat([user_embedding, movie_embedding], axis=1))
            #print("RatingModel rating_result : ", rating_result)

            return rating_result

    #for x in rating.take(10).batch(1):
    #    print(RatingModel()(x))

    class RankingModel(tfrs.models.Model):
        def __init__(self):
            super().__init__()
            self.rating_model: tf.keras.Model = RatingModel()
            # 我的天 找了一天问题，原来是配置loss和metrices时没有实例化，
            self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

        # 通过model(xxx)调用该方法，所以必须定义。
        def call(self, inputs, training=None, mask=None):
            return self.rating_model(inputs)

        def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
            #print("RankingModel inputs : ", inputs)
            labels = inputs.pop('user_rating')
            pre = self.rating_model({
                'user_id': inputs['user_id'],
                'movie_title': inputs['movie_title']
            })
            #print("RankingModel type of labels : ", type(labels))
            #print("RankingModel type of pre : ", type(pre))
            #print("RankingModel numpy of pre : ", pre.numpy())
            #print("RankingModel numpy of labels : ", labels.numpy())

            return self.task(labels=labels, predictions=pre)

    ranking_model = RankingModel()
    ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    ranking_model.fit(cached_train, epochs=10)
    train_accuracy = ranking_model.evaluate(cached_train, return_dict=True)
    test_accuracy = ranking_model.evaluate(cached_test, return_dict=True)
    print("ranking_model evaluate result : ", train_accuracy, test_accuracy)

    # 预测  有个奇怪的地方：保存模型之前，必须先调用一次模型，否则保存模型并加载模型后，调用模型失败，提示模型is not callable。
    _, titles = index(input)
    ranking_model_result = ranking_model(input_rank)
    print("ranking_model_result : ", ranking_model_result)
    print(f"Recommendations for user {input['user_id']} : {titles[0, :3]}")

    tf.saved_model.save(index, save_retrieval_model_path)
    tf.saved_model.save(ranking_model, save_rank_model_path)
    ranking_model_loaded = tf.saved_model.load(save_rank_model_path)
    index_model_loaded = tf.saved_model.load(save_retrieval_model_path)
    print("index_model_loaded result :", index_model_loaded(input))
    print("ranking_model_loaded :", ranking_model_loaded(input_rank))


if __name__ == '__main__':
    recommender()
