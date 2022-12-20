"""
使用keras预处理层对结构化数据进行分类
收获：
数据处理方式，如何预处理结构化数据，包括数值数据进行规范化、类型数据转为独热向量或嵌入或hash
使用keras预处理层 将 结构化数据列 映射到 用户训练模型的特征
tf.feature_column已经废弃
使用 预处理层 与 keras模型函数API 构建模型
keras预处理层更直观，可以轻松包含到模型中以简化部署。
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# train_test_split 用于将数组或矩阵分割两段，可指定test集比例或train集比例，分割前是否洗牌等
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def init_dataframe():
    """
    从数据源获取数据，并生成pandas dataframe
    :return:
    """
    # 数据url
    dataset_url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"
    # 从url下载文件 fname保存为文件名 origin文件url extract是否提取压缩文件内文件 cache_dir下载位置如'D:\\temp'
    path = tf.keras.utils.get_file(fname='petfinder-mini.zip', origin=dataset_url, extract=True, cache_dir='.')
    print(path)
    # 读取的文件位置
    csv_file = "datasets/petfinder-mini/petfinder-mini.csv"
    dataframe = pd.read_csv(csv_file)
    return dataframe

def clean_dataframe(dataframe):
    """
    清理特征列 如清除无用列 组织标签列数据等
    :return:
    """
    # 创建标签列 0未被领养 1被领养
    # pandas dataframe 添加列和读取列的方式
    # numpy where 数组操作
    dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4, 0, 1)
    # 删除无用的列 pandas drop 可删除行和列
    dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])
    return dataframe


def split_2_train_val_test(dataframe):
    """
    分割出train val test数据集
    :param dataframe:
    :return: train val test
    """
    # 分割出train val test数据集
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print('\ntrain len ', len(train), '\nval len ', len(val), '\ntest len ', len(test))

    return train, val, test


def preprocessing_and_model(train_ds):
    """
    数据预处理，构建模型
    :return:
    """
    all_inputs, encoded_features = preprocessing_feature(train_ds)

    # 多个特征张量 拼接为 一个张量
    all_features = tf.keras.layers.concatenate(encoded_features)
    print('all_inputs: ', list(all_inputs),
          '\n encoded_features : ', list(encoded_features),
          '\n all_features', all_features,
          '\n all_features shape', all_features.shape,
          '\n all_features dtype', all_features.dtype)

    # 构建模型
    # Keras 函数式 API 是一种比 tf.keras.Sequential API 更灵活的创建模型的方式
    # 函数式 API 将模型视为层的 有向无环图DAG
    # 在函数式 API 中，输入规范（形状和 dtype）是预先创建的（使用 Input）。
    # 每次调用层时，该层都会检查传递给它的规范是否符合其假设，如不符合，它将引发有用的错误消息。例如类型检测
    x = tf.keras.layers.Dense(32, activation='relu')(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(all_inputs, output)
    model.summary()
    # 编译模型
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # 生成层计算图 展示从数据预处理到模型输出层的层次关系。这个图不会自动打开，而是自动保存到项目路径。
    # tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    return model


def experimental_preprocessing():
    """
    使用宠物数据集，预测宠物是否会被领养。每行数据表示一个宠物，每列表示特征。
    \n特征信息表，列名、含义、数据类型（字符串、整数）、特征类型（分类、数值、文本）
    \n依据特征类型，判断进行哪些预处理，如规范化、类别处理
    :return:
    """

    dataframe = init_dataframe()
    print(dataframe.head())

    dataframe = clean_dataframe(dataframe)

    train, val, test = split_2_train_val_test(dataframe)

    # test_encode(train)

    # 构建元素为分批数据的dataset
    batch_size = 256
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    model = preprocessing_and_model(train_ds)

    # 训练模型
    model.fit(train_ds, epochs=10, validation_data=val_ds)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)

    # 保存函数式模型的标准方式是调用 model.save() 将整个模型保存为单个文件
    # 保存的文件包括：
    # 模型架构
    # 模型权重值（在训练过程中得知）
    # 模型训练配置（如果有的话，如传递给 compile）
    # 优化器及其状态（如果有的话，用来从上次中断的地方重新开始训练）
    model.save('my_pet_classifier')


def preprocessing_feature(train_ds):
    """
    对特征进行预处理
    :param train_ds: 训练集 用于预处理层学习 如mean 方差 字符集
    :return:
    """
    all_inputs = []
    encoded_features = []

    # 数字规范化
    for header in ['PhotoAmt', 'Fee']:
        # 实例化一个张量 shape表示元素为x维数组  name要求在model中名称唯一
        # keras 函数 api 需要Input指定输入
        numeric_column = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_column)
        all_inputs.append(numeric_column)
        encoded_features.append(encoded_numeric_col)

    # 对正数特征进行分类处理
    for header in ['Age']:
        age_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
        category_encoding_layer = get_category_encoding_layer(header, train_ds, dtype='int64', max_tokens=5)
        encoded_col = category_encoding_layer(age_col)
        all_inputs.append(age_col)
        encoded_features.append(encoded_col)

    # 对字符特征进行分类处理
    categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                        'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
    for header in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                     max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    return all_inputs, encoded_features


def predict_one():
    model = tf.keras.models.load_model('my_pet_classifier')
    sample = {
        'Type': 'Cat',
        'Age': 3,
        'Breed1': 'Tabby',
        'Gender': 'Male',
        'Color1': 'Black',
        'Color2': 'White',
        'MaturitySize': 'Small',
        'FurLength': 'Short',
        'Vaccinated': 'No',
        'Sterilized': 'No',
        'Health': 'Healthy',
        'Fee': 100,
        'PhotoAmt': 2,
    }

    # {key: val张量}
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = model.predict(input_dict)
    prob = tf.nn.sigmoid(predictions[0])

    print(
        "This particular pet had a %.1f percent probability "
        "of getting adopted." % (100 * prob), predictions[0]
    )


def test_encode(train):

    # 构建元素为分批数据的dataset
    batch_size = 5
    train_ds = df_to_dataset(train, batch_size=batch_size)
    # 从ds中取count个元素，构成新的ds
    print('\n', list(train_ds.take(1).as_numpy_iterator()))
    print_ds(train_ds)

    [(train_features, train_labels)] = train_ds.skip(2).take(1)
    # 预处理层的使用
    normalization_layer = get_normalization_layer('PhotoAmt', train_ds)
    print('PhotoAmt origin ', train_features['PhotoAmt'], ' after normal ', normalization_layer(train_features['PhotoAmt']))

    category_layer = get_category_encoding_layer('Type', train_ds, 'string')
    print('\nType origin ', train_features['Type'], ' after ', category_layer(train_features['Type']))

    category_layer = get_category_encoding_layer('Age', train_ds, 'int64', 5)
    print('\nAge origin ', train_features['Age'], ' after ', category_layer(train_features['Age']))
    category_layer = get_category_encoding_layer('Age', train_ds, 'int64')
    print('\nAge origin ', train_features['Age'], ' after ', category_layer(train_features['Age']))


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    """
    分类列 预处理层
    \n将数据映射到词汇表的正数索引，并且对特征进行多热编码
    :param name: 特征名称
    :param dataset:
    :param dtype: 特征数据类型
    :param max_tokens:
    :return:
    """
    if dtype == 'string':
        # 字符串特征 转为 整数索引
        # max_tokens指定词汇表最大量。仅在需要自动学习词汇表时，可以进行设置；如果不设置，则没有上限，要看自动学习到词汇表的大小或指定的词汇表大小。默认None。
        # 如果max_tokens数量小于学习到的词汇表，则支取前max_tokens个词汇作为词汇表，且用这个词汇表进行映射。
        # num_oov_indices指定在词汇表之外的词语的索引数量。如果设置大于1，则通过hash确定oov值。默认1。
        # mask_token指定需要掩盖的词汇
        # oov_token指定oov索引返回的词汇，仅在invert为True时使用。
        # vocabulary指定词汇表 可以使数组 也可是文件路径，文件内容要求每个词汇占用一行。如果指定词汇表则不需要adapt
        # idf_weights 仅用于output_mode为tf_idf时，指定词汇表中每个词汇的idf
        # invert 仅output_mode为int，索引映射为词汇
        # output_mode 默认int，可选int输出词汇的索引，one_hot输出词汇的独热，multi_hot，count，tf_idf输出词汇的tf_idf值
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        # 整数特征 映射到 邻近范围
        # max_tokens
        # 如果max_tokens数量小于学习到的词汇表，则支取前max_tokens个词汇作为词汇表，且用这个词汇表进行映射。
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)

    feature_ds = dataset.map(lambda x, y: x[name])
    # 学习 词汇表
    index.adapt(feature_ds)
    print('index vocabulary : ', list(index.get_vocabulary()))
    # 对整数进行编码
    # num_tokens支持的最大数量
    # output_mode 默认multi_hot。支持multi_hot，one_hot，count
    print(index.vocabulary_size())
    encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())

    # 特征值找到索引，然后编码
    return lambda feature: encoder(index(feature))


def get_normalization_layer(name, dataset):
    """
    数值列 预处理层
    \n数值型列的规范化层，Normalization会自动计算平均值和方差，以此对数据进行规范化，结果为每个特征的平均值为 0，且其标准差为 1。
    :param name:
    :param dataset:
    :return:
    """
    # 创建一个Normalization层
    normalizer = preprocessing.Normalization(axis=None)
    # 从dataset中获取某个特征的所有数据并构建ds，若源ds是分批的，则新ds也是分批的
    feature_ds = dataset.map(lambda x, y: x[name])
    # print(list(feature_ds.as_numpy_iterator()))
    # 学习 数据的统计
    adapt = normalizer.adapt(feature_ds)
    return normalizer


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    """
    基于pandas dataframe 创建 tf.data dataset, 每个元素是一个批
    \n结果：元素为({key:val张量,key:val张量},label张量)的ds
    :param dataframe: pandas dataframe
    :param shuffle:  是否洗牌 默认True
    :param batch_size: 批大小
    :return:
    """
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    # 构建一个dataset，其元素为对张量进行的分片
    # dict构建字典 {key:val,key:val}
    # 元素为({key: val, key: val}, label)的ds
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    print_ds(ds)
    if shuffle:
        # 返回一个洗牌后的dataset，buffer_size理解为返回的ds长度， reshuffle_each_iteration每次迭代是否自动重新洗牌
        ds.shuffle(buffer_size=len(dataframe))
    # batch用ds中连续元素构建批，由批构成dataset，batch_size每个批的大小
    # prefetch构建可预取元素的dataset，buffer_size每次预取的元素数量。
    # 要求大部分dataset都应已prefetch结尾，可减少延迟，增加吞吐量。
    ds = ds.batch(batch_size).prefetch(batch_size)
    return ds


def print_ds(ds):
    [(train_features, train_labels)] = ds.take(1)
    print('\n features keys', list(train_features.keys()))
    print('\n batch vals of age', train_features['Age'])
    print('\n batch vals of label', train_labels)


if __name__ == '__main__':
    experimental_preprocessing()
    print('===will predict one===')
    predict_one()
