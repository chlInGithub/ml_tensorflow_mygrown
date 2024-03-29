# 收获:
# 使用神经网络实现回归模型
# 训练模型不是必须海量数据
# 迭代次数越多 不一定 模型越好，这里涉及了早停法
# 回归的常见损失函数 均方差 MSE
# 回归的常用衡量标准 平均绝对误差 MAE
# 数字特征 进行 规范化
# 数据的图形化

# 步骤：
# 第0步：准备原始数据
# 第一步：观察数据
# 第二步：数据清洗
# 第三步：拆分数据集
# 第四步：散点图矩阵 观察关系
# 第五步：组织标签
# 第六步：数据规范化
# 第七步：构建模型
# 第八步：训练模型
# 第九步：观察训练结果，以便调整学习速率、正则化率等
# 第十步：训练集和验证集 指标结果 满足一定条件后，模型应用于测试集。
# 第十一步：根据测试集预测结果，决定重复八九十步。

# for question: Initializing libiomp5md.dll, but found libiomp5 already initialized.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pathlib
# 画图
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf

# tf的一种神经网络工具包
from tensorflow import keras as keras
# 神经网络层
from keras import layers as layers

import keras_tuner as kt

def regression():
    """
    使用keras实现回归模型
    :return:
    """
    print(f'tf version: {tf.__version__}')

    print(tf.concat([[[1], [2]], [[3], [4]]], 1))


    # 第0步：准备原始数据
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    # 列名
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    # 使用pandas，从文件读取数据
    dataset = pd.read_csv(
        # 数据路径
        dataset_path,
        # 列名
        names=column_names,
        # 哪些数据看作NA
        na_values="?",
        # 行分割标记
        comment="\t",
        # 列分割标记
        sep=" ",
        #
        skipinitialspace=True)

    # 第一步：观察数据
    #print(dataset.tail(10))
    # 统计数据NA情况 输出列名 NA数量
    #print(dataset.isna().sum())

    # 第二步：数据清洗
    # 丢弃包含NA的行
    dataset = dataset.dropna()
    # 分类数据处理。弹出分类列，即效果为数据集中不包含这个列
    origin = dataset.pop('Origin')
    #print(origin)
    # 分类中包含哪些具体分类,向行添加列,相当于每个行具有了该分类的独热向量.
    # 向量中每个元素进行操作，生成一个新向量
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    #print(dataset.tail(10))

    # 第三步：拆分数据集
    # 数据集中随机取80%作为训练集样本
    train_dataset = dataset.sample(frac=0.8, random_state=1)
    # 数据集中删除训练集的样本。drop可以根据行索引或列删除数据，返回剩余的数据集。
    test_dataset = dataset.drop(train_dataset.index)

    # 第四步：散点图矩阵 观察关系
    # 从dataFrame，根据列名获取数据集合。
    # print(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]])
    #sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    #plt.show()

    # 第五步：组织标签
    # 训练集的标签
    train_labels = train_dataset.pop("MPG")
    # 测试集的标签
    test_labels = test_dataset.pop("MPG")

    # 第六步：数据规范化
    # 数据集的描述，结果为每个列进行数据统计，如每个列的平均数 标准差 最大值 中位数等，可用于规范化
    train_stats = train_dataset.describe()
    # print(train_stats)
    # 矩阵转换 列变行
    train_stats = train_stats.transpose()
    #print(train_stats)

    # z-score方式
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    # 训练集和测试集均进行数据规范化
    normed_train_dataset = norm(train_dataset)
    normed_test_dataset = norm(test_dataset)
    #print(normed_train_dataset.tail(10))

    # 第七步：构建模型
    #print(len(train_dataset.keys()))
    print(f'features size: {len(train_dataset.keys())}')
    #model = build_model(len(train_dataset.keys()))

    # 监控val_loss验证集的loss指标，每patience次迭代之后没有改善则停止迭代
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    epochs = 2000

    # 实例化一种超参调节器，keras-tuner超参调节器 可以帮助编码者找到较好的超参数值，如某个神经层的神经元数量，学习速率等。
    # max_epochs 在不使用超参调节器之前，使用早停法训练模型，得到较好模型时的迭代次数
    # objective 优化的方向
    # directory project_name 记录了调节过程和结果，下次执行时直接拿结果
    tuner = kt.Hyperband(build_model_hyperparameters, objective='val_loss', max_epochs=100, factor=3, directory='my_dir', project_name='intro_to_kt')
    # 查找较好的超参数 参数必须与后续model.fit的参数一致
    tuner.search(normed_train_dataset, train_labels, batch_size=80, epochs=epochs, validation_split=0.2, callbacks=[early_stop])
    # 获取结果
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""regularize_rate:{best_hps.get('regularize_rate')} l1_units:{best_hps.get('l1_units')} l2_units:{best_hps.get('l2_units')} learning_rate:{best_hps.get('learning_rate')}""")
    # 使用调节后的超参数 构建模型
    model = tuner.hypermodel.build(best_hps)

    print(model.summary())

    # 用少量样本测试模型
    #print(model.predict(normed_train_dataset[:10]))

    # 第八步：训练模型
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    # 下边有两种训练模型的办法
    # 1.完成所有迭代

    # 2.使用早停法。若训练集和验证集的误差，没有随着迭代而改善，考虑早停法。
    callbacks = [early_stop, PrintDot()]

    # batch_size 小批次梯度下降算法 其中每批次中样本数量
    # validation_split 从训练集中取百分x的数据作为验证集
    # verbose 对结果无影响 只是每次迭代结果的显示形式
    # callbacks 每迭代的回调
    history = model.fit(normed_train_dataset, train_labels, batch_size=80, epochs=epochs, validation_split=0.2, verbose=0, callbacks=callbacks)
    print()

    # 获取最优的迭代次数
    val_acc_per_epoch = history.history['val_loss']
    best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
    print('best epoch %d', best_epoch)
    # 重新构造模型 并 使用最优迭代次数
    model = tuner.hypermodel.build(best_hps)
    #model = build_model(len(train_dataset.keys()))
    history = model.fit(normed_train_dataset, train_labels, batch_size=80, epochs=best_epoch, validation_split=0.2, verbose=0, callbacks=callbacks)
    print()

    # 第九步：观察训练结果，以便调整学习速率、正则化率等
    # 训练集预测值的衡量标准  与 验证集  对比
    plot_history(history)

    # 第十步：训练集和验证集 指标结果 满足一定条件后，模型应用于测试集。
    # 应用到测试集  观察衡量指标
    print('应用到测试集 观察衡量指标')
    model.evaluate(normed_test_dataset, test_labels, verbose=2)

    # 保存模型方式1：HDF5标准 产生一个文件  较方式2占用空间小
    #save_model_path = 'saveModeL/my_model.h5'
    #model.save(save_model_path)
    # 加载模型
    #new_model = tf.keras.models.load_model(save_model_path)
    #new_model.evaluate(normed_test_dataset, test_labels, verbose=2)

    # 保存模型方式2：saveModel格式 产生多个文件
    #save_model_path = 'saveModel/save_model'
    #model.save(save_model_path)
    #new_model = tf.keras.models.load_model(save_model_path)
    #new_model.evaluate(normed_test_dataset, test_labels, verbose=2)

    # 应用到测试集  得到预测值
    print('应用到测试集  得到预测值')
    test_predictions = model.predict(normed_test_dataset).flatten()

    # 预测值与标签对比
    # 根据给定的x和y数据画散点
    plt.scatter(test_labels, test_predictions)
    # 轴名称
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    #
    plt.axis('square')
    plt.axis('square')
    # 轴的上下限
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    # 画一条线
    plt.plot([-100, 100], [-100, 100])
    plt.show()

    # 误差与数量
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    plt.ylabel("Count")
    plt.show()


def build_model(input_column_size):
    """构建神经网络模型 超参数硬编码"""
    # 使用Sequential构建一个神经网络，在其中设置多个层.
    # 神经网络，例如选择L1正则化，调整正则化率
    regularizer = tf.keras.regularizers.L2(0.01)
    model = keras.Sequential([
        # 使用Dense定义一个神经网络层
        # 第一层 64个神经元，  激活函数relu， 权重使用的正则化方式， 输入的结构
        # 神经元数量可以设置为2的n次方，n越大模型越复杂
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer, input_shape=[input_column_size]),
        # 丢弃，按比例将下一层的输入设置为0
        #layers.Dropout(0.5),
        # 第二层
        layers.Dense(480, activation='relu', kernel_regularizer=regularizer),
        #layers.Dropout(0.5),
        # 第三层
        layers.Dense(1)
    ])

    # 选择一种梯度下降算法，设置学习速率
    # optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=0.0005)
    # optimizer = keras.optimizers.RMSprop(0.001)

    # 自动调整学习速率
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001,
      decay_steps=100,
      decay_rate=1,
      staircase=False)
    optimizer = tf.keras.optimizers.experimental.RMSprop(lr_schedule)

    # 配置和编译模型  损失函数使用均方差  优化器  衡量标准使用绝对平均误差和均方差
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model
def build_model_hyperparameters(hp):
    """构建神经网络模型 hp表示超参调节器结果，使用该结果构建模型"""
    # 使用Sequential构建一个神经网络，在其中设置多个层.
    # 神经网络，例如选择L1正则化，调整正则化率
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_regularize_rate = hp.Choice('regularize_rate', values=[1e-2, 1e-3, 1e-4])
    hp_units_l1 = hp.Int('l1_units', min_value=32, max_value=512, step=32)
    hp_units_l2 = hp.Int('l2_units', min_value=32, max_value=512, step=32)
    regularizer = tf.keras.regularizers.L2(hp_regularize_rate)
    model = keras.Sequential([
        # 使用Dense定义一个神经网络层
        # 第一层 64个神经元，  激活函数relu， 权重使用的正则化方式， 输入的结构
        # 神经元数量为2的n次方，n越大模型越复杂
        layers.Dense(hp_units_l1, activation='relu', kernel_regularizer=regularizer, input_shape=[9]),
        # 丢弃，按比例将下一层的输入设置为0
        #layers.Dropout(0.5),
        # 第二层
        layers.Dense(hp_units_l2, activation='relu', kernel_regularizer=regularizer),
        #layers.Dropout(0.5),
        # 第三层
        layers.Dense(1)
    ])

    # 选择一种梯度下降算法，设置学习速率
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    optimizer = tf.keras.optimizers.experimental.RMSprop(hp_learning_rate)

    # 配置和编译模型  损失函数使用均方差  优化器  衡量标准使用绝对平均误差和均方差
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model

def plot_history(history):
    """绘制 训练集 和 验证集 的 mae mse 曲线，对比"""
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    # 创建一个图形
    plt.figure()
    # 坐标轴的名称
    plt.xlabel('Epoch')
    plt.ylabel('mae')
    # 使用给定的x和y数据画线
    plt.plot(hist['epoch'], hist['mae'], label='train error')
    plt.plot(hist['epoch'], hist['val_mae'], label='val error')
    # y轴的limit
    plt.ylim([0, 20])
    # 在图形上做图
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('mse')
    plt.plot(hist['epoch'], hist['mse'], label='train error')
    plt.plot(hist['epoch'], hist['val_mse'], label='val error')
    plt.ylim([0, 20])
    plt.legend()

    plt.show()

if __name__ == '__main__':
    regression()
