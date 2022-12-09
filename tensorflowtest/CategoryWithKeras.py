'''
使用神经网络实现分类模型
收获：
RGB8 像素的格式化
样本数据维度必须相同
若神经网络输入为多维数组，需要展平为一维数组
构建模型，层数越多，神经元数量越多，模型越复杂，拟合程度越好
神经网络输出层神经元数量 = 分类数量
使用softmax转为概率
分类模型使用 交叉熵 作为 损失函数
分类模型衡量指标 精准度
'''

# for question: Initializing libiomp5md.dll, but found libiomp5 already initialized.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def category():
    # Fashion MNIST 数据集，该数据集包含 10 个类别的 70,000 个灰度图像。这些图像以低分辨率（28x28 像素）展示了单件衣物
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 训练集 60000个图像 每个图像28*28像素，每个像素的颜色使用简单的RGB8表示；
    # 标签是 0 - 9
    print(train_images.shape, train_labels.shape)

    # 标签值 对应的 类名称
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 必须预处理数据
    # 创建新画布
    #plt.figure()
    # 将图像加到画布中
    #plt.imshow(train_images[1500])
    # 颜色条 数值与颜色的直观表示
    #plt.colorbar()
    # 是否显示网格
    #plt.grid(True)
    #plt.show()

    # 训练集和测试集 数据进行规范化 都除以255 变为0-1之间的数
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 画布，figsize 规定画布的宽高 单位英寸
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     # 添加子画布，位于5*5子画布集中的第i+1位置
    #     plt.subplot(5, 5, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(True)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    # 构建模型，层数越多，神经元数量越多，模型越复杂，拟合程度越好
    model = tf.keras.Sequential([
        # 平铺 将二维数组转为一维数组
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.L2(0.005), activation='relu'),
        # 表示10个分类
        tf.keras.layers.Dense(10)
    ])

    # 编译模型
    # 损失函数 测量模型在训练期间的准确程度
    # 指定优化器 决定如何更新模型，如速率
    # 衡量指标 监控训练和测试步骤
    model.compile(
        # tf.keras.optimizers.Adam 默认学习速率0.001
        optimizer='adam',
        # 稀疏的明确的交叉熵
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 训练 模型与训练数据进行拟合  输出损失和指标
    model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

    # 评估 在测试集合上评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('test accuracy:', test_acc, '\n')

    predictions = model.predict(test_images)
    print('no softmax, predict:', predictions[1000], '\n')
    # softmax将线性输出转化为概率
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    # 预测测试集
    predictions = probability_model.predict(test_images)
    print('use softmax, predict:', predictions[1000], '\n')
    # argmax 返回最大值的index
    print(
        np.argmax(predictions[100]), test_labels[100],
        np.argmax(predictions[1000]), test_labels[1000],
        np.argmax(predictions[1500]), test_labels[1500]
    )

    # 预测单个图片
    index = 100
    img = test_images[index]
    # print(img.shape)
    # 添加index0 坐标轴，得到由单个img构成的数据集
    imgs = np.expand_dims(img, 0)
    # print(img.shape)
    predictions = probability_model.predict(imgs)
    print(np.argmax(predictions[0]), test_labels[index])


if __name__ == '__main__':
    category()