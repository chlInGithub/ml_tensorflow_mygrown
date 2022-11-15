import tensorflow as tf


def test():
    """
    来自tf的初学者示例
    """
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 构建一个神经网络模型，构建线性层栈
    model = tf.keras.models.Sequential([
        # 输入
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # 该层神经元数量为128，选择relu激活函数
        tf.keras.layers.Dense(128, activation='relu'),
        # 选择丢弃正则化
        tf.keras.layers.Dropout(0.2),
        # 输出神经元数量
        tf.keras.layers.Dense(10)
    ])

    # 训练集第一个样本带入模型，得到预测值
    m = model(x_train[:1])
    predictions = model(x_train[:1]).numpy()
    print("第一个样本的预测值：")
    print(predictions)

    # 离散类别交叉熵 损失函数 使用对数进行计算
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # 标签与预测值，计算损失
    print("第一个样本的损失：")
    print(loss_fn(y_train[:1], predictions).numpy())

    # 配置和编译模型
    model.compile(
        # 优化器 例如指定学习速率 默认0.001
        optimizer='adam',
        # 损失函数
        loss=loss_fn,
        # 衡量方式 使用精准度
        metrics=['accuracy']
    )

    # 使用训练集训练模型 迭代5次
    model.fit(x_train, y_train, epochs=5)

    # 使用测试集验证模型
    model.evaluate(x_test,  y_test, verbose=2)

if __name__ == '__main__':
    test()