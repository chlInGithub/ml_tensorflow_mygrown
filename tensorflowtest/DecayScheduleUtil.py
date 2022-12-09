"""
学习速率衰减计划,随着模型训练过程的进行，学习速率逐渐衰减即减小。
"""
import tensorflow as tf


def getCosineDecay():
    """
    余弦。
    alpha 必须小于1，否则学习速率增加。
    decay_steps 越大，学习速率变化越小。
    alpha 越大，学习速率变化越小。
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_learning_rate * decayed
    :return:
    """
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001,
                                                            decay_steps=10000,
                                                            alpha=0.5)
    return lr_schedule

def getInverseTimeDecay():
    """
    随着step的增加，学习速率一定减少。decay_steps一定时，decay_rate越大，学习速率变化越大；decay_rate一定时，decay_steps越小，学习速率变化越大。
    staircase=False 顺滑的逐渐减小 initial_learning_rate / (1 + decay_rate * step / decay_step)
    staircase=Ture 楼梯式减少 initial_learning_rate / (1 + decay_rate * floor(step / decay_step))
    :return:
    """
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.001,
                                                                 decay_steps=1000,
                                                                 decay_rate=0.9,
                                                                 staircase=False)
    return lr_schedule


def getExponentialDecay():
    """
    指数型  decay_rate 必须小于1，否则学习速率会指数型变大。
     decay_steps一定时，decay_rate越小，学习速率变化越大；decay_rate一定时，decay_steps越小，学习速率变化越快；
    staircase=False initial_learning_rate * decay_rate ^ (step / decay_steps)
    staircase=Ture 楼梯式 initial_learning_rate * decay_rate ^ floor(step / decay_step)
    :return:
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,
                                                                 decay_steps=1000,
                                                                 decay_rate=0.9,
                                                                 staircase=False)
    return lr_schedule



def getPiecewiseConstantDecay():
    """
    分段式
    step <= boundaries[0], use values[0]
    step > boundaries[0] and step <= boundaries[1], use values[1]
    step > boundaries[1], use values[2]
    :return:
    """
    boundaries = [1000, 5000]
    values = [1.0, 0.5, 0.1]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    return lr_schedule



def getPolynomialDecay():
    """
    多项式
    decay_steps 越大，学习速率变化越慢。
    power=0，学习速率不变。
    power=1，学习速率线性变化。
    power > 1, 学习速率变小。
    step = min(step, decay_steps)
    ((initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ^ (power)) + end_learning_rate
    :return:
    """
    starter_learning_rate = 0.1
    end_learning_rate = 0.01
    decay_steps = 5000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0)
    return learning_rate_fn

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    # for question: Initializing libiomp5md.dll, but found libiomp5 already initialized.
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    step = np.linspace(0, 10000)

    lr_schedule = getInverseTimeDecay()
    lr = lr_schedule(step)
    plt.figure(figsize=(8, 6))
    plt.plot(step, lr)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate by InverseTimeDecay')
    plt.show()

    lr_schedule = getExponentialDecay()
    lr = lr_schedule(step)
    plt.figure(figsize=(8, 6))
    plt.plot(step, lr)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate by ExponentialDecay')
    plt.show()

    lr_schedule = getCosineDecay()
    lr = lr_schedule(step)
    plt.figure(figsize=(8, 6))
    plt.plot(step, lr)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate by CosineDecay')
    plt.show()

    lr_schedule = getPolynomialDecay()
    lr = lr_schedule(step)
    plt.figure(figsize=(8, 6))
    plt.plot(step, lr)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate by PolynomialDecay')
    plt.show()

    try:
        lr_schedule = getPiecewiseConstantDecay()
        lr = lr_schedule(step)
        plt.figure(figsize=(8, 6))
        plt.plot(step, lr)
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate by PiecewiseConstantDecay')
        #plt.show()
    except:
        pass
