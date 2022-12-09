# This is a sample Python script.
import matplotlib.pyplot
import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def testArray():
    import numpy as np
    x = np.random.randn(10, 5)
    print(x)
    #
    print(f'取行，0至index-1行 {x[:5]}')
    #
    print(f'取行，从index行起所有行 {x[1:]}')

    print(f'取列，index列 {x[...,1]}')
    print(f'取列，从0到index-1列{x[...,:4]}')
    print(f'取列，index起所有列{x[...,1:]}')

def drawScatterDiagram():
    """
    测试绘散点图
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 随机rowNum行columnNum列矩阵
    row_num = 10
    column_num = 10
    x = np.random.randn(row_num, column_num) + 5
    # 随机3行2列矩阵
    y = np.random.randn(row_num, column_num) * 3

    z = np.random.randn(row_num, column_num)

    print(x)
    print(y)

    # 散点图  row_num * column_num 个点
    #plt.scatter(x, y)
    #plt.show()

    # 三维散点图
    axis = plt.figure().add_subplot(1, 1, 1, projection='3d')
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    plt.scatter(x, y, z)
    plt.show()

    # 连线
    # plt.plot(x, y)
    # plt.show()

def test_np_expand_dims():
    '''
    测试 在数组上添加坐标轴
    :return:
    '''
    # 一维数组，坐标轴index0 [1 3 5 7 9]
    x = np.arange(1, 10, 2)
    # 1
    print(x, x.shape, x[0])

    # 在原坐标轴中插入坐标轴，插入到坐标轴index0处，现在有两个坐标轴index0和1
    # [[1 3 5 7 9]]
    x1 = np.expand_dims(x, 0)
    # [1 3 5 7 9]
    print(x1, x1.shape, x1[0])

    # 在原坐标轴中插入坐标轴，插入到坐标轴index1处
    # [[1]
    #  [3]
    #  [5]
    #  [7]
    #  [9]]
    x2 = np.expand_dims(x, 1)
    # [1]
    print(x2, x2.shape, x2[0])

    # [1 3 5 7 9]  -> [[1 3 5 7 9]] -> [[[1 3 5 7 9]]]
    x3 = np.expand_dims(x, (0, 1))
    # [1 3 5 7 9]
    print(x3, x3[0, 0])

    # [1 3 5 7 9]  -> [[1 3 5 7 9]] ->
    # [[[1]
     #  [3]
     #  [5]
     #  [7]
     #  [9]]]
    x4 = np.expand_dims(x, (0, 2))
    # 1
    print(x4, x4[0,0,0])

    # 等同于 np.expand_dims(x, (0, 2))
    x5 = np.expand_dims(x, (2, 0))
    # [1]
    print(x5, x5[0,0])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_np_expand_dims()
    # testArray()
    #print_hi('PyCharm')
    #drawScatterDiagram()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
