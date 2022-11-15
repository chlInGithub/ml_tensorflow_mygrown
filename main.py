# This is a sample Python script.
import matplotlib.pyplot


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def testArray():
    import numpy as np
    x = np.random.randn(10, 2, 2)
    print(x)
    print(x[:1])

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #testArray()
    #print_hi('PyCharm')
    drawScatterDiagram()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
