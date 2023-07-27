import hashlib
import itertools
import secrets
import threading
import unittest


def media(list: []):
    sortedList = sorted(list)
    if len(sortedList) % 2 == 1:
        return sortedList[(len(sortedList) - 1) // 2]
    else:
        return sum(sortedList[(len(sortedList) // 2) + sortedList[len(sortedList) // 2 - 1]]) // 2


def guess_a_num():
    import random
    guess_count = 0
    name = input("hello, what's your name ?")
    print("welcome, {}".format(name))
    target_num = random.randint(10, 50)
    while guess_count < 5:
        guess_num = int(input("you can guess the num and input it:"))
        guess_count = guess_count + 1
        if guess_num < target_num:
            print("the num that you input is low")
        elif guess_num > target_num:
            print("the num of you input is big")
        else:
            break
    print("targetNum is {}, guessNum is {}".format(target_num, guess_num))
    if guess_num == target_num:
        print("you get it!")
    else:
        print("sorry")


def many_args(commonParam, commonIntParam: int, *params, **keywords):
    print("commonParam ", commonParam)
    print("commonIntParam ", commonIntParam)
    for e in params:
        print("params e ", e)
    for e in keywords:
        print("keywords e ", e, keywords[e])


from enum import Enum


class MyEnum(Enum):
    A = 1
    B = 2


class MyTestCase(unittest.TestCase):
    def test_something(self):
        print([1] * 10)
        myList = list(range(1, 8))
        print(myList)
        self.assertEqual(media(myList), 4)  # add assertion here

    def test_guess_num(self):
        guess_a_num()

    def test_many_args(self):
        many_args("common", "1", "2", "3", a="1", b="2")

    def test_enum(self):
        MyEnum.A
        print(MyEnum(1))
        print(MyEnum(2))
        try:
            print(MyEnum(3))
        except ValueError:
            print()

    def test_random(self):
        from random import choices
        for i in range(0, 20):
            result = choices(list(range(1, 10)), list(reversed(range(1, 10))), k=1)
            print("random choice is ", result)

    def test_itertools(self):
        for x in itertools.count(10):
            print(x)

    def test_secret(self):
        print(secrets.token_hex(16))
        print(secrets.token_hex(32))
        print(secrets.token_urlsafe(32))

    def test_hashlib(self):
        str1 = 'this is xxx, and it is from xxxx.'
        b1 = str1.encode()
        print(hashlib.sha256(b1).hexdigest())

    def test_threading(self):
        def print_thread_info():
            print("active_count : ", threading.active_count())
            print("current : {} , id : {} ".format(threading.current_thread(), threading.get_ident()))

        t1 = threading.Thread(target=print_thread_info)
        t2 = threading.Thread(target=print_thread_info)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def test_class(self):
        class MyClass():
            # class成员变量
            class_field01: int
            class_field02: list = []

            # 定义实例成员变量 方式1  使用_或__前缀，注意实例无法直接访问，必须提供读写方法。
            __instance_field: int = 1
            __instance_field01: list = list()
            _instance_field02: str = ''

            # 定义实例成员变量 方式2 通过构造函数 该方式定义的成员变量可以被直接访问 不需要提供读写方法  使用起来方便
            def __init__(self, val1):
                self.instance_field01 = val1

            def get_instance_field(self):
                return self.__instance_field

            def set_instance_field(self, val):
                self.__instance_field = val

            def get_instance_field02(self):
                return self._instance_field02

            def set_instance_field02(self, val):
                self._instance_field02 = val

        MyClass.class_field01 = 1
        MyClass.class_field02.extend('Hello')
        print(f'class_field01={MyClass.class_field01}, class_field02={MyClass.class_field02}')

        my_class = MyClass(20)
        my_class.set_instance_field(10)
        my_class.set_instance_field02('val for _instance_field02')
        print(
            'from instance 01 : '
            f'get_instance_field01={my_class.get_instance_field()}, '
            f' get_instance_field02={my_class.get_instance_field02()}, '
            f' instance_field01={my_class.instance_field01}, '
            f' class_field01={my_class.class_field01}')

        my_class02 = MyClass(30)
        print(
            'from instance 02 : '
            f'get_instance_field01={my_class02.get_instance_field()}, '
            f' get_instance_field02={my_class02.get_instance_field02()}, '
            f' instance_field01={my_class02.instance_field01}, '
            f' class_field01={my_class02.class_field01}')


if __name__ == '__main__':
    unittest.main()
