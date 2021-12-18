# -*- autograph的机制原理
import tensorflow as tf


@tf.function(autograph=True)
def myadd(a, b):
    for i in tf.range(3):
        tf.print(i)
    c = a + b
    print("tracing")
    return c


#  一、生成动态图
# 第一件事情是创建计算图。
# 第二件事情是执行计算图。
# 因此我们先看到的是第一个步骤的结果：即Python调用标准输出流打印"tracing"语句。
# 然后看到第二个步骤的结果：TensorFlow调用标准输出流打印1,2,3。
myadd(tf.constant("hello"), tf.constant("world"))

#  二、重用动态图
# 再次用相同的输入参数类型调用这个被@tf.function装饰的函数时
# 只会发生一件事情，那就是上面步骤的第二步，执行计算图。
# 所以这一次我们没有看到打印"tracing"的结果。
myadd(tf.constant("good"), tf.constant("morning"))

#  三、重建动态图
# 再次用不同的的输入参数类型调用这个被@tf.function装饰的函数时
# 由于输入参数的类型已经发生变化，已经创建的计算图不能够再次使用。
# 需要重新做2件事情：创建新的计算图、执行计算图。
# 所以我们又会先看到的是第一个步骤的结果：即Python调用标准输出流打印"tracing"语句。
# 然后再看到第二个步骤的结果：TensorFlow调用标准输出流打印1,2,3。

myadd(tf.constant(1), tf.constant(2))

#  四、总结
# 了解了以上Autograph的机制原理，我们也就能够理解Autograph编码规范的3条建议了。
#
# 1，被@tf.function修饰的函数应尽量使用TensorFlow中的函数而不是Python中的其他函数。
# 例如使用tf.print而不是print.
# 解释：Python中的函数仅仅会在跟踪执行函数以创建静态图的阶段使用，
# 普通Python函数是无法嵌入到静态计算图中的，所以 在计算图构建好之后再次调用的时候，
# 这些Python函数并没有被计算，而TensorFlow中的函数则可以嵌入到计算图中。
# 使用普通的Python函数会导致 被@tf.function修饰前【eager执行】
# 和被@tf.function修饰后【静态图执行】的输出不一致。
#
# 2，避免在@tf.function修饰的函数内部定义tf.Variable.
# 解释：如果函数内部定义了tf.Variable,那么在【eager执行】时，
# 这种创建tf.Variable的行为在每次函数调用时候都会发生。
# 但是在【静态图执行】时，这种创建tf.Variable的行为只会发生在第一步跟踪Python代码逻辑创建计算图时，
# 这会导致被@tf.function修饰前【eager执行】和被@tf.function修饰后【静态图执行】的输出不一致。
# 实际上，TensorFlow在这种情况下一般会报错。
#
# 3，被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。
# 解释：静态计算图是被编译成C++代码在TensorFlow内核中执行的。
# Python中的列表和字典等数据结构变量是无法嵌入到计算图中，
# 它们仅仅能够在创建计算图时被读取，在执行计算图时是无法修改Python中的列表或字典这样的数据结构变量的。
