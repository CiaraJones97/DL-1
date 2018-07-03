import tensorflow as tf
import random

# Set the three random matrices
a = tf.constant([[random.randint(1,5), random.randint(1,5)],
                 [random.randint(1,5), random.randint(1,5)]], dtype= tf.int32, shape = [2,2])
b = tf.constant([[random.randint(1,5), random.randint(1,5)],
                 [random.randint(1,5), random.randint(1,5)]], dtype= tf.int32, shape = [2,2])
c = tf.constant([[random.randint(1,5), random.randint(1,5)],
                 [random.randint(1,5), random.randint(1,5)]], dtype = tf.int32, shape = [2,2])

# Calculate a^2
power = tf.pow(a,2)
# Calculate a^2 + b
addition = tf.add(power,b)
# Calculate (a^2 +b) * c
multiply = tf.matmul(addition, c)

with tf.Session() as sess:
    print("A: ", sess.run(a))
    print("B: ", sess.run(b))
    print("C: ", sess.run(c))
    print("a^2: ", sess.run(power))
    print("a^2+b: ", sess.run(addition))
    print("(a^2+b)*c = ", sess.run(multiply))
