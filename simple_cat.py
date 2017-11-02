import tensorflow as tf

file = open('cat_walk.png', 'rb')
data = file.read()
file.close()

image0 = tf.image.decode_png(data, channels=4)
image = tf.expand_dims(image0, 0)

sess = tf.Session()
writer = tf.summary.FileWriter('logs')
summary_op = tf.summary.image("image1", image)

rotated_image = tf.image.rot90(image0, k=1)
summary_rotated = tf.summary.image('image2b', tf.expand_dims(rotated_image, 0))
summary1 = sess.run(summary_rotated)
writer.add_summary(summary1)

summary = sess.run(summary_op)
writer.add_summary(summary)

writer.close()
sess.close()
