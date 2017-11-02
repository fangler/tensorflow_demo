## 基于Tensorflow的手写数字识别方案

### 环境准备
安装工具和库如下
1. 安装 tensorflow 
    >[官网安装方式](https://www.tensorflow.org/install/install_windows)
    >本人安装在windows下，只需要先安装好python3.6.3版本，配置到系统环境变量
    >
    >然后使用`pip3 install --upgrade tensorflow`即可
    >
2. 安装python的图片库PIL，[**Pillow**](https://github.com/python-pillow/Pillow)
    >`pip3 install Pillow`
    >
    >后面读取手写数字图片的时候会用到
    >

### MNIST数据库

[官网](http://yann.lecun.com/exdb/mnist/)

这是一个手写数字的数据库，提供了六万的训练集和一万的测试集。

进入官网后我们只需要下载红色链接的四个文件即可，下载的四个文件保存在一个目录后续会用到

### 编写代码
此处参考了tensorflow的官方源码和其他同学的方案
官方的源码在[这里](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist)， 这也是一个建议的demo。

后来找到一个更加[完善一点的](https://github.com/wlmnzf/tensorflow-train)，包含了测试数字图片，所以直接拿来用了。

1. 下载的训练数据放在`MNIST_data`目录

2. 需要测试的数据放在`test_num`目录 和 `MNIST_data`是同级目录

3. python代码 和上面两个目录也是同级目录

在命令行执行`python mnist_softmax.py`来测试，会输出对应的结果

本工程已上传至github 作为学习使用~



## TensorBoard搭建

TensorBoard是tensorflow的可视化工具，可以自动显示我们所建造的神经网络流程图

运行方式如下:

```
D:\Codes\tensorflow\tensorflow>python "C:\Program Files\Python36\Lib\site-packag
es\tensorboard\main.py" --logdir logs
```

`python ` **python tensorboard库安装目录下的 main.py**  --logdir  **训练数据目录**
如果**训练数据的目录**是空的话我们在tensorBoard看不到任何东西

如果上一步执行的命令没有错误，命令行会显示`TensorBoard 0.4.0rc2 at http://ubt-zhangfeng:6006 (Press CTRL+C to quit)` 字样，这表示tensorboard启动成功。

在浏览器输入`localhost:6006`即可以进去tensorboard，推荐使用chrome。。。

上面说到如果没有训练数据，进去的时候啥都没有，所以这里我们需要自己创建训练数据。

下面写一个简单的训练方式

`cat.py`

```
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
```

我们读取当前目录下的`cat_walkl.png`图片，然后有原图和旋转90的图，最后将训练结果写入到`logs`目录中。

使用`python cat.py`后，会在`logs`目录生成一个训练的数据文件。 

**注意：由于读取文件使用的是`image0 = tf.image.decode_png(data, channels=4)`， 所以需要保证`cat_walkl.png`是png格式的图片，否则会报错**



这个logs目录刚好是我们启动tensorBoard指向的目录，所以启动tensorBoard后我们可以看到训练数据显示出来到了。。



**目前还没有清楚tensorBoard上的数据和按钮怎么操作。。。后面在学习。**



本文项目已上传至[github](https://github.com/fangler/tensorflow_demo).



