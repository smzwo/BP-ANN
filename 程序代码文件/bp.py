import numpy as np
import tensorflow as tf
import random
import os
import glob
import cv2
batch_size = 40
def SetLoadImage(folder=u'/Users/sunmingzhe/Desktop/大三下实验/模式识别ORL的BP神经网络/ORL_faces-master/data_faces', sampleCount=5): #加载图像集，随机选择sampleCount张图片用于训练
    trainData = [];
    testData = [];
    yTrain2=[];
    yTest2=[]
    for k in range(40):
        yTrain1 = np.zeros(40)
        yTest1 = np.zeros(40)
        folder2 = os.path.join(folder, 's%d' % (k+1))
        data=[cv2.imread(d,0) for d in glob.glob(os.path.join(folder2, '*.pgm'))]
#cv2.imread()第二个参数为0的时候读入为灰度图，即使原图是彩色图也会转成灰度图#glob.glob匹配所有的符合条件的文件，并将其以list的形式返回
        sample = random.sample(range(10), sampleCount)
        trainData.extend([data[i].ravel() for i in range(10) if i in sample])
	#降维
        testData.extend([data[i].ravel() for i in range(10) if i not in sample])#40*5*112*92
        yTrain1[k]=1
        yTest1[k]=1
        yTrain= np.tile(yTrain1,5)
        yTest=np.tile(yTest1,5)
        yTrain2.extend(yTrain)
        yTest2.extend(yTest)
    return np.array(trainData),  np.array(yTrain2), np.array(testData), np.array(yTest2)
#构造添加一个神经层的函数
def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    layer_name="layer%s"%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b')
            tf.summary.histogram(layer_name+"/biases",biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs',outputs)
        return outputs
def main():
xTrain_, yTrain, xTest_, yTest = SetLoadImage()
yTrain.shape=200,40
    yTest.shape=200,40
    # 放置占位符，用于在计算时接收输入值
    with tf.name_scope('inputs'):
        x = tf.placeholder("float", [None, 10304],name='x_in')
        y_ = tf.placeholder("float", [None, 40],name='y_in')
    # 创建两个变量，分别用来存放权重值W和偏置值b
    # 为了进行训练，需要把正确值一并传入网络
    #定义隐含层，输入神经元个数=特征10304
    l1 = add_layer(x,10304,1000,n_layer=1 ,activation_function=tf.nn.sigmoid)
    l2 = add_layer(l1, 1000, 700, n_layer=2, activation_function=tf.nn.sigmoid)
    l3 = add_layer(l2, 700,200, n_layer=3, activation_function=tf.nn.sigmoid)
    #定义输出层。此时的输入就是隐藏层的输出——90，输入有90层（隐藏层的输出层），输出有40层
    y= add_layer(l3,200,40,n_layer=4, activation_function=tf.nn.softmax
    # 计算交叉墒
    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        tf.summary.scalar('loss', cross_entropy )  # 用于观察常量的变化
    # 定义命名空间，使用tensorboard进行可视化
    with tf.name_scope('train'):
        # 使用梯度下降算法以0.01的学习率最小化交叉墒
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # 开始训练模型，每次都会随机抓取训练数据中的10条数据，然后作为参数替换之前的占位符来运行train_step
    def return_next_batch(batch_size, pos):
        start = pos * batch_size
        end = start + batch_size
        return xTrain_[start:end], yTrain[start:end]
    # 启动初始化，为其分配固定数量的显存
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #tf.merge_all_summaries() 方法会对我们所有的 summaries 合并到一起
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', tf.get_default_graph())
    # 初始化之前创建的变量的操作
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            for j in range(5):
                batch_x, batch_y =return_next_batch(batch_size, j)
                sess.run(train_step, feed_dict={x: np.matrix( batch_x/255), y_: np.matrix(batch_y)})
                print(sess.run(cross_entropy, feed_dict={x: np.matrix((batch_x/255)), y_: np.matrix(batch_y)}))
            if i % 50 == 0: 
                result = sess.run(merged, feed_dict={x: np.matrix((batch_x/255)), y_: np.matrix(batch_y)})
                writer.add_summary(result, i)
        saver.save(sess, "./model/model.ckpt")
    # 评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 计算正确预测项的比例，因为tf.equal返回的是布尔值，使用tf.cast可以把布尔值转换成浮点数，tf.reduce_mean是求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 在session中启动accuracy，输入是orl中的测试集
    print("识别率:%.2f%%" %((sess.run(accuracy, feed_dict={x:np.matrix(xTest_/255), y_: np.matrix(yTest)}))*100))

if __name__ == '__main__':
    main()
