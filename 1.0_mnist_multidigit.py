import tensorflow as tf
import cv2
import numpy as np
print("Tensorflow version " + tf.__version__)

INPUT_IMAGE = "digitgroup-02.jpg"
OUTPUT_IMAGE = "output.jpg"


##MODEL##################################################
# neural network structure for this sample:
#
# (input data, 1-deep)                 X [batch, 28, 28, 1]
# conv. layer 6x6x1=>6 stride 1        W1 [5, 5, 1, 6]          B1 [6]
#                                      Y1 [batch, 28, 28, 6]
# conv. layer 5x5x6=>12 stride 2       W2 [5, 5, 6, 12]         B2 [12]
#                                      Y2 [batch, 14, 14, 12]
# conv. layer 4x4x12=>24 stride 2      W3 [4, 4, 12, 24]        B3 [24]
#                                      Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
# fully connected layer (relu+dropout) W4 [7*7*24, 200]         B4 [200]
#                                      Y4 [batch, 200]
# fully connected layer (softmax)      W5 [200, 10]             B5 [10]
#                                      Y [batch, 10]
#
# # input X: 28x28 grayscale image
X = tf.placeholder(tf.float32, [1, 28, 28, 1],name="Input_Image")
#
# # three convolutional layers with their channel counts, and a
# # fully connected layer (the last layer has 10 softmax neurons)
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1), name="W1")  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]), name="B1")
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1), name="W2")
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]), name="B2")
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1), name="W3")
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]), name="B3")

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1), name="W4")
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]), name="B4")
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1), name="W5")
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]), name="B5")

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME', name="conv_X_W1") + B1, name="Y1_relu1")
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME', name="conv_Y1_W2") + B2, name="Y2_relu")
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME',name="conv_Y2_W3") + B3, name="Y3_relu")

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M], name="reshape_Y3")

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4, name="relu_Y4")
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits,name="Softmax_Y")

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
#######################################################

##Input Image##########################################
image = cv2.imread("input_data/" + INPUT_IMAGE)
#######################################################

##GrayScale Image######################################
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
if np.mean(img_gray) > 127:
    img_gray = 255 - img_gray
check = img_gray[:]
#######################################################

##Contours############################################
ret,thresh = cv2.threshold(check, 127, 255, 0)
check,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
# cv2.drawContours(image, contours, -1, (0,255,0), 1)
######################################################

##Run Session########################################
with tf.Session() as sess:
    saver.restore(sess, "1.0_checkpoint/mnist_model.ckpt")
    # writer = tf.summary.FileWriter('./1.0_graphs', sess.graph) #--Save History for Tensorboard
    for i in range(len(contours)):
        c,r,w,h = cv2.boundingRect(contours[i])

        ##Extract Image##############################
        factor = w/h
        height = 22
        width = int(round(factor*height))
        extract = cv2.resize(check[r:r+h,c:c+w], (width, height), interpolation = cv2.INTER_LINEAR)
        res = np.zeros([28,28])
        c_ = int(round((28-width)/2))
        r_ = int(round((28-height)/2))
        if width > 28:
            # print(width)
            # continue
            width = 22
            res = np.zeros([28,28])
            c_ = int(round((28-width)/2))
            extract = cv2.resize(check[r:r+h,c:c+w], (width, height), interpolation = cv2.INTER_LINEAR)
        res[r_:r_+height, c_:c_+width] = extract
        #############################################

        # cv2.drawContours(image, contours, i, (0,255,0), 1)
        cv2.rectangle(image,(c,r),(c+w,r+h),(0,0,255),1)

        ##Detect#####################################
        Xnp = np.reshape(res, (1, 28, 28, 1))
        # print(Xnp.shape)
        Xnp.astype(dtype=np.float32)
        output = sess.run(Y[0], feed_dict={X: Xnp})
        if output[np.argmax(output)] >= 0.5:
            number = np.argmax(output)
            # print(output)  #--debug
        cv2.putText(image, str(number),(c,r),cv2.FONT_HERSHEY_COMPLEX,1,color=(127,127,0),thickness=2)
        # cv2.imshow("extrct",res)      #--debug
        # print(number)                 #--debug
        # cv2.imshow("image",image)
        # cv2.waitKey(0)                #--debug
        #############################################

##Display############################################
# writer.close() #--Tensorboard
cv2.imshow("image",image)
cv2.imwrite("output_data/"+ OUTPUT_IMAGE,image)
cv2.waitKey(0)
cv2.destroyAllWindows()
####################################################
