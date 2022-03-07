
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import time


losses = []

lambd = 1
regularizer = 0
learn_rate = 5e-4
archi = [16,32,32,64,128]
keep_prob = 1
stddev = 0.05#0.02#0.05
regularizer = tf.contrib.layers.l2_regularizer(scale=5e-5)

TotTrainData = 78000
TotTestData = 2000
num_iters = 1000000
num_iters2 = 200
batch_size = 500
num_batch = (TotTrainData//batch_size)
TotTrainData = num_batch*batch_size

input_dim1 = 64
input_dim2 = 64
output_dim = 6

dataset_pre = np.load('ellipse_img.npy')
dataset2_pre = np.load('ellipse_compliance.npy')
dataset = np.zeros((dataset_pre.shape[0],dataset_pre.shape[1]))
dataset2 = np.zeros((dataset2_pre.shape[0],dataset2_pre.shape[1]))

np.random.seed(1000)
dataset_index = np.random.permutation(dataset_pre.shape[0])
for i in range(dataset_pre.shape[0]):
	dataset[i,:] = dataset_pre[dataset_index[i],:]
	dataset2[i,:] = dataset2_pre[dataset_index[i],:]

dataset_reshaped = dataset.reshape((dataset.shape[0],input_dim1,input_dim2,1))
# x_data = dataset_reshaped[0:TotTrainData,:,:,:]      # <--- Uncomment this section in order to provide the training dataset
# y_data_pre = dataset2[0:TotTrainData,:].reshape(TotTrainData,output_dim)
# x_data_pred = dataset_reshaped[0:1000,:,:,:]
# y_data_pred_pre = dataset2[0:1000,:].reshape(1000,output_dim)
# x_data_test = dataset_reshaped[TotTrainData:TotTrainData+TotTestData,:,:,:]
# y_data_test_pre = dataset2[TotTrainData:TotTrainData+TotTestData,:].reshape(TotTestData,output_dim)
x_data_test = dataset_reshaped[:,:,:,:]     # <---- Comment this section for retraining the model
y_data_test_pre = dataset2[:,:].reshape(TotTestData,output_dim)

# y_data = np.zeros((y_data_pre.shape[0],y_data_pre.shape[1]))     # <---- Uncomment this section for retraining the model
# y_data_pred = np.zeros((y_data_pred_pre.shape[0],y_data_pred_pre.shape[1]))
# y_data_test = np.zeros((y_data_test_pre.shape[0],y_data_test_pre.shape[1]))
y_data_test = np.zeros((y_data_test_pre.shape[0],y_data_test_pre.shape[1]))

dataset_norm_max = np.array([[8.81035523e10,3.71412684e10,5.67242560e09,8.82832606e10,5.67249178e09,2.52859939e10]])  # <--- Normalization for ellipse dataset
dataset_norm_min = np.array([[1.69938962e10,3.68522556e09,-5.91331492e09,1.34373105e10,-5.60411153e09,2.75239281e09]])

# dataset_norm_max = np.array([[6.96788743e10,2.66089327e10,4.31007681e09,6.99159122e10,5.25037685e09,1.93591978e10]])   # <--- Normalization for circle square dataset
# dataset_norm_min = np.array([[1.62600719e10,3.23096317e09,-5.86459610e09,1.37196507e10,-4.80139401e09,1.26109822e09]])


for i in range(y_data.shape[1]):
    # y_data[:,i] = (y_data_pre[:,i]-np.amin(y_data_pre[:,i]))/(np.amax(y_data_pre[:,i])-np.amin(y_data_pre[:,i]))  # <---- Uncomment this section to retrain the model
    # y_data_pred[:,i] = (y_data_pred_pre[:,i]-np.amin(y_data_pre[:,i]))/(np.amax(y_data_pre[:,i])-np.amin(y_data_pre[:,i]))
    # y_data_test[:,i] = (y_data_test_pre[:,i]-np.amin(y_data_pre[:,i]))/(np.amax(y_data_pre[:,i])-np.amin(y_data_pre[:,i]))
    y_data_test[:,i] = (y_data_test_pre[:,i]-dataset_norm_min[0,i])/(dataset_norm_max[0,i]-dataset_norm_min[0,i]) # <---- Comment this section to retrain the model
print('--load finish--')


def get_loss(loss):
    print(loss)
    losses.append(loss)

def conv2d(x, w1, w2, win, wout, wx, wy):
	with tf.name_scope('mat_pred'):
	    w = tf.Variable(tf.truncated_normal([w1,w2,win,wout], stddev=stddev))
	    b = tf.Variable(tf.zeros([wout]))
	    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
	    return tf.nn.relu((tf.nn.conv2d(x, w, strides=[1,wx,wy,1], padding='SAME'))+b)

	
def add_layer(L_Prev, num_nodes_LPrev, num_nodes_LX, activation_LX):
	with tf.name_scope('mat_pred'):
		Weights_LX = tf.Variable(tf.random_normal([num_nodes_LPrev,num_nodes_LX],stddev=stddev),name = 'w')
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, Weights_LX)
		biases_LX = tf.Variable(tf.zeros([1,num_nodes_LX]),name = 'b')
		xW_plus_b_LX = tf.matmul(L_Prev,Weights_LX)+biases_LX
		if activation_LX is None:
			LX = xW_plus_b_LX
		else:
			LX = tf.add(xW_plus_b_LX,activation_LX(xW_plus_b_LX))
		return LX



x = tf.placeholder(tf.float32, [None, input_dim1, input_dim2, 1])  
y = tf.placeholder(tf.float32, [None, output_dim])

## conv1 layer ##
c1 = conv2d(x,3,3,1,archi[0],2,2)		#64x64x1 --> 32x32x16
c2 = conv2d(c1,3,3,archi[0],archi[1],2,2)	#32x32x16 --> 16x16x32
c3 = conv2d(c2,3,3,archi[1],archi[2],2,2)	#16x16x32 --> 8x8x32
c4 = conv2d(c3,3,3,archi[2],archi[3],2,2)	#8x8x32 --> 4x4x64
c5 = conv2d(c4,3,3,archi[3],archi[4],2,2)	#4x4x64 --> 2x2x128

## fully connected layer ##
flat1 = tf.reshape(c5, [-1, 4*archi[4]])
L1 = add_layer(flat1,4*archi[4],128,tf.nn.tanh)
L2 = add_layer(L1,128,32,tf.nn.tanh)

prediction = add_layer(L2,32,output_dim,None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_mean(tf.square(y-prediction)))

saver = tf.train.Saver()

def FirstOrder():
	train = tf.train.AdamOptimizer(learn_rate).minimize(loss)
	iii = 0
	best_loss = 100000#7.7710e-05

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		savefileid = '/checkpoint/cnn_save_ellipse/model'

		plt.figure()
		plt.ion()

		for step in range(num_iters):
			start = time.time()
			for batch_i in range(num_batch):
				x_batch = x_data[batch_i*batch_size:(batch_i+1)*batch_size,:,:,:]
				y_batch = y_data[batch_i*batch_size:(batch_i+1)*batch_size,:]
				sess.run([train],feed_dict={x:x_batch,y:y_batch})
				losses.append(np.mean(sess.run(loss,feed_dict={x:x_batch,y:y_batch})))
				iii = iii+1
			end = time.time()
			if losses[iii-1] < 1e-9:
				break
			print('iter: {0}, loss = {1}, time = {2}'.format(step,losses[iii-1],(end-start)))
			if (step+1)%10 == 0 :
				if losses[-1] < best_loss:
					saver.save(sess, savefileid)
					best_loss = losses[-1]

			if (step+1)%5 == 0:
				plt.gcf().clear()
				prediction1 = sess.run(prediction,feed_dict={x:x_data_test})

				### Test data plot ###		
				
				y_data_test1 = y_data_test[:,0]
				y_data_test2 = y_data_test[:,1]
				y_data_test3 = y_data_test[:,2]
				y_data_test4 = y_data_test[:,3]
				y_data_test5 = y_data_test[:,4]
				y_data_test6 = y_data_test[:,5]

				plt.subplot(2,3,1)
				plt.scatter(y_data_test1,prediction1[:,0])
				plt.plot(y_data_test1,y_data_test1,'r-')
				plt.plot(y_data_test1,y_data_test1+0.05,'c-')
				plt.plot(y_data_test1,y_data_test1-0.05,'c-')

				plt.subplot(2,3,2)
				plt.scatter(y_data_test2,prediction1[:,1])
				plt.plot(y_data_test2,y_data_test2,'r-')
				plt.plot(y_data_test2,y_data_test2+0.05,'c-')
				plt.plot(y_data_test2,y_data_test2-0.05,'c-')

				plt.subplot(2,3,3)
				plt.scatter(y_data_test3,prediction1[:,2])
				plt.plot(y_data_test3,y_data_test3,'r-')
				plt.plot(y_data_test3,y_data_test3+0.05,'c-')
				plt.plot(y_data_test3,y_data_test3-0.05,'c-')

				plt.subplot(2,3,4)
				plt.scatter(y_data_test4,prediction1[:,3])
				plt.plot(y_data_test4,y_data_test4,'r-')
				plt.plot(y_data_test4,y_data_test4+0.05,'c-')
				plt.plot(y_data_test4,y_data_test4-0.05,'c-')

				plt.subplot(2,3,5)
				plt.scatter(y_data_test5,prediction1[:,4])
				plt.plot(y_data_test5,y_data_test5,'r-')
				plt.plot(y_data_test5,y_data_test5+0.05,'c-')
				plt.plot(y_data_test5,y_data_test5-0.05,'c-')

				plt.subplot(2,3,6)
				plt.scatter(y_data_test6,prediction1[:,5])
				plt.plot(y_data_test6,y_data_test6,'r-')
				plt.plot(y_data_test6,y_data_test6+0.05,'c-')
				plt.plot(y_data_test6,y_data_test6-0.05,'c-')

				plt.draw(); plt.pause(0.01)
		plt.ioff
		plt.show()

def Test_func():
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	savefileid = 'cnn_save_ellipse/model'
	print(savefileid)
	saver.restore(sess, savefileid)
	prediction1 = sess.run(prediction,feed_dict={x:x_data_test})

	### Test data plot ###		
	
	y_data_test1 = y_data_test[:,0]
	y_data_test2 = y_data_test[:,1]
	y_data_test3 = y_data_test[:,2]
	y_data_test4 = y_data_test[:,3]
	y_data_test5 = y_data_test[:,4]
	y_data_test6 = y_data_test[:,5]

	plt.subplot(2,3,1)
	plt.scatter(y_data_test1,prediction1[:,0],alpha=0.1)
	plt.plot(y_data_test1,y_data_test1,'r-')
	plt.plot(y_data_test1,y_data_test1+0.05,'c-')
	plt.plot(y_data_test1,y_data_test1-0.05,'c-')

	plt.subplot(2,3,2)
	plt.scatter(y_data_test2,prediction1[:,1],alpha=0.1)
	plt.plot(y_data_test2,y_data_test2,'r-')
	plt.plot(y_data_test2,y_data_test2+0.05,'c-')
	plt.plot(y_data_test2,y_data_test2-0.05,'c-')

	plt.subplot(2,3,3)
	plt.scatter(y_data_test3,prediction1[:,2],alpha=0.1)
	plt.plot(y_data_test3,y_data_test3,'r-')
	plt.plot(y_data_test3,y_data_test3+0.05,'c-')
	plt.plot(y_data_test3,y_data_test3-0.05,'c-')

	plt.subplot(2,3,4)
	plt.scatter(y_data_test4,prediction1[:,3],alpha=0.1)
	plt.plot(y_data_test4,y_data_test4,'r-')
	plt.plot(y_data_test4,y_data_test4+0.05,'c-')
	plt.plot(y_data_test4,y_data_test4-0.05,'c-')

	plt.subplot(2,3,5)
	plt.scatter(y_data_test5,prediction1[:,4],alpha=0.1)
	plt.plot(y_data_test5,y_data_test5,'r-')
	plt.plot(y_data_test5,y_data_test5+0.05,'c-')
	plt.plot(y_data_test5,y_data_test5-0.05,'c-')

	plt.subplot(2,3,6)
	plt.scatter(y_data_test6,prediction1[:,5],alpha=0.1)
	plt.plot(y_data_test6,y_data_test6,'r-')
	plt.plot(y_data_test6,y_data_test6+0.05,'c-')
	plt.plot(y_data_test6,y_data_test6-0.05,'c-')


losses = []
# FirstOrder()  # <--- Uncomment this function to retrain the whole network.
Test_func()
plt.show()

		

