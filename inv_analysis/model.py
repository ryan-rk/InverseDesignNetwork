from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
import matplotlib.pyplot as plt

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=64, input_width=64, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.data_dir = data_dir

    self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def mconv2d(self, x, w1, w2, win, wout, wx, wy):
    with tf.name_scope('mat_pred'):
      # stride [1, x_movement, y_movement, 1]
      # Must have strides[0] = strides[3] = 1
        w = tf.Variable(tf.truncated_normal([w1,w2,win,wout]))
        b = tf.Variable(tf.zeros([wout]))
        return tf.nn.relu((tf.nn.conv2d(x, w, strides=[1,wx,wy,1], padding='SAME'))+b)
   
  def add_layer(self, L_Prev, num_nodes_LPrev, num_nodes_LX, activation_LX):
    with tf.name_scope('mat_pred'):
      Weights_LX = tf.Variable(tf.random_normal([num_nodes_LPrev,num_nodes_LX]),name = 'w')
      biases_LX = tf.Variable(tf.zeros([1,num_nodes_LX]),name = 'b')
      xW_plus_b_LX = tf.matmul(L_Prev,Weights_LX)+biases_LX
      if activation_LX is None:
        LX = xW_plus_b_LX
      else:
        LX = tf.add(xW_plus_b_LX,activation_LX(xW_plus_b_LX))
      return LX

  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)
    
    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    # self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    global_vars = tf.global_variables()

    self.saver = tf.train.Saver([var for var in global_vars if 'd_' in var.name or 'g_' in var.name or 'batch_norm' in var.name])


#______________________________________CNN Material Prediction Part_______________________________________

    self.opt_target = tf.get_variable(
      'optimize_target', [1,self.z_dim],initializer=tf.random_uniform_initializer(minval=-1,maxval=1))
    self.opt_sampler_pre = self.sampler(self.opt_target)
    self.opt_sampler = 1/2*(self.opt_sampler_pre+1)
    self.matprop = tf.placeholder(tf.float32, [None, 6])

    c1 = self.mconv2d(self.opt_sampler,3,3,1,16,2,2)   #64x64x1 --> 32x32x16
    c2 = self.mconv2d(c1,3,3,16,32,2,2) #32x32x16 --> 16x16x32
    c3 = self.mconv2d(c2,3,3,32,32,2,2) #16x16x32 --> 8x8x32
    c4 = self.mconv2d(c3,3,3,32,64,2,2) #8x8x32 --> 4x4x64
    c5 = self.mconv2d(c4,3,3,64,128,2,2) #4x4x64 --> 2x2x128

    flat1 = tf.reshape(c5, [-1, 4*128])
    L1 = self.add_layer(flat1,4*128,128,tf.nn.tanh)
    L2 = self.add_layer(L1,128,32,tf.nn.tanh)
    self.prediction = self.add_layer(L2,32,6,None)
    self.matprop_loss = tf.reduce_mean(tf.square(self.matprop-self.prediction))
    self.saver_matprop = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mat_pred'))

  def train(self, config):

    bounds = []
    for i in range(25):
      bounds.append([-1,1])
    matprop_2optim = tf.contrib.opt.ScipyOptimizerInterface(self.matprop_loss,var_list=[var for var in tf.global_variables() if 'optimize_target' in var.name],
        var_to_bounds={self.opt_target:(-1,1)},options={'maxiter': 600})

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    counter = 1
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    self.saver_matprop.restore(self.sess, "checkpoint/cnn_save_ellipse/model")

    dataset_norm_max = np.array([[8.81035523e10,3.71412684e10,5.67242560e09,8.82832606e10,5.67249178e09,2.52859939e10]])   # <--- Normalization for ellipse dataset
    dataset_norm_min = np.array([[1.69938962e10,3.68522556e09,-5.91331492e09,1.34373105e10,-5.60411153e09,2.75239281e09]])
    
    # dataset_norm_max = np.array([[6.96788743e10,2.66089327e10,4.31007681e09,6.99159122e10,5.25037685e09,1.93591978e10]])   # <--- Normalization for circle square dataset
    # dataset_norm_min = np.array([[1.62600719e10,3.23096317e09,-5.86459610e09,1.37196507e10,-4.80139401e09,1.26109822e09]])

    dataset1_0 = np.load('ellipse_compliance.npy') #Change this to the desired test dataset (currently used ellipse dataset)
    dataset2 = np.zeros((dataset1_0.shape[0],dataset1_0.shape[1]))
    for i in range(dataset1_0.shape[1]):
      dataset2[:,i] = (dataset1_0[:,i]-dataset_norm_min[0,i]/(dataset_norm_max[0,i]-dataset_norm_min[0,i]) #Normalize the dataset using pretrained values
    

#_____________ Training Process _______________________________
    num_matprop = 64
    num_topo = 10
    output_arr = np.zeros((num_matprop,4096))
    output_acc = np.zeros((num_matprop,1))
    output_lat = np.zeros((num_matprop,self.z_dim))
    output_discri = np.zeros((num_matprop*num_topo,1))
    output_bestdiscri = np.zeros((num_matprop,1))
    for iii in range(num_matprop):
        best_loss = 1e20
        bestmatloss = 0
        matprop = dataset2[iii].reshape(1,6)
        for jjj in range(num_topo):
            start_time = time.time()
            self.sess.run(tf.assign(self.opt_target, tf.random_uniform([1,self.z_dim],minval=-1,maxval=1)))
            matprop_2optim.minimize(self.sess, feed_dict={self.matprop: matprop})
            end_time = time.time()
            curr_matploss, curr_reconimg, lat_var = self.sess.run([self.matprop_loss,self.opt_sampler,self.opt_target],feed_dict={self.matprop: matprop})
            discri_loss = self.sess.run(self.D_,feed_dict={self.z: lat_var})
            print(iii,' of ',jjj,' matprop_loss',curr_matploss,' maxval:',np.amax(lat_var),' minval:',np.amin(lat_var),' discri_loss:',discri_loss,' time:', (end_time-start_time))
            output_discri[[iii*num_topo+jjj]] = discri_loss
            if (1e11*curr_matploss)+(1-discri_loss) < best_loss:
                best_loss = (1e11*curr_matploss)+(1-discri_loss)
                output_arr_tmp = curr_reconimg.reshape(1,4096)
                output_arr[iii,:] = output_arr_tmp
                output_lat[iii,:] = lat_var
                output_bestdiscri[iii] = discri_loss
                bestmatloss = curr_matploss
                
        output_acc[iii] = bestmatloss


    np.savetxt('./test_results/test0.txt',output_arr,delimiter=',')
    np.save('./test_results/bestloss.npy',output_acc)
    np.save('./test_results/latvar.npy',output_lat)
    np.save('./test_results/discriloss.npy',output_discri)
    np.save('./test_results/bestdiscri.npy',output_bestdiscri)


  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'), train=False))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'), train=False))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv'), train=False))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        
        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')
        
        return tf.nn.sigmoid(h3), h3

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0, train=False))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1, train=False))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def sampler(self, z=None, y=None):
    # self.zz = tf.cast(tf.Variable(np.random.uniform(-1, 1, size=(1 , self.z_dim)),name='optimize_target'),dtype='float32')
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      # z = tf.Variable(tf.random_normal([1,self.z_dim]), name = 'opt_tar')

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  @property
  def model_dir(self):
    return "{}_{}_{}".format(
        self.dataset_name,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
