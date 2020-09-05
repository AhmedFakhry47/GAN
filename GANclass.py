'''
Cycle GAN implementation
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class cGAN:
	def __init__(self,parameters):
		assert isinstance(parameters,dict),"Please Insert a dictionary"
		#Parameters of Generator -- Width and Height of the window
		self.n_HieghtGenerator = parameters['HieghtGenerator'] 		
		self.n_WidthGenerator  = parameters['WidthGenerator']

		#Parameters of Discriminator
		self.n_HieghtDiscrimi  = parameters['HeightDiscrimi']
		self.n_WidthDiscrimi   = parameters['WidthDiscrimi'] 

		self.batch_size 	   = parameters['BatchSize']
		self.lr 		       = parameters['LearningRate']

		#place holders for data
		##Gen Window
		self.Gen_placeholder = tf.placeholder(tf.float32,[None,self.n_HieghtGenerator,self.n_WidthGenerator,1])

		##Dis window 
		self.Dis_placeholder = tf.placeholder(tf.float32,[None,self.n_HieghtDiscrimi,self.n_WidthDiscrimi,1])

		self.model()
		
	def _log(self,x):
	    return tf.log(x + 1e-8)

	def generator(self,inp):
		with tf.name_scope('generator'):
			self.GW1 = tf.get_variable("GW1", [33, 33,1, 5], initializer=tf.contrib.layers.xavier_initializer(seed=0))
			self.GW2 = tf.get_variable("GW2", [5, 5, 5, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))

			conv_gst = tf.nn.conv2d(inp,self.GW1,strides=[1,1,1,1],padding='VALID',name='Conv_generator_st')
			gact_oust= tf.nn.relu(conv_gst,name='Activiation_gen_st')

			conv_gnd = tf.nn.conv2d(gact_oust,self.GW2,strides=[1,1,1,1],padding='VALID')
			self.generat_out =  tf.nn.sigmoid(conv_gnd)

		'''
		Descriminator is a stack of two convolution layers
		'''

	def descriminator(self,inp):
		with tf.name_scope('descriminator'):
			self.DW1 = tf.get_variable("DW1",[4,4,1,8],initializer = tf.contrib.layers.xavier_initializer(seed=0))
			self.DW2 = tf.get_variable("DW2",[2,2,8,16],initializer= tf.contrib.layers.xavier_initializer(seed=0))

			#Descriminator Conv
			conv_st     = tf.nn.conv2d(inp,DW1,strides=[1,1,1,1],padding='VALID',name='Conv_descrimi_st')
			act_out_st  = tf.nn.relu(conv_st,name='Activation_descrimi_st')
			pooling_st  = tf.nn.max_pool(act_out_st,ksize=[1,8,8,1], strides = [1, 8, 8, 1], padding='SAME',name='Pooling_Output_descrimi_st')

			conv_nd     = tf.nn.conv2d(pooling_st,DW2,strides=[1,1,1,1],padding='VALID',name='Conv_descrimi_nd')
			act_out_nd  = tf.nn.relu(conv_nd,name='Activation_descrimi_nd')
			pooling_nd  = tf.nn.max_pool(act_out_nd,ksize=[1,4,4,1], strides = [1, 4, 4, 1], padding='SAME',name='Pooling_Output_descrimi_nd')

			self.discrimi_out = tf.contrib.layers.flatten(pooling_nd)
			self.discrimi_out = tf.contrib.layers.fully_connected(self.discrimi_out)
			self.discrimi_out = tf.nn.sigmoid(self.discrimi_out)

	def model(self):
		self.pred = self.generator(self.Gen_placeholder)
		self.real = self.descriminator(self.Dis_placeholder)
		self.fake = self.descriminator(self.pred)

		with tf.name_scope('loss'):
			self.descrimi_loss = - tf.reduce_mean(self.log(self.real) +  self.log(1. - self.fake))
			self.gen_loss 	   = - tf.reduce_mean(self.log(self.fake))

	
