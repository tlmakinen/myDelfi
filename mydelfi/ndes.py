import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
import pickle
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers


class MixtureDensityNetwork(tf.Module):

	def __init__(self, n_dimensions=None, n_conditionals=None, n_components=3, conditional_shift=None, conditional_scale=None, output_shift=None, output_scale=None, n_hidden=[64, 64], activation=[tf.nn.leaky_relu, tf.nn.leaky_relu], optimizer=tf.keras.optimizers.Adam(), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None), restore=False, restore_filename=None, eps=1e-10):

		#super(MixtureDensityNetwork, self).__init__(reparameterization_type=None, validate_args=False, allow_nan_stats=True)

		# load parameters if restoring saved model
		if restore is True:
			n_dimensions, n_conditionals, n_components, conditional_shift, conditional_scale, output_shift, output_scale, n_hidden, activation, kernel_initializer, loaded_trainable_variables = pickle.load(open(restore_filename, 'rb'))

		# dimension of data and parameter spaces
		self.n_dimensions = n_dimensions
		self.n_conditionals = n_conditionals

		# number of mixture components and network architecture
		self.n_components = n_components

		# how many outputs do we need?
		self.n_network_outputs = self.n_dimensions * n_components + self.n_components + self.n_components * self.n_dimensions * (self.n_dimensions + 1) / 2

		# required size of output layer for a Gaussian mixture density network
		self.n_hidden = n_hidden
		self.activations = activation + [tf.identity]
		self.architecture = [self.n_conditionals] + self.n_hidden + [self.n_network_outputs]
		self.n_layers = len(self.architecture) - 1
		self.kernel_initializer = kernel_initializer

		# shift and scale
		self.conditional_shift = tf.convert_to_tensor(conditional_shift, dtype=tf.float32) if conditional_shift is not None else tf.zeros(self.n_conditionals, dtype=tf.float32)
		self.conditional_scale = tf.convert_to_tensor(conditional_scale, dtype=tf.float32) if conditional_scale is not None else tf.ones(self.n_conditionals, dtype=tf.float32)
		self.output_shift = tf.convert_to_tensor(output_shift, dtype=tf.float32) if output_shift is not None else tf.zeros(self.n_dimensions, dtype=tf.float32)
		self.output_scale = tf.convert_to_tensor(output_scale, dtype=tf.float32) if output_scale is not None else tf.ones(self.n_dimensions, dtype=tf.float32)

		# construct and initialize network model
		self.model = tfk.Sequential([tfkl.Dense(self.architecture[i+1], activation=self.activations[i], kernel_initializer=kernel_initializer) for i in range(self.n_layers)])
		_ = self.model(tf.ones((1, self.n_conditionals)))

		# optimizer
		self.optimizer = optimizer

		# load in the saved weights if restoring
		if restore is True:
			for model_variable, loaded_variable in zip(self.model.trainable_variables, loaded_trainable_variables):
				model_variable.assign(loaded_variable)

		# constants
		self.lnsqrt2pi = tf.constant(np.log(np.sqrt(2.)*np.pi), dtype=tf.float32)
		self.eps = eps

	def __call__(self, conditional):

		return self.model((conditional - self.conditional_shift)/self.conditional_scale)

	def mixture_components(self, conditional):

		# split the outputs
		mu, r, L = tf.split(self.__call__(conditional), (self.n_components * self.n_dimensions, self.n_components, int(self.n_components * self.n_dimensions * (self.n_dimensions + 1) // 2) ), axis=-1)
		
		# reshape the outputs
		mu = tf.reshape(mu, mu.shape[0:-1] + [self.n_dimensions, self.n_components])
		L = tf.reshape(L, L.shape[0:-1] + [self.n_components, int(self.n_dimensions * (self.n_dimensions + 1) // 2) ])

		# softmax the weights
		r = tf.nn.softmax(r, axis=-1)

		# construct the cholesky factors of the inverse covariances
		L = tfp.math.fill_triangular(L)
		L = L - tf.linalg.diag(tf.linalg.diag_part(L)) + tf.linalg.diag(tf.math.exp(tf.linalg.diag_part(L)) + self.eps)

		return mu, r, L

	def log_prob(self, x, conditional=None):

		# compute mixture components
		mu, r, L = self.mixture_components(conditional)
		logdetL = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=-1)

		# component log probs
		component_log_prob = -0.5*tf.reduce_sum(tf.square(tf.einsum('...kij,...jk->...ik', L, x[...,tf.newaxis] - mu)), axis=-2) + logdetL - self.n_dimensions * self.lnsqrt2pi + tf.math.log(r)

		#return tf.math.reduce_logsumexp(component_log_prob, axis=-1)
		return tf.math.log(tf.reduce_sum(tf.exp(component_log_prob), axis=-1) + 1e-37)

	def prob(self, x, conditional=None):

		return tf.exp(self.log_prob(x, conditional=conditional))

	def distribution(self, conditional):

		# compute mixture components
		mu, r, L = self.mixture_components(conditional)

		# cholesky of covariance rather than inverse
		L = tf.linalg.inv(L)

		# components distributions
		components = [tfd.MultivariateNormalTriL(loc=mu[0,:,i], scale_tril=L[0,i,:,:]) for i in range(self.n_components)]

		return tfd.Mixture(cat=tfd.Categorical(probs=r[0,:]), components=components)

	def save(self, filename):

		pickle.dump([self.n_dimensions, self.n_conditionals, self.n_components, self.conditional_shift, self.conditional_scale, self.output_shift, self.output_scale, self.n_hidden, self.activation, self.kernel_initializer] + [tuple(variable.numpy() for variable in self._network.trainable_variables)], open(filename, 'wb'))

	@tf.function
	def loss(self, x, w, conditional=None):

		return -tf.reduce_sum(tf.squeeze(w, -1) * self.log_prob(x, conditional=conditional)) / tf.reduce_sum(tf.squeeze(w, -1) )

	@tf.function
	def training_step(self, x, w, conditional=None):

		with tf.GradientTape() as tape:

			loss = self.loss(x, w, conditional=conditional)

		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return loss

	def fit(self, training_data, validation_split=0.1, epochs=1000, batch_size=128, patience=20, progress_bar=True, save=False, filename=None):
		
		# validation and training samples sizes
		n_total = int(training_data[0].shape[0])
		n_validation = int(n_total * validation_split)
		n_training = n_total - n_validation

		# set up training loss
		training_loss = [np.infty]
		validation_loss = [np.infty]
		best_loss = np.infty
		early_stopping_counter = 0

		with trange(epochs) as t:
			for epoch in t:

				# create and shuffle the dataset
				dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(n_total)

				# training / validation split
				training_set = dataset.take(n_training).batch(batch_size)
				validation_set = dataset.skip(n_training).batch(n_validation)

				# loop over batches for a single epoch
				for conditionals, outputs, weights in training_set:

					loss = self.training_step(outputs, weights, conditional=conditionals)
					t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

				# compute total loss and validation loss
				for conditionals, outputs, weights in validation_set:
					validation_loss.append(self.loss(outputs, weights, conditionals).numpy())
				training_loss.append(loss.numpy())

				# update progress bar
				t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

				# early stopping condition
				if validation_loss[-1] < best_loss:
					best_loss = validation_loss[-1]
					early_stopping_counter = 0
					if save:
						self.save(filename)
				else:
					early_stopping_counter += 1
				if early_stopping_counter >= patience:
					break
		return training_loss, validation_loss




class MSEUtilityNetworkEnsemble(tf.Module):
    
    def __init__(self, n_inputs=2, n_ensemble=10, n_hidden=[256, 256], activation=[tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.leaky_relu], lr=1e-4, kernel_initializer=tfk.initializers.RandomNormal(stddev=1e-3)):
        
        # parameters
        self.n_ensemble = n_ensemble
        self.n_inputs = n_inputs
        
        # architecture
        self.architecture = [n_inputs] + n_hidden + [1]
        self.n_layers = len(self.architecture) - 1
        self.activation = activation
        self.initializer = kernel_initializer
        
        # models and optimizers
        self.model = []
        self.optimizer = []
        for n in range(self.n_ensemble):
            
            # model
            self.model.append(tfk.Sequential([tfkl.Dense(self.architecture[i+1], activation=self.activation[i], kernel_initializer=self.initializer) for i in range(self.n_layers)]))
            _ = self.model[n](tf.ones((1, self.n_inputs))) # initialize
            
            # optimizer
            self.optimizer.append(tfk.optimizers.Adam(lr=lr))
        
        # model weights
        self.log_model_weights = np.zeros(self.n_ensemble, dtype=np.float32)
        self.model_weights = np.ones(self.n_ensemble, dtype=np.float32) / self.n_ensemble
        
    # compute model weighted expected utility over a grid
    def __call__(self, inputs):

        # construct weighted expectation
        for i in range(self.n_ensemble):
            if i == 0:
                weighted_expectation = tf.squeeze(self.model[i](inputs).numpy(), -1) * self.model_weights[i]
            else:
                weighted_expectation += tf.squeeze(self.model[i](inputs).numpy(), -1) * self.model_weights[i]
        
        return weighted_expectation
    
    # loss
    def loss(self, utility, action, summaries, weights, i):
        
        return tf.reduce_sum(weights * (utility - self.model[i](tf.concat([action, summaries], axis=-1)))**2 ) / tf.reduce_sum(weights)
    
    # training step
    def training_step(self, utility, action, summaries, weights, i):
        
        with tf.GradientTape() as tape:
            tape.watch(self.model[i].trainable_variables)
            
            loss = self.loss(utility, action, summaries, weights, i)
            
        gradients = tape.gradient(loss, self.model[i].trainable_variables)
        self.optimizer[i].apply_gradients(zip(gradients, self.model[i].trainable_variables))
        
        return loss
    
    # train all members of the ensemble
    def train(self, training_data, epochs=1000, batch_size=128, validation_split=0.2, patience=20):
        
        # validation and training samples sizes
        n_total = int(training_data[0].shape[0])
        n_validation = int(n_total * validation_split)
        n_training = n_total - n_validation
                
        # loop over ensemble members
        for n in range(self.n_ensemble):

            # set up training loss
            training_loss = [np.infty]
            validation_loss = [np.infty]
            best_loss = np.infty
            early_stopping_counter = 0
            
            # loop over epochs
            with trange(epochs) as t:
                
                for epoch in t:

                    # create and shuffle the dataset
                    dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(n_total)

                    # training / validation split
                    training_set = dataset.take(n_training).batch(batch_size)
                    validation_set = dataset.skip(n_training).batch(n_validation)

                    # loop over batches for a single epoch
                    for utility, action, summaries, weights in training_set:

                        loss = self.training_step(utility, action, summaries, weights, n)
                        t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)
                            
                    # compute total loss and validation loss
                    for utility, action, summaries, weights in validation_set:
                        validation_loss.append(self.loss(utility, action, summaries, weights, n).numpy())
                    training_loss.append(loss.numpy())

                    # update progress bar
                    t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

                    # if we get a nan or inf loss, break the loop, re-initialize the network, and set it's log model weight to -ve large number
                    if np.isnan(loss.numpy()) or np.isinf(loss.numpy()):

                        # re-initialize network weights
                        for variable in self.model[n].trainable_variables:
                            variable.assign(self.initializer(variable.shape))
                            
                        # set the log model weight to - large number
                        self.log_model_weights[n] = -1e32
                            
                        # break the loop
                        break
                    
                    # set the log model weight
                    self.log_model_weights[n] = -validation_loss[-1]
                        
                    # early stopping condition
                    if validation_loss[-1] < best_loss:
                        best_loss = validation_loss[-1]
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        break
                        
        # sort out the model weights      
        self.model_weights = np.exp(self.log_model_weights - max(self.log_model_weights))
        self.model_weights = self.model_weights / sum(self.model_weights)


class GaussianUtilityNetworkEnsemble(tf.Module):
    
    def __init__(self, n_inputs=2, n_ensemble=10, n_hidden=[256, 256], activation=[tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.leaky_relu], lr=1e-4, kernel_initializer=tfk.initializers.RandomNormal(stddev=1e-3)):
        
        # parameters
        self.n_ensemble = n_ensemble
        self.n_inputs = n_inputs
        
        # architecture
        self.architecture = [n_inputs] + n_hidden + [2]
        self.n_layers = len(self.architecture) - 1
        self.activation = activation
        self.initializer = kernel_initializer
        
        # models and optimizers
        self.model = []
        self.optimizer = []
        for n in range(self.n_ensemble):
            
            # model
            self.model.append(tfk.Sequential([tfkl.Dense(self.architecture[i+1], activation=self.activation[i], kernel_initializer=self.initializer) for i in range(self.n_layers)]))
            _ = self.model[n](tf.ones((1, self.n_inputs))) # initialize
            
            # optimizer
            self.optimizer.append(tfk.optimizers.Adam(lr=lr))
            
        # constants
        self.lnsqrt2pi = tf.constant(np.log(np.sqrt(2*np.pi)), dtype=tf.float32)
        
        # model weights
        self.log_model_weights = np.zeros(self.n_ensemble, dtype=np.float32)
        self.model_weights = np.ones(self.n_ensemble, dtype=np.float32) / self.n_ensemble
        
    # compute model weighted expected utility over a grid
    def __call__(self, inputs):

        # construct weighted expectation
        for i in range(self.n_ensemble):
            if i == 0:
                weighted_expectation = self.expectation(inputs, i).numpy() * self.model_weights[i]
            else:
                weighted_expectation += self.expectation(inputs, i).numpy() * self.model_weights[i]
        
        return weighted_expectation

    # expected utility (per model)
    def expectation(self, inputs, i):

       # split network outputs and softmax the component weights
       mu, logsigma = tf.split(self.model[i](inputs), (1, 1), axis=-1)

       # expected utility
       return tf.squeeze(mu, axis=-1)
        
    # log prob for a given model (i)
    def log_prob(self, utility, action, summaries, i):
        
        mu, logsigma = tf.split(self.model[i](tf.concat([action, summaries], axis=-1)), (1,1), axis=-1)
        logsigma = tf.clip_by_value(logsigma, clip_value_min=-10, clip_value_max=10)

        return tf.squeeze(-0.5*(mu - utility)**2 / tf.exp(2*logsigma) - logsigma - self.lnsqrt2pi, -1)
    
    # loss
    def loss(self, utility, action, summaries, weights, i):
        
        return -tf.reduce_sum(tf.squeeze(weights, -1) * self.log_prob(utility, action, summaries, i)) / tf.reduce_sum(tf.squeeze(weights, -1))
    
    # training step
    def training_step(self, utility, action, summaries, weights, i):
        
        with tf.GradientTape() as tape:
            tape.watch(self.model[i].trainable_variables)
            
            loss = self.loss(utility, action, summaries, weights, i)
            
        gradients = tape.gradient(loss, self.model[i].trainable_variables)
        self.optimizer[i].apply_gradients(zip(gradients, self.model[i].trainable_variables))
        
        return loss
    
    # train all members of the ensemble
    def train(self, training_data, epochs=1000, batch_size=128, validation_split=0.2, patience=20):
        
        # validation and training samples sizes
        n_total = int(training_data[0].shape[0])
        n_validation = int(n_total * validation_split)
        n_training = n_total - n_validation
                
        # loop over ensemble members
        for n in range(self.n_ensemble):

            # set up training loss
            training_loss = [np.infty]
            validation_loss = [np.infty]
            best_loss = np.infty
            early_stopping_counter = 0
            
            # loop over epochs
            with trange(epochs) as t:
                
                for epoch in t:

                    # create and shuffle the dataset
                    dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(n_total)

                    # training / validation split
                    training_set = dataset.take(n_training).batch(batch_size)
                    validation_set = dataset.skip(n_training).batch(n_validation)

                    # loop over batches for a single epoch
                    for utility, action, summaries, weights in training_set:

                        loss = self.training_step(utility, action, summaries, weights, n)
                        t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)
                            
                    # compute total loss and validation loss
                    for utility, action, summaries, weights in validation_set:
                        validation_loss.append(self.loss(utility, action, summaries, weights, n).numpy())
                    training_loss.append(loss.numpy())

                    # update progress bar
                    t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

                    # if we get a nan or inf loss, break the loop, re-initialize the network, and set it's log model weight to -ve large number
                    if np.isnan(loss.numpy()) or np.isinf(loss.numpy()):

                        # re-initialize network weights
                        for variable in self.model[n].trainable_variables:
                            variable.assign(self.initializer(variable.shape))
                            
                        # set the log model weight to - large number
                        self.log_model_weights[n] = -1e32
                            
                        # break the loop
                        break
                    
                    # set the log model weight
                    self.log_model_weights[n] = -validation_loss[-1]
                        
                    # early stopping condition
                    if validation_loss[-1] < best_loss:
                        best_loss = validation_loss[-1]
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        break
                        
        # sort out the model weights      
        self.model_weights = np.exp(self.log_model_weights - max(self.log_model_weights))
        self.model_weights = self.model_weights / sum(self.model_weights)

class GaussianMixtureUtilityNetworkEnsemble(tf.Module):
    
    def __init__(self, n_inputs=2, n_components=2, n_ensemble=10, n_hidden=[256, 256], activation=[tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.leaky_relu], lr=1e-4, kernel_initializer=tfk.initializers.RandomNormal(stddev=1e-3)):
        
        # parameters
        self.n_ensemble = n_ensemble
        self.n_inputs = n_inputs
        
        # architecture
        self.n_components = n_components
        self.architecture = [n_inputs] + n_hidden + [n_components * 3]
        self.n_layers = len(self.architecture) - 1
        self.activation = activation
        self.initializer = kernel_initializer
        
        # models and optimizers
        self.model = []
        self.optimizer = []
        for n in range(self.n_ensemble):
            
            # model
            self.model.append(tfk.Sequential([tfkl.Dense(self.architecture[i+1], activation=self.activation[i], kernel_initializer=self.initializer) for i in range(self.n_layers)]))
            _ = self.model[n](tf.ones((1, self.n_inputs))) # initialize
            
            # optimizer
            self.optimizer.append(tfk.optimizers.Adam(lr=lr))
            
        # constants
        self.lnsqrt2pi = tf.constant(np.log(np.sqrt(2*np.pi)), dtype=tf.float32)
        
        # model weights
        self.log_model_weights = np.zeros(self.n_ensemble, dtype=np.float32)
        self.model_weights = np.ones(self.n_ensemble, dtype=np.float32) / self.n_ensemble
        
    # compute model weighted expected utility over a grid
    def __call__(self, inputs):

        # construct weighted expectation
        for i in range(self.n_ensemble):
            if i == 0:
                weighted_expectation = self.expectation(inputs, i).numpy() * self.model_weights[i]
            else:
                weighted_expectation += self.expectation(inputs, i).numpy() * self.model_weights[i]
        
        return weighted_expectation

    # expected utility (per model)
    def expectation(self, inputs, i):

       # split network outputs and softmax the component weights
       r, mu, logsigma = tf.split(self.model[i](inputs), (self.n_components, self.n_components, self.n_components), axis=-1)
       r = tf.nn.softmax(r, axis=-1)

       # expected utility
       return tf.reduce_sum(mu * r, axis=-1)
        
    # log prob for a given model (i)
    def log_prob(self, utility, action, summaries, i):
        
        # split network outputs
        r, mu, logsigma = tf.split(self.model[i](tf.concat([action, summaries], axis=-1)), (self.n_components, self.n_components, self.n_components), axis=-1)

        # clip the logsigma
        logsigma = tf.clip_by_value(logsigma, clip_value_min=-10, clip_value_max=10)

        # softmax the component weights
        r = tf.nn.softmax(r, axis=-1)

        # component log probs (with weights)
        component_log_probs = -0.5*(mu - utility)**2 / tf.exp(2*logsigma) - logsigma - tf.math.log(r) - self.lnsqrt2pi

        #return tf.math.log(tf.reduce_sum(tf.exp(component_log_probs), axis=-1) + 1e-37)
        return tf.reduce_logsumexp(tf.exp(component_log_probs), axis=-1)

    
    # loss
    def loss(self, utility, action, summaries, weights, i):
        
        return -tf.reduce_sum(tf.squeeze(weights, -1) * self.log_prob(utility, action, summaries, i)) / tf.reduce_sum(tf.squeeze(weights, -1))
    
    # training step
    def training_step(self, utility, action, summaries, weights, i):
        
        with tf.GradientTape() as tape:
            tape.watch(self.model[i].trainable_variables)
            
            loss = self.loss(utility, action, summaries, weights, i)
            
        gradients = tape.gradient(loss, self.model[i].trainable_variables)
        self.optimizer[i].apply_gradients(zip(gradients, self.model[i].trainable_variables))
        
        return loss
    
    # train all members of the ensemble
    def train(self, training_data, epochs=1000, batch_size=128, validation_split=0.2, patience=20):
        
        # validation and training samples sizes
        n_total = int(training_data[0].shape[0])
        n_validation = int(n_total * validation_split)
        n_training = n_total - n_validation
                
        # loop over ensemble members
        for n in range(self.n_ensemble):

            # set up training loss
            training_loss = [np.infty]
            validation_loss = [np.infty]
            best_loss = np.infty
            early_stopping_counter = 0
            
            # loop over epochs
            with trange(epochs) as t:
                
                for epoch in t:

                    # create and shuffle the dataset
                    dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(n_total)

                    # training / validation split
                    training_set = dataset.take(n_training).batch(batch_size)
                    validation_set = dataset.skip(n_training).batch(n_validation)

                    # loop over batches for a single epoch
                    for utility, action, summaries, weights in training_set:

                        loss = self.training_step(utility, action, summaries, weights, n)
                        t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)
                            
                    # compute total loss and validation loss
                    for utility, action, summaries, weights in validation_set:
                        validation_loss.append(self.loss(utility, action, summaries, weights, n).numpy())
                    training_loss.append(loss.numpy())

                    # update progress bar
                    t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

                    # if we get a nan or inf loss, break the loop, re-initialize the network, and set it's log model weight to -ve large number
                    if np.isnan(loss.numpy()) or np.isinf(loss.numpy()):

                        # re-initialize network weights
                        for variable in self.model[n].trainable_variables:
                            variable.assign(self.initializer(variable.shape))
                            
                        # set the log model weight to - large number
                        self.log_model_weights[n] = -1e32
                            
                        # break the loop
                        break
                    
                    # set the log model weight
                    self.log_model_weights[n] = -validation_loss[-1]
                        
                    # early stopping condition
                    if validation_loss[-1] < best_loss:
                        best_loss = validation_loss[-1]
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        break
                        
        # sort out the model weights      
        self.model_weights = np.exp(self.log_model_weights - max(self.log_model_weights))
        self.model_weights = self.model_weights / sum(self.model_weights)