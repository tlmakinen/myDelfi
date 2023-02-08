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


class MixtureDensityNetwork2(tf.Module):

    def __init__(self, n_dimensions=None, n_conditionals=None, n_components=3, conditional_shift=None, conditional_scale=None, 
                 output_shift=None, output_scale=None, n_hidden=[64, 64], activation=[tf.nn.leaky_relu, tf.nn.leaky_relu], 
                 optimizer=tf.keras.optimizers.Adam(lr=1e-4), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None), restore=False, restore_filename=None, eps=1e-10):
        """

        Args:
            n_dimensions (_type_, optional): _description_. Defaults to None.
            n_conditionals (_type_, optional): _description_. Defaults to None.
            n_components (int, optional): _description_. Defaults to 3.
            conditional_shift (_type_, optional): _description_. Defaults to None.
            conditional_scale (_type_, optional): _description_. Defaults to None.
            output_shift (_type_, optional): _description_. Defaults to None.
            output_scale (_type_, optional): _description_. Defaults to None.
            n_hidden (list, optional): _description_. Defaults to [64, 64].
            activation (list, optional): _description_. Defaults to [tf.nn.leaky_relu, tf.nn.leaky_relu].
            optimizer (_type_, optional): _description_. Defaults to tf.keras.optimizers.Adam(lr=1e-4).
            kernel_initializer (_type_, optional): _description_. Defaults to tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None).
            restore (bool, optional): _description_. Defaults to False.
            restore_filename (_type_, optional): _description_. Defaults to None.
            eps (_type_, optional): _description_. Defaults to 1e-10.
        """

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
        self.activation = activation + [tf.identity]
        self.architecture = [self.n_conditionals] + self.n_hidden + [self.n_network_outputs]
        self.n_layers = len(self.architecture) - 1
        self.kernel_initializer = kernel_initializer

        # shift and scale
        self.conditional_shift = tf.convert_to_tensor(conditional_shift, dtype=tf.float32) if conditional_shift is not None else tf.zeros(self.n_conditionals, dtype=tf.float32)
        self.conditional_scale = tf.convert_to_tensor(conditional_scale, dtype=tf.float32) if conditional_scale is not None else tf.ones(self.n_conditionals, dtype=tf.float32)
        self.output_shift = tf.convert_to_tensor(output_shift, dtype=tf.float32) if output_shift is not None else tf.zeros(self.n_dimensions, dtype=tf.float32)
        self.output_scale = tf.convert_to_tensor(output_scale, dtype=tf.float32) if output_scale is not None else tf.ones(self.n_dimensions, dtype=tf.float32)

        # construct and initialize network model
        self.model = tfk.Sequential([tfkl.Dense(self.architecture[i+1], activation=self.activation[i], kernel_initializer=kernel_initializer) for i in range(self.n_layers)])
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
        """_summary_

        Args:
            conditional (_type_): _description_

        Returns:
            _type_: _description_
        """

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
        """_summary_

        Args:
            x (_type_): _description_
            conditional (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        # compute mixture components
        mu, r, L = self.mixture_components(conditional)
        logdetL = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=-1)

        # shift and scale the variables
        x_ = (x - self.output_shift) / self.output_scale

        # component log probs
        component_log_prob = -0.5*tf.reduce_sum(tf.square(tf.einsum('...kij,...jk->...ik', L, x_[...,tf.newaxis] - mu)), axis=-2) + logdetL - self.n_dimensions * self.lnsqrt2pi + tf.math.log(r)

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

        pickle.dump([self.n_dimensions, self.n_conditionals, self.n_components, self.conditional_shift, self.conditional_scale, \
                     self.output_shift, self.output_scale, self.n_hidden, \
                     self.activation, self.kernel_initializer] + [tuple(variable.numpy() for variable in self.model.trainable_variables)], open(filename, 'wb'))

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

    def fit(self, training_variables=None, training_conditionals=None, training_weights=None, validation_split=0.1, epochs=1000, batch_size=128, patience=20, progress_bar=True, save=False, filename=None):
        
        # validation and training samples sizes
        n_total = int(training_variables.shape[0])
        n_validation = int(n_total * validation_split)
        n_training = n_total - n_validation

        # set up training loss
        training_loss = [np.infty]
        validation_loss = [np.infty]
        best_loss = np.infty
        early_stopping_counter = 0

        # set weights to one if not otherwise set
        if training_weights is None:
            training_weights = tf.ones((training_variables.shape[0], 1), dtype=tf.float32)

        with trange(epochs) as t:
            for epoch in t:

                # create and shuffle the dataset
                dataset = tf.data.Dataset.from_tensor_slices((training_conditionals, training_variables, training_weights)).shuffle(n_total)

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