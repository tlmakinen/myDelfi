import numpy as np
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from tqdm import trange
import pickle
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

from typing import Sequence


class ConditionalMaskedAutoregressiveFlow(tf.Module):    
    """Conditional Masked Autoregressive Flow module
    Parameterize a conditional density p(x|y) using a masked autoregressive
    flow. 

    Parameters
    ----------
    n_dimensions : int
        The number of parameter dimesions; dim(x)
    n_conditionals : int
        The number of conditional dimensions; dim(y)
    n_mades : int
        The number of Masked Autoencoders for Density Estimation (MADEs) to chain together
    n_hidden : Sequence[int]
        Hidden layer architecture
    optimizer : <tf.keras.optimizers>
        Learning optimizer
    distances : utils container
        Holds accepted and rejected distances of summaries to targets and
        some summaries of results

    Methods
    -------
    bijector:
        Shift and scale function for conditional input; phi = bijector(y)
    log_prob:
        Evaluates the log probability of a particular x data, given some conditional y;
        ln p(x | y)
    sample:
        Sample x ~ p(x | y) from the conditional distribution given a conditional y
    fit:
        Train the CMAF for p(x|y)
    """
    def __init__(self, n_dimensions=None, n_conditionals=None, n_mades=1, n_hidden=[50,50], 
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            input_order="random",
            activation=keras.layers.LeakyReLU(0.01), prior=None,
            all_layers=True,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None), 
            bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None),
            kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None,
            bias_constraint=None):

        super(ConditionalMaskedAutoregressiveFlow, self).__init__('CMAF')
        # extract init parameters
        self.n_dimensions = n_dimensions
        self.n_conditionals = n_conditionals
        self.n_mades = n_mades
        self.optimizer = optimizer
        # construct the base (normal) distribution
        self.base_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_dimensions), scale_diag=tf.ones(self.n_dimensions))
        # put the conditional inputs to all layers, or just the first layer?
        if all_layers == True:
            all_layers = "all_layers"
        else:
            all_layers = "first_layer"
        # construct stack of conditional MADEs
        self.MADEs = [tfb.AutoregressiveNetwork(
                        params=2,
                        hidden_units=n_hidden,
                        activation=activation,
                        event_shape=[n_dimensions],
                        conditional=True,
                        conditional_event_shape=[n_conditionals],
                        conditional_input_layers=all_layers,
                        input_order=input_order,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        ) for i in range(n_mades)
        ]
        self.prior = prior

    # bijector for x | y (chain the conditional MADEs together)
    def bijector(self, y):
        """bijector for x | y (chain the conditional MADEs together

        Args:
            y (array_like): conditional input

        Returns:
            bijector function: bijector function to be evaluated at conditional input
        """
        # start with an empty bijector
        MAF = tfb.Identity() 
        # pass through the MADE layers (passing conditional inputs each time)
        for i in range(self.n_mades):
            MAF = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=lambda x: self.MADEs[i](x, conditional_input=y))(MAF)
        return MAF
    # construct distribution P(x | y)
    def __call__(self, y):
        return tfd.TransformedDistribution(
            self.base_distribution,
            bijector=self.bijector(y))
    # log probability ln P(x | y)
    def log_prob(self, x, y):
        """ln p(x | y)

        Args:
            x (array_like): data for which to evaluate the probability
            y (array_like): conditional input

        Returns:
            array_like: log-probability of x
        """
        return self.__call__(y).log_prob(x)

    # sample n samples from P(x | y)
    def sample(self, n, y):
        """sample n samples from p(x | y)

        Args:
            n (int): number of samples to draw
            y (array_like): conditional input y

        Returns:
            array_like: sampled values of x ~ p(x|y)
        """
        # base samples
        base_samples = self.base_distribution.sample(n)
        # biject the samples
        return self.bijector(y).forward(base_samples)


    def loss(self, x, y):
        """-ln p(x | y ) - p(y)

        Args:
            x (array_like): data for which to evaluate the probability
            y (array_like): conditional input

        Returns:
            array_like: "posterior" log-loss
        """
        if self.prior is None:
            priorprob = 0
        else:
            priorprob = self.prior.log_prob(y)
        return - self.log_prob(x,y) - priorprob



    @tf.function
    def training_step(self, x, y):
        with tf.GradientTape() as tape:

            loss = K.mean(self.loss(x, y))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    @tf.function
    def validation_step(self, x, y):
        loss = K.mean(self.loss(x, y))
        return loss
    

    def fit(self, training_variables, training_conditionals, training_weights=None, validation_split=0.1, epochs=1000, batch_size=128, patience=20, progress_bar=True, save=False, filename=None):
        """_summary_

        Args:
            training_variables (array_like): x data to train p(x|y)
            training_conditionals (array_like): y data to train p(x|y)
            training_weights (array_like, optional): training data weights. Defaults to None.
            validation_split (float, optional): _description_. Defaults to 0.1.
            epochs (int, optional): _description_. Defaults to 1000.
            batch_size (int, optional): _description_. Defaults to 128.
            patience (int, optional): _description_. Defaults to 20.
            progress_bar (bool, optional): _description_. Defaults to True.
            save (bool, optional): _description_. Defaults to False.
            filename (string, optional): _description_. Defaults to None.

        Returns:
            history (list, list): training and validation loss history 
        """
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


        # create and shuffle the dataset
        dataset = tf.data.Dataset.from_tensor_slices((training_conditionals, training_variables, training_weights)) #.shuffle(n_total)
        # training / validation split
        training_set = dataset.take(n_training).batch(batch_size).shuffle(n_total)
        validation_set = dataset.skip(n_training).batch(n_validation).shuffle(n_total)

        with trange(epochs) as t:
            for epoch in t:

                # loop over batches for a single epoch
                for conditionals, outputs, weights in training_set:

                    loss = self.training_step(outputs, conditionals)
                    t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

                val_loss = []
                # compute total loss and validation loss
                for conditionals, outputs, weights in validation_set:
                    
                    val_loss.append(self.validation_step(outputs, conditionals).numpy())

                validation_loss.append(val_loss[-1])
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