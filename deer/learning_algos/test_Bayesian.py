import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, merge, Activation, Conv2D, MaxPooling2D, Reshape, Permute

from keras import backend as K
from keras import optimizers, activations, initializers, regularizers, constraints
from keras.engine import InputSpec, Layer
from keras.regularizers import Regularizer,l1

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

class VariationalRegularizer(Regularizer):
    """Regularization layer for BayesianDense Layer Weights"""
    def __init__(self, weight=1e-3):
        Regularizer.__init__(self)
        self.weight=weight
        self.uses_learning_phase=True

    def __call__(self, loss):
        reg = - 0.5 * K.mean(1 + self.p - K.exp(self.p), axis=None)
        return K.in_train_phase(loss + self.weight*reg, loss)

    def get_config(self):
        config = {"weight":float(self.weight)}
        base_config = super(VariationalRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def init_uniform(shape, lb, ub, name=None):
    return K.random_uniform_variable(shape, lb, ub, name=name)

def l2_regu(weight_matrix):
    l2_strength=K.variable(1.,dtype='float32')
    weight_matrix=weight_matrix
    return l2_strength * K.sum(K.square(weight_matrix))
        
        
class DenseBayesian(Layer):
    """Modification to the regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

#    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseBayesian, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_mu = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel_mu',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.W_log_sigma = self.add_weight(shape=(input_dim, self.units),
                                      initializer=initializers.RandomNormal(mean=0.0, stddev=1., seed=None),
                                      name='kernel_sigma',
                                      regularizer=l2_regu,#self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.b_mu = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias_mu',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.b_log_sigma = self.add_weight(shape=(self.units,),
                                        initializer=initializers.RandomNormal(mean=0.0, stddev=1., seed=None),
                                        name='bias_sigma',
                                        regularizer=l2_regu,#self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        print "inputs.shape"
        print K.int_shape(inputs)
        print "compute_output_shape"
        print self.compute_output_shape(inputs.shape)
        e = K.random_normal((K.int_shape(inputs)[1],self.compute_output_shape(inputs.shape)[1]))#((x.shape[0], self.input_dim, self.output_dim))
        w = self.W_mu+0.1*e*K.log(1+K.exp(self.W_log_sigma/2))
        output = K.dot(inputs, w)
        ##output = K.dot(inputs, self.kernel)
        if self.use_bias:
            eb = K.random_normal((self.compute_output_shape(inputs.shape)[1],))#x.shape[0], self.output_dim))
            b = self.b_mu+0.1*eb*K.log(1+K.exp(self.b_log_sigma))
            output = K.bias_add(output, b)
            ##output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



inp = Input(shape=(1,))

x = DenseBayesian(20, activation='relu')(inp)
x = DenseBayesian(20, activation='relu')(x)

#out = BayesianDense(self._Natoms, activation='softmax',
#    W_sigma_regularizer=VariationalRegularizer(1e-3),
#    b_sigma_regularizer=VariationalRegularizer(1e-3),
#    W_regularizer=l1(0.01)) (x) 
out = DenseBayesian(1)(x)
#out = Dense(self._Natoms, activation='softmax')(x)
      
model = Model(input=inp, output=out)

rmsprop=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='mean_squared_error',
              metrics=['accuracy'])
              
data=3.14*np.arange(-1,0.5,0.1)
data=np.append(data,3.14)
labels=np.sin(data)
print data, labels
for i in range(10):
    model.fit(data, labels, epochs=1000, batch_size=32)  # starts training

    plt.plot(3.14*np.arange(-1,0.5,0.1),np.sin(3.14*np.arange(-1,0.5,0.1)),'--',c='b')
    plt.plot(3.14*1,np.sin(3.14*1),'x',c='b')
    for j in range (10):
        estimated_labels=model.predict(3.14*np.arange(-1,2,0.1))
        plt.plot(3.14*np.arange(-1,2,0.1),estimated_labels)
    plt.ylabel('some numbers')
    plt.show()



