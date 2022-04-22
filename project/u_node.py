import tensorflow as tf

from typing import Any
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, concatenate
from tfdiffeq.models.dense_odenet import ODEBlock
from tfdiffeq.models.conv_odenet import Conv2dTime

MAX_NUM_STEPS = 1000 # Maximum number of steps for ODE solver

def get_non_linearity(name: str)->tf.keras.layers.Layer:
    """Resolve non-linearity to layer

    Arguments:
        name {str} -- name of non-linearity/activation function

    Returns:
        tf.keras.layers.Layer -- target activation layer
    """
    if name == 'relu':
        return tf.keras.layers.ReLU()
    elif name == 'softplus':
        return tf.keras.layers.Activation('softplus')
    elif name == 'lrelu':
        return tf.keras.layers.LeakyReLU(alpha=0.2)
    elif name == 'swish':
        return tf.keras.layers.Activation('swish')
    else:
        return tf.keras.layers.Activation(name)    

class Conv2dODEFunc(Model):

    def __init__(self, num_filters, augment_dim=0,
                 time_dependent=False, non_linearity='relu', **kwargs):
        """
        Convolutional block modeling the derivative of ODE system.
        # Arguments:
            num_filters : int
                Number of convolutional filters.
            augment_dim: int
                Number of augmentation channels to add. If 0 does not augment ODE.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            non_linearity : string
                Activation function for activation layer. One of 'relu', 'softplus', 
                'lrelu' and 'swish'
        """
        dynamic = kwargs.pop('dynamic', True)
        super(Conv2dODEFunc, self).__init__(**kwargs, dynamic=dynamic)

        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.num_filters = num_filters
        self.input_dim = input

        if time_dependent:
            self.norm1 = BatchNormalization()
            self.conv1 = Conv2dTime(self.num_filters, kernel_size=3, stride=1, padding=0)
            self.norm2 = BatchNormalization()
            self.conv2 = Conv2dTime(self.num_filters, kernel_size=3, stride=1, padding=1)

        else:
            self.norm1 = BatchNormalization()
            self.conv1 = tf.keras.layers.Conv2D(self.num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')
            self.norm2 = BatchNormalization()
            self.conv2 = tf.keras.layers.Conv2D(self.num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')

        self.non_linearity = get_non_linearity(non_linearity)
            
    def build(self, input_shape):
        if input_shape:
            self.built = True
        

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        """
        Parameters
        ----------
        t : Tensor
            Current time.
        x : Tensor
            Shape (batch_size, input_dim)
        """

        self.nfe += 1

        if self.time_dependent:
            out = self.norm1(x)
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
        else:
            out = self.norm1(x)
            out = self.conv1(out)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(out)
            out = self.non_linearity(out)

        return out
class UNode(Model):
    """Creates a U-Net with an ODEBlock and a convolutional ODEFunc, therefore a U-Node
    Parameters
    ----------
    num_filters : int
        Number of convolutional filters.
    input_dim : tuple of ints
        Tuple of (height, width, channels).
    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.
    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        Activation function for activation layer. One of 'relu', 'softplus', 
        'lrelu' and 'swish'
    tol : float
        Error tolerance.
    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    solver: ODE solver. Defaults to DOPRI5.
    """
    def __init__(self, num_filters, input_dim, output_dim=1,
                 augment_dim=0, time_dependent=False, non_linearity='relu', out_strides=(1, 1),
                 tol=1e-3, adjoint=False, solver='dopri5', **kwargs):

        dynamic = kwargs.pop('dynamic', True)
        super(UNode, self).__init__(**kwargs, dynamic=dynamic)

        self.nf = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.tol = tol
        self.solver = solver
        self.output_strides = out_strides
        self.input_dim = input_dim
        self.non_linearity = get_non_linearity(non_linearity)

        self.input_layer = Conv2D(filters=num_filters, kernel_size=(1, 1), padding='same', input_shape=input_dim)
        self.norm_range = tf.keras.layers.Lambda(lambda x: x / 255)

        #Contraction path
        ode_down1 = Conv2dODEFunc(num_filters=num_filters, augment_dim=augment_dim, time_dependent=time_dependent,
                                    non_linearity=non_linearity)
        self.odeblock_down1 = ODEBlock(odefunc=ode_down1, is_conv=True, tol=tol, adjoint=adjoint, solver=solver)
        self.conv_down1_2 = Conv2D(filters=num_filters*2, kernel_size=(1, 1), padding='same')
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))

        ode_down2 = Conv2dODEFunc(num_filters=num_filters*2, augment_dim=augment_dim, time_dependent=time_dependent,
                                    non_linearity=non_linearity)
        self.odeblock_down2 = ODEBlock(odefunc=ode_down2, is_conv=True, tol=tol, adjoint=adjoint, solver=solver)
        self.conv_down2_3 = Conv2D(filters=num_filters*4, kernel_size=(1, 1), padding='same')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))

        ode_down3 = Conv2dODEFunc(num_filters=num_filters*4, augment_dim=augment_dim, time_dependent=time_dependent,
                                    non_linearity=non_linearity)
        self.odeblock_down3 = ODEBlock(odefunc=ode_down3, is_conv=True, tol=tol, adjoint=adjoint, solver=solver)
        self.conv_down3_4 = Conv2D(filters=num_filters*8, kernel_size=(1, 1), padding='same')
        self.maxpool3 = MaxPooling2D(pool_size=(2, 2))

        ode_down4 = Conv2dODEFunc(num_filters=num_filters*8, augment_dim=augment_dim, time_dependent=time_dependent,
                                    non_linearity=non_linearity)
        self.odeblock_down4 = ODEBlock(odefunc=ode_down4, is_conv=True, tol=tol, adjoint=adjoint, solver=solver)
        self.conv_down4_embed = Conv2D(filters=num_filters*16, kernel_size=(1, 1), padding='same')
        self.maxpool4 = MaxPooling2D(pool_size=(2, 2))

        ode_embed = Conv2dODEFunc(num_filters=num_filters*16, augment_dim=augment_dim, time_dependent=time_dependent,
                                    non_linearity=non_linearity)
        self.odeblock_embending = ODEBlock(odefunc=ode_embed, is_conv=True, tol=tol, adjoint=adjoint, solver=solver)

        #Expansive path
        self.transpose1 = Conv2DTranspose(filters=num_filters*8, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.conv_up_embed_1 = Conv2D(filters=num_filters*8, kernel_size=(1, 1), padding='same')
        ode_up1 = Conv2dODEFunc(num_filters=num_filters*8, augment_dim=augment_dim, time_dependent=time_dependent,
                                    non_linearity=non_linearity)
        self.odeblock_up1 = ODEBlock(odefunc=ode_up1, is_conv=True, tol=tol, adjoint=adjoint, solver=solver)

        self.transpose2 = Conv2DTranspose(filters=num_filters*4, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.conv_up1_2 = Conv2D(filters=num_filters*4, kernel_size=(1, 1), padding='same')
        ode_up2 = Conv2dODEFunc(num_filters=num_filters*4, augment_dim=augment_dim, time_dependent=time_dependent,
                                    non_linearity=non_linearity)
        self.odeblock_up2 = ODEBlock(odefunc=ode_up2, is_conv=True, tol=tol, adjoint=adjoint, solver=solver)

        self.transpose3 = Conv2DTranspose(filters=num_filters*2, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.conv_up2_3 = Conv2D(filters=num_filters*2, kernel_size=(1, 1), padding='same')
        ode_up3 = Conv2dODEFunc(num_filters=num_filters*2, augment_dim=augment_dim, time_dependent=time_dependent,
                                    non_linearity=non_linearity)
        self.odeblock_up3 = ODEBlock(odefunc=ode_up3, is_conv=True, tol=tol, adjoint=adjoint, solver=solver)

        self.transpose4 = Conv2DTranspose(filters=num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.conv_up3_4 = Conv2D(filters=num_filters, kernel_size=(1, 1), padding='same')
        ode_up4 = Conv2dODEFunc(num_filters=num_filters, augment_dim=augment_dim, time_dependent=time_dependent,
                                    non_linearity=non_linearity)
        self.odeblock_up4 = ODEBlock(odefunc=ode_up4, is_conv=True, tol=tol, adjoint=adjoint, solver=solver)

        self.classifier = Conv2D(filters=output_dim, kernel_size=(1, 1), activation='sigmoid')


    def call(self, inputs, return_features: bool=False):
        
        norm_inputs = self.norm_range(inputs)
        inps = self.input_layer(norm_inputs)
        x_cast = tf.cast(inps, dtype=tf.float64)
        
        # Contraction path
        features1 = self.odeblock_down1(x_cast)
        x = self.non_linearity(self.conv_down1_2(features1))
        x = self.maxpool1(x)
        x = Dropout(0.1)(x)

        features2 = self.odeblock_down2(x)
        x = self.non_linearity(self.conv_down2_3(features2))
        x = self.maxpool2(x)
        x = Dropout(0.1)(x)

        features3 = self.odeblock_down3(x)
        x = self.non_linearity(self.conv_down3_4(features3))
        x = self.maxpool3(x)
        x = Dropout(0.2)(x)

        features4 = self.odeblock_down4(x)
        x = self.non_linearity(self.conv_down4_embed(features4))
        x = self.maxpool4(x)
        x = Dropout(0.2)(x)

        x = self.odeblock_embending(x)

        #Expansive path
        x = self.transpose1(x)
        x = concatenate([x, features4])
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.odeblock_up1(x)

        x = self.transpose2(x)
        x = concatenate([x, features3])
        x = self.non_linearity(self.conv_up1_2(x))
        x = self.odeblock_up2(x)

        x = self.transpose3(x)
        x = concatenate([x, features2])
        x = self.non_linearity(self.conv_up2_3(x))
        x = self.odeblock_up3(x)

        x = self.transpose4(x)
        x = concatenate([x, features1])
        x = self.non_linearity(self.conv_up3_4(x))
        x = self.odeblock_up4(x)

        pred = self.classifier(x)

        if return_features:
            return features4, pred
        else:
            return pred

# Get U-Node model
def get_unode_model(input_dim: tuple, filters: int=4, non_linearity: str='lrelu', solver: str='adams') -> UNode:
    """Get an instance of UNode model

    Arguments:
        input_dim {tuple} -- shape of input in the form (height, width, channels) e.g. (256, 256, 3)

    Keyword Arguments:
        filters {int} -- number of filters to apply in convolutions (default: {4})
        non_linearity {str} -- activation function (default: {'lrelu'})
        solver {str} -- ODE solver to be applied (default: {'adams'})

    Returns:
        UNode -- an instance of the UNode model class
    """
    return UNode(num_filters=filters, input_dim=input_dim, non_linearity=non_linearity, solver=solver)