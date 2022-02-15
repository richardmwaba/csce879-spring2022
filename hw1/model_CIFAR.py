### Model imports 
import util_CIFAR as utils

from tensorflow.keras.models import Sequential


## Create model
def create_model(filters: int, kernel: int, strides: int, network_arch: dict, classes: int, **kwargs) -> Sequential:
    """Create network model

    Arguments:
        filters {int} -- number of filters
        kernel {int} -- size of kernel
        strides {int} -- size of strides
        network_arch {dict} -- network architecture as key, value pair
        classes {int} -- number of target classes

    Returns:
        Sequential -- network model
    """

    net_track = [x for x in network_arch]
    # Network architecture
    net_architecture = []
    # Current filter
    current_filter = filters

    for val in net_track:
       if val == 'convolutional':
           layers, new_filters = utils.convolutional_layers(filters=current_filter, kernel=kernel, strides=strides,
                                    num=network_arch[val], mult_filter=True)
           current_filter = new_filters
           net_architecture += layers

       if val == 'residual' and 'x' in kwargs:
           net_architecture += utils.residual_block(filters=current_filter, strides=strides, resnum=network_arch[val], 
                                initial_x=kwargs['x'])
       else:
           continue
          
       if val == 'dense':
           if val == net_track[-1]:
               net_architecture += utils.dense_layer(units=classes, dense_num=network_arch[val], activation='softmax')
           else:
               net_architecture += utils.dense_layer(units=100, dense_num=net_architecture[val], activation='relu')

       if val == 'normalization':
           net_architecture.append(utils.batch_normalization())

       if val == 'pooling':
           net_architecture.append(utils.pooling_layer())

       if val == 'flatten':
           net_architecture.append(utils.flatten())

    model = Sequential(net_architecture)

    return model


