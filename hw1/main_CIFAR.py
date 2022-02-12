import model_CIFAR as model
import util_CIFAR as utils


## Load data

## Partition data

## Perform one-hot encoding on targets

## Define a number of different architectures to use for training

## Loop to through all architectures training in each iteration

## Save best model to disk

## Validate best model on test data

## Analyze performance and tabulate statistics

# Sample architecture
architec = {'convolutional': 2, 'normalization': 1, 'pooling': 1, 'residual': 3, 'pooling': 1, 
            'residual': 2, 'pooling': 1, 'flatten': 1, 'dense': 1}
# Sample network
conv_classifier = model.create_model(filters=32, kernel=3, strides=2, network_arch=architec, classes=100)

print("\n===========================================================================")
print("Model has been created successfully: ")
print(conv_classifier)