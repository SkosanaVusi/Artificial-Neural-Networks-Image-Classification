These Artificial Neural Networks can classify objects in the CIFAR10 dataset

![[images.png]]

# Multi-layer Perceptron

A Standard Multi-layer Perceptron which is a type of artificial neural network (ANN) commonly used in machine learning. It is a feedforward neural network that consists of multiple layers of nodes or "neurons".

![[mlp.png]]

# Usage:

python MLP.py

# Output

Train loss = a, Test accuracy = b

- Where a and b is a percentage number

# Requirements

torch, torchvision, matplotlib

# Example:

python MLP.py

# Convolutional Neural Network

CNN stands for Convolutional Neural Network, which is a type of neural network that is commonly used in deep learning applications for image and video recognition, natural language processing, and other tasks that involve analyzing input data with a grid-like topology.

The key characteristic of CNNs is the use of convolutional layers, which apply a set of filters to the input data in order to identify features or patterns. These filters are typically small in size, and they are slid across the input data to generate a set of output feature maps. The output of each convolutional layer is then passed through a non-linear activation function, such as ReLU, to introduce non-linearity into the model.

![[cnn.png]]

# Usage:

python CNN.py

# Output

Train loss = a, Test accuracy = b

- Where a and b is a percentage number

# Requirements

torch, torchvision, matplotlib

# Example:

python CNN.py

# Residual Network

ResNet (short for Residual Network) is a type of neural network architecture that was introduced in 2015 by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun of Microsoft Research.

The key innovation of ResNet was the introduction of residual connections, which allow for much deeper neural networks to be trained. In a traditional neural network, each layer is connected only to the previous layer. However, in a ResNet, each layer is connected to the previous layer and the layer two steps back. This means that information can bypass one or more layers and travel directly to the next layer, which allows for the network to learn more complex features and improve accuracy.

# Usage:

python ResNet.py

# Output

Train loss = a, Test accuracy = b

- Where a and b is a percentage number

# Requirements

torch, torchvision, matplotlib

# Example:

python ResNet.py
