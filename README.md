# AI Course - Introduction to Neural Networks

In this course we will teach you the basis of the modern approach to the topic of artificial intelligence - the neural networks. These models solve problems that have long been considered unsolvable by a computer, like image classification, precise language translation or even generating new content like text and images. But how actually can you do those things just by the power of math and a computer? This course will provide you the basic skills to understand what a neural network is, what you can do with it and how you can program your own neural network using the Python framework PyTorch, which is used by a lot of companies and universities today.

## Agenda

### The data

First we will have to look at the data we can feed into a neural network. As we are working with computers all our data like tables, images or text has to be transformed into a mathematical format. We will talk about _vectors_, _matrices_ and _tensors_ and how we can transfer data into these formats. We will build a data loader using PyTorch which loads our data from our disk and transforms it into the mathematical representation.

##### Notebooks

- Notebook 1: Load data from a CSV file and transform it into a tensor. Iterate over the data and print it.
- Noteboook 2: Load images from the disk with a data loader and iterate over it. Apply image augmentation to the data like rotating the images, so that the more variance is added to the data. This can help neural networks to achieve better results.

### The neuron

The basic building block of a neural network is a neuron. We will talk about what a (mathematical) neuron is, what it has to do with matrices and what it is able to do. You don't need to know a lot of math, because neurons are actually very simple. You just need to know how to multipy two matrices and then apply a function to the resullting vector. That's all. It is more important to visually see what a neuron can do by plotting the input and output vectorsof a neuron which we will do using matplotlib. We will also see that we can build complex functions just by building layers of parallel neurons and networks of chained neuron layers.

##### Notebooks

- Notebook 1: The power of matrix multiplication: Let's see what matrix multiplication can do. You can rotate, translate and scale vectors just by multiplying them with a matrix. You can even increase or decrease the dimension of data points by matrix multiplication, which is an important concept if we want to map data points to classes in a classification problem oor to predict y-values in a regression probllem. In this notebook you will plot the input vectors before multiplying them with a matrix and the results after multiplying them with a matrix.

- Notebook 2: Activation functions: After we have seen the power of matrix multipication we will have a look on the second part of a neuron - the activation function. An activation function works like a filter that excludes data points from the operation a matrix applies on the data points. This is necessary because matrix multipplication is a linear operation, which means the same operation is applied to all data points. If we want to apply a transformation like rotation only to a fraction of the data points we need activation function to define which data points should be transformed and which should not. We will build two neurons in this notebook. The first one rotates half of the data points to the left, the other neuron rotates the other half of the data points to the right. Finally we reduce the dimension of the rotated data points with another neuron. Actually we now have built our first neural network!

### The Backpropagation Algorithm

After we understood what a neural network actually is - namely a transformation of data points from one shape into another shape - we can now have a look on how we train these networks to do what we actually want them to do. In this lecture we introduce terms like _loss functions_ and _gradients_ which can help us to find the right parameters of a model. You will see that math you learnt in school is absolutely sufficient to compute all we need so that a neural network can actually find the best parameters.

- Notebook 1: We will implement the backpropagation algorithm by applying it to a very easy example. We train a linear function of the form `y = a*x`, where a is the only parameter we can choose, to go through a single data point by defining a loss function, computing the gradient of the loss function and then use this gradient to update the parameter of our linear function. We will do this until the linear function goes through the given data point. We will see that the backpropagation algorithm actually is very simple, and neural networks use the same algorithm just for learning more complex functions. We will have a look at the _L1_ and _L2_ loss functions and check which one works better for our example.
- Notebook 2: After we have trained the linear function of form `y=a*x` with the backpropagation algorithm we will see how we can train a function of the form `y=a*x+b`, which has two parameters. This is neccessary to understand layers in a neural network, where we have multiple input variables.
- Notebook 3: Finally we will train a function where we run the results of the first function of form `y=a*x+b` through another function, which corresponds to having multiple layers in a neural network. We will see how we can use the _chain rule_ to compute the gradient of neural networks with multiple layers.

### Using PyTorch to define and train neural networks

Until now we have written our neural networks by hand, so it is time to introduce a framework which actually can help us doing this. We will build two neural networks with PyTorch that will work with the data of our CSV files. The first network performs a regression on our data points. The second notebook will classify data points in another CSV file. Most of the code will be already implemented in those notebooks, your task will be to understand this code and fill the missing parts.

- Notebook 1: You will load data from a CSV file and plot the data points. You will see that the data points look like a parabola. We will now train a neural network that tries to predict this parabola. You will use the data loader defined in the previous lectures and run these data points through the neural network. We will use PyTorch to define a neural network, choose which loss function we will use, compute the gradients by comparing the predictions of the neural network to the true outputs and optimizing the parameters by performing the backpropagation step. You will see that PyTorch hides all the math from you, which is great because it makes the code very readable. However it is good to know what PyTorch is actually doing under the hood, which is why we have done the mathematical lectures before.
- Notebook 2: In this notebook we will use the famous IRIS dataset, which contains some metrics about flowers. Your task will be to predict the type of flower based on those metrics. So in this notebook we will introduce classification problems and how you can solve them with PyTorch.

After this lecture you are able to understand the fundamentals of PyTorch like data loaders, neural network definitions, training loops, the computation of the loss and the update step. You will find these concept in all PyTorch projects, whether you are predicting data points, classifying images or translating text.

### Final Project: Image classification and transfer learning

In this lecture we are finally working with more complex data. We build a neural network that is able to classify images. For that we will work with the MNIST dataset which contains black and white images of handwritten numbers. When working with images we need a new kind of layer in our neural network which is called a convolutional layer. We will see what this kind of layer can do, how it helps us to extract information from an image and how we can build a convolutional neural network with PyTorch.
Also we will talk about the concept of transfer learning, which means that we will load a pretrained neural network and only train the last classification layer. We will see that this is much more efficient than training our own network from scratch, needs less time for training and achieving better results.

- Notebook 1: Understanding convolutional layers. We will have a look at convolutional layers in convolutional neural networks. You will get a notework where you will find a symbol in an image by programming a matrix that is slided over the image and multiplied with the overlapping image fraction time. The result of this operation will be a heatmap which has the highest value at the point where the symbol is located. For simplicity we will do this with a simple black and white image and with simple symbols like a diamond and a square. This is essential for understanding how convolutional neural networks work.
- Notebook 2: Load the MNIST dataset and visualize it using matplotlib. We will have a look on the dataset in this notebook, so that we can see how the images actually look like. We will then define our own simple convolutional neural network and train it on the MNIST dataset. Again most of the code in the notebook will be written fro you. You task is to fill the missing parts.
- Notebook 3: We will replace our own convolutional network by a pretrained neural network named "MobileNet". You will learn how you can load pretrained neural networks from PyTorch Hub and how you can adapt these networks to your own classification problems. We will train MobileNet on the MNIST dataset and compare the results to our own convolutional network. You are free to try other pretrained convolutional networks like ResNet or EfficientNet if you like. If you feel secure with convolutional networks you can even try to classify images of cats and dogs or of different flower types.

### Final Notes

In this course we have learned the basics of neural networks. You have learned the mathematical concepts behind neural networks like matrix multiplications, activation funcitons, loss functions, gradients and the backpropagation algorithm. Also you have learned how to transfer data like CSV files or images into a mathematical format called tensors. You can define data loaders, neural networks and training loops with PyTorch to train neural networks on your own data. And you understand the concept of transfer learning to achieve much better results with neural networks by using pretrained neural networks that have been trained by companies on large datasets that would not be possible for us as private persons.

But there are also types of neural networks we haven't discussed in this course like models for translating text or generating images. We want to provide you resources to continue learning neural networks in this lecture and an overview what else is possible using neural networks. You will get an overview on how neural networks can work with text or audio, what sequence models like RNNs or Transformers are and what kind of models exist that are used for generative AI like chat bots or image generators. In the internet there are a lot of good resources that can help you to learn more about neural networks, even lots of free lectures. You will get an overview of courses and platforms to learn more about neural networks and gain official certifications in that area.

### Resources

##### YouTube

A complete lecture about Computer Vision by the Michigan University can be found on YouTube.
https://youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&si=Lf2_d2DJ51xJP7R7

Another good YouTube channel about neural networks is the channel of Luis Serrano.
https://www.youtube.com/@SerranoAcademy

##### Udemy

On Udemy there are really good courses about Neural Networks on the channel named "Lazy Programmer Inc."
https://www.udemy.com/user/lazy-programmer/

##### Coursera

Another good learning platform is Coursera. Here I recommend the channel "DeepLearning.AI" (https://www.coursera.org/deeplearning-ai)

##### Datacamp

Datacamp is a nice platform to learn Python skills and the most used libraries like Numpy, Pandas, Matplotlib or Seaborn. There are also courses about working with neural networks, preparing data for training and much more.
https://www.datacamp.com

##### Udacity

For professional certificates I can highly recommend the platform Udacity. On Udacity you can take part in so called nanodegree programs, which consist of lectures about the theory of neural networks and practical projects which you have to submit at the end of the nanodegree program. These courses are not cheap, so try to learn the theory on other platforms before. You can also get a discount on Udacity when using your student email. On Udacity you will have to submit three projects to get a certification. These projects are really close to industrial use cases, so if you pass them you have proven that you really can work with neural networks. Also you can find these project notebooks by using Google, because participants of a nanodegree program are allowed to upload their solutions to their GitHub profiles tp proove their skills. On Udacity I there are basically five nanodegree programs for deep learning:

- Introcution to Machine Learning with PyTorch: https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229
- Deep Learning: https://www.udacity.com/course/deep-learning-nanodegree--nd101
- Computer Vision: https://www.udacity.com/course/computer-vision-nanodegree--nd891
- Natural Language Processing: https://www.udacity.com/course/natural-language-processing-nanodegree--nd892
- Deep Reinforcement Learning: https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
