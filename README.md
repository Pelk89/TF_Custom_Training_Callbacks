# Tensorflow Custom Training with Custom Callbacks

## Overview

Use git clone for using the repository locally.

```bash
git clone https://github.com/Pelk89/TF_Custom_Training_Callbacks.git
```

Low-level training and evaluation loops based on following MNIST **[Tutorial](https://keras.io/guides/writing_a_training_loop_from_scratch/)** by Keras.

## Usage

You want to use your low-level training and evaluation loops and don't want to miss the convenience of the Keras callbacks? No problem! In this tutorial, I will show you how to simply implement Keras callbacks in low-level training and evaluation loop. With a callback, it is possible to perform different actions at several stages while training your deep neural network. By default, Keras provides easy and convenient built-in callbacks like
[ReduceLROnPlateau](https://keras.io/api/callbacks/reduce_lr_on_plateau/) (Reduce learning Rate on Plateau) or 
[EarlyStopping](https://keras.io/api/callbacks/early_stopping/) for your training and evaluation loop. Their usage is covered in the `.fit()` method of a model and can be passed as a list of callbacks: 

```python
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(dataset, epochs=10, callbacks=my_callbacks)
```
At each stage of the training (e.g. at the start or end of an epoch)  all relevant methods will be called automatically.


---

Tensorflow Custom Training with Custom Callbacks

You want to use your low-level training and evaluation loops and don't want to miss the convenience of the Keras callbacks? No problem! In this tutorial, I will show you how to simply implement Keras callbacks in low-level training and evaluation loop. For this article, I assume that you are familiar with the basic terminology and principles of Machine Learning and have done some toy examples with Keras.
HINT: When I first got in touch with Keras, I was overwhelmed with the different terminology of Machine Learning and was happy that I found the Machine Learning Glossary by Google.
With a callback, it is possible to perform different actions at several stages while training your deep neural network. By default, Keras provides easy and convenient built-in callbacks like ReduceLROnPlateau (Reduce learning Rate on Plateau) or EarlyStopping for your training and evaluation loop. Their usage is covered in the .fit() method of a model and can be passed as a list of callbacks:

At each stage of the training (e.g. at the start or end of an epoch) all relevant methods will be called automatically. You can find more detailed information about the callback methods in the Keras documentation. To write your own Callbacks I recommend the article Building Custom Callbacks with Keras and TensorFlow 2 by B. Chen. 
I won't go into detail of how to implement a custom training loop. So if you want to have more information about that I recommend the Keras Classification Tutorial. For this tutorial we will slightly modify the mentioned classification tutorial for the MNIST dataset. The MNIST dataset contains handwritten digits and is commonly used as toy example for Machine Learning. Furthermore we will implement the widely used callback ReduceLROnPlateau and add also an exponential reducing of the learning rate. 
This tutorial can be found on my Github. Use git clone to run it locally on your computer
git clone https://github.com/Pelk89/TF_Custom_Training_Callbacks.git

---

Reduce Learning Rate On Plateau
Reducing the learning rate is often used when a metric has stopped improving. Moreover once the learning stagnates Machine Learning Models often benefit from reducing the learning rate linearly. But what if we want to reduce the learning rate exponentially when the metric has stopped improving? No problem at all! To understand whats going on under the hood we need to go deeper into Keras library. Let's dig a little bit to the heart of the Keras Callbacks in tf.keras.callbacks.Callback. Thanks to Francois Chollet and the other authors we will find a incredible clean and comprehensible code for the class ReduceLROnPlateau. With this basis we easily can reuse the existing code and use it for our purposes. How great is that?

When we analyze the class we will see for example the explanation for the arguments that can be passed to the __init__ method of the class. We will go into detail for the important arguements for this tutorial. A more detailed overview can be found at Keras ReduceLROnPlateau documentation.
Arguments
monitor: quantity (e.g. validation loss) to be monitored.
factor: factor by which the learning rate will be reduced. new_lr = lr * factor.
patience: number of epochs with no improvement after which learning rate will be reduced.
cooldown: number of epochs to wait before resuming normal operation after learning rate has been reduced.

And more importantly we can identify the underlying algorithm and when its called while training your neural network. The class uses two methods
on_train_begin(): Called once when training begins and resetting wait and cooldown timer
on_epoch_end(): Called on every end of an epoch and change the learning rate linearly depended on the defined cooldown timer and patience.

Modify Reduce Learning Rate On Plateau
Now that we have a detailed overview of the Reduce Learning Rate On Plateau algorithm, we are able to modify it for our needs. Moreover we can implement the algorithm at the right position in our custom training and evaluation loop. In this tutorial we will monitor the validation loss of our training model. First of all we need to know that we cant use the internal arguments self.model as the model is an instance of keras.models.Model and a reference of the model being trained. So everything with self.model needs to be replaced and passed as argument into the __init__ or on_epoch_end() method. The changes are marked as 
## Custom modification: "Reason for Modification"
If we want to reduce the learning rate exponentially we need to add a argument to __init__ method and also modify the method on_epoch_end(). 
Modfiy __init__
Because we measure the validation loss of our neural network we can remove the monitor argument. Also we want to set a boolean to set control whether we want to reduce the learning rate exponentially with reduce_exp and set its default value to false. Likewise we need to pass the optimizer to the method.

Modify on_epoch_end()
Now to the more complex part of the modification of the algorightm. Normally its method is called on every end of an epoch during training. In our case we additionally need to pass the loss of our validation dataset on every epoch end of our training to the method. 

Putting it all together

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)