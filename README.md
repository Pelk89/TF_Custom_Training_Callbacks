
# Hands-on: Kears Custom Training with Custom Callbacks

<img src="https://github.com/Pelk89/TF_Custom_Training_Callbacks/blob/main/tf_keras.png" width="100%">



You want to use low-level training and evaluation loops and don’t want to miss the convenience of the Keras callbacks? No problem! In this tutorial, we will implement a Keras callback in a low-level training and evaluation loop. For this article, I assume that you are familiar with the basic terminology and principles of Machine Learning and have done some toy examples with Keras.

*HINT: When I first got in touch with Tensorflow and Keras, I was overwhelmed with the different terminology of Machine Learning and was happy that I found the [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/#l) by Google.*

With a callback, it is possible to perform different actions at several stages while training your deep neural network. By default, Keras provides easy and convenient built-in callbacks like [ReduceLROnPlateau](https://keras.io/api/callbacks/reduce_lr_on_plateau/) (Reduce learning Rate on Plateau) or [EarlyStopping](https://keras.io/api/callbacks/early_stopping/) for your training and evaluation loop. Their usage is covered in the *.fit()* method of a model and can be passed as a list of callbacks:

```python
# Found on https://keras.io/api/callbacks/

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(dataset, epochs=10, callbacks=my_callbacks)
```


At each stage of the training (e.g. at the start or end of an epoch) all relevant methods will be called automatically. You can find more detailed information about the callback methods in the Keras [documentation](https://keras.io/guides/writing_your_own_callbacks/). To write your Callbacks you should give the article [Building Custom Callbacks with Keras and TensorFlow 2](https://towardsdatascience.com/building-custom-callbacks-with-keras-and-tensorflow-2-85e1b79915a3) by [B. Chen](https://bindichen.medium.com/) a try.

I won’t go into detail about how to implement a custom training loop. So if you want to have more information about that I recommend the [Keras Classification Tutorial](https://keras.io/guides/writing_a_training_loop_from_scratch/). For this tutorial, we will slightly modify the mentioned classification tutorial for the MNIST dataset. The MNIST dataset contains handwritten digits and is commonly used as toy example for Machine Learning. Furthermore, we will implement the widely used callback ReduceLROnPlateau and add also an exponential reduction of the learning rate.

This tutorial can be found on [Github](https://github.com/Pelk89/TF_Custom_Training_Callbacks). Use git clone to run it locally in Jupyter notebook.
```bash
git clone https://github.com/Pelk89/TF_Custom_Training_Callbacks.git
```
## Reduce Learning Rate On Plateau

Reducing the learning rate is often used when a metric has stopped improving. Moreover once the learning stagnates Machine Learning Models often benefit from reducing the learning rate *linearly*. But what if we want to reduce the learning rate exponentially when the metric has stopped improving? No problem at all! To understand what’s going on under the hood we need to go deeper into Keras library. Let’s dig a little bit to the heart of the Keras Callbacks in *tf.keras.callbacks.Callback.* Thanks to Francois Chollet and the other authors we will find an incredibly clean and comprehensible code for the class ReduceLROnPlateau. With this basis, we easily can reuse the existing code and use it for our purposes. How great is that?

```python
# Original Callback found in tf.keras.callbacks.Callback
# Copyright The TensorFlow Authors and Keras Authors.

@keras_export('keras.callbacks.ReduceLROnPlateau')
class ReduceLROnPlateau(Callback):
  """Reduce learning rate when a metric has stopped improving.

  Models often benefit from reducing the learning rate by a factor
  of 2-10 once learning stagnates. This callback monitors a
  quantity and if no improvement is seen for a 'patience' number
  of epochs, the learning rate is reduced.

  Example:

  
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
  model.fit(X_train, Y_train, callbacks=[reduce_lr])
  

  Arguments:
      monitor: quantity to be monitored.
      factor: factor by which the learning rate will be reduced. new_lr = lr *
        factor
      patience: number of epochs with no improvement after which learning rate
        will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of {auto, min, max}. In `min` mode, lr will be reduced when the
        quantity monitored has stopped decreasing; in `max` mode it will be
        reduced when the quantity monitored has stopped increasing; in `auto`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
      cooldown: number of epochs to wait before resuming normal operation after
        lr has been reduced.
      min_lr: lower bound on the learning rate.
  """

  def __init__(self,
               monitor='val_loss',
               factor=0.1,
               patience=10,
               verbose=0,
               mode='auto',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0,
               **kwargs):
    super(ReduceLROnPlateau, self).__init__()

    self.monitor = monitor
    if factor >= 1.0:
      raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
    if 'epsilon' in kwargs:
      min_delta = kwargs.pop('epsilon')
      logging.warning('`epsilon` argument is deprecated and '
                      'will be removed, use `min_delta` instead.')
    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.mode = mode
    self.monitor_op = None
    self._reset()

  def _reset(self):
    """Resets wait counter and cooldown counter.
    """
    if self.mode not in ['auto', 'min', 'max']:
      logging.warning('Learning Rate Plateau Reducing mode %s is unknown, '
                      'fallback to auto mode.', self.mode)
      self.mode = 'auto'
    if (self.mode == 'min' or
        (self.mode == 'auto' and 'acc' not in self.monitor)):
      self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
      self.best = np.Inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
      self.best = -np.Inf
    self.cooldown_counter = 0
    self.wait = 0

  def on_train_begin(self, logs=None):
    self._reset()

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = K.get_value(self.model.optimizer.lr)
    current = logs.get(self.monitor)
    if current is None:
      logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))

    else:
      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

      if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
      elif not self.in_cooldown():
        self.wait += 1
        if self.wait >= self.patience:
          old_lr = float(K.get_value(self.model.optimizer.lr))
          if old_lr > self.min_lr:
            new_lr = old_lr * self.factor
            new_lr = max(new_lr, self.min_lr)
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
              print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                    'rate to %s.' % (epoch + 1, new_lr))
            self.cooldown_counter = self.cooldown
            self.wait = 0

  def in_cooldown(self):
    return self.cooldown_counter > 0
```

When we analyze the class we will see for example the explanation for the arguments that can be passed to the *\_\_init\_\_* method. Some of the arguments are listed below. A more detailed overview can be found at Keras [ReduceLROnPlateau](https://keras.io/api/callbacks/reduce_lr_on_plateau/) documentation.

**Arguments**

* **monitor**: quantity (e.g. validation loss) to be monitored.

* **factor**: factor by which the learning rate will be reduced. new_lr = lr * factor.

* **patience**: number of epochs with no improvement after which learning rate will be reduced.

* **cooldown**: number of epochs to wait before resuming normal operation after learning rate has been reduced.

But more importantly, we can identify the underlying algorithm and when it’s called while training our neural network. The class uses two methods:

* **on_train_begin()**: Called once when training begins and resetting wait and cooldown timer

* **on_epoch_end()**: Called on every end of an epoch and change the learning rate linearly depended on the defined cooldown timer and patience.



## Modify Reduce Learning Rate On Plateau

Now that we have a detailed overview of the Reduce Learning Rate On Plateau algorithm, we can modify it for our needs. Moreover, we can implement the algorithm at the right position in our custom training and evaluation loop. In this tutorial, we will monitor the validation loss of our training model. First of all, we need to know that we can not use the arguments *self.model *as the model is an instance of *keras.models.Model* and a reference of the model being trained. So everything with *self.model* needs to be replaced and passed as an argument into the *\_\_init\_\_* or *on_epoch_end()* method. I marked the changes in the code as following:

    ## Custom modification: "Reason for Modification"

For reducing the learning rate exponentially we need to add an argument to *\_\_init\_\_* method and also modify the method *on_epoch_end().*

### Modfiy \_\_init\_\_ and _reset

Because we measure the validation loss of our neural network we can remove the *monitor* argument. Next, we want to set a boolean *reduce_exp* to control whether we want to reduce the learning rate exponentially. Likewise, we need to pass the optimizer to the method.

```python
# Original Callback found in tf.keras.callbacks.Callback
# Copyright The TensorFlow Authors and Keras Authors.
# Modification by Alexander Pelkmann

def __init__(self,
              ## Custom modification:  Deprecated due to focusing on validation loss
              # monitor='val_loss',
              factor=0.1,
              patience=10,
              verbose=0,
              mode='auto',
              min_delta=1e-4,
              cooldown=0,
              min_lr=0,
              sign_number = 4,
              ## Custom modification: Passing optimizer as arguement
              optim_lr = None,
              ## Custom modification:  Exponentially reducing learning
              reduce_lin = False,
              **kwargs):

    ## Custom modification:  Deprecated
    # super(ReduceLROnPlateau, self).__init__()

    ## Custom modification:  Deprecated
    # self.monitor = monitor
    
    ## Custom modification: Optimizer Error Handling
    if tf.is_tensor(optim_lr) == False:
        raise ValueError('Need optimizer !')
    if factor >= 1.0:
        raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
    ## Custom modification: Passing optimizer as arguement
    self.optim_lr = optim_lr  

    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.mode = mode
    self.monitor_op = None
    self.sign_number = sign_number
    

    ## Custom modification: Exponentially reducing learning
    self.reduce_lin = reduce_lin
    self.reduce_lr = True
    

    self._reset()
```
Next we need to remove the self.monitor in the _reset() method:

```python
    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            print('Learning Rate Plateau Reducing mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                ## Custom modification: Deprecated due to focusing on validation loss
                # (self.mode == 'auto' and 'acc' not in self.monitor)):
                (self.mode == 'auto')):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

```

### Modify on_epoch_end()

Normally the method is called on every end of an epoch during training. In our case we additionally need to pass the loss of our validation dataset and the epoch to *epoch_end()* on every epoch end of our training

```python
# Original Callback found in tf.keras.callbacks.Callbac
# Copyright The TensorFlow Authors and Keras Authors.
# Modification by Alexander Pelkmann

def on_epoch_end(self, epoch, loss, logs=None):


    logs = logs or {}
    ## Custom modification: Optimizer
    # logs['lr'] = K.get_value(self.model.optimizer.lr) returns a numpy array
    # and therefore can be modified to          
    logs['lr'] = float(self.optim_lr.numpy())

    ## Custom modification: Deprecated due to focusing on validation loss
    # current = logs.get(self.monitor)

    current = float(loss)
    
    ## Custom modification: Deprecated due to focusing on validation loss
    # if current is None:
    #     print('Reduce LR on plateau conditioned on metric `%s` '
    #                     'which is not available. Available metrics are: %s',
    #                     self.monitor, ','.join(list(logs.keys())))

    # else:

    if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

    if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
    elif not self.in_cooldown():
        self.wait += 1
        if self.wait >= self.patience:
            
            ## Custom modification: Optimizer Learning Rate
            # old_lr = float(K.get_value(self.model.optimizer.lr))
            old_lr = float(self.optim_lr.numpy())
            if old_lr > self.min_lr and self.reduce_lr == True:
                ## Custom modification: Linear learning Rate
                if self.reduce_lin == True:
                    new_lr = old_lr - self.factor
                    ## Custom modification: Error Handling when learning rate is below zero
                    if new_lr <= 0:
                        print('Learning Rate is below zero: {}, '
                        'fallback to previous learning rate: {}. '
                        'Stop reducing learning rate during training.'.format(new_lr, old_lr))  
                        new_lr = old_lr
                        self.reduce_lr = False                           
                else:
                    new_lr = old_lr * self.factor                   
                

                new_lr = max(new_lr, self.min_lr)


                ## Custom modification: Optimizer Learning Rate
                # K.set_value(self.model.optimizer.lr, new_lr)
                self.optim_lr.assign(new_lr)
                
                if self.verbose > 0:
                    print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                            'rate to %s.' % (epoch + 1, float(new_lr)))
                self.cooldown_counter = self.cooldown
                self.wait = 0
```

## Putting it all together

Let’s put everything together and implement it in our custom training loop! First, we need to import and initiate the class like this:

```python
from custom_callback import CustomReduceLRoP


## Custom Modification: Set reduce_lin true for linear reducing; Set factor to reduce learning rate by 0.0001

reduce_rl_plateau = CustomReduceLRoP(patience=2,
                              factor=0.0001,
                              verbose=1, 
                              optim_lr=optimizer.learning_rate, 
                              reduce_lin=True)
```



We set *reduce_exp* on true since we want to reduce the learning rate exponentially. Furthermore, we know from our analysis of the callback, that the callback will be called once on training start and whenever the epochs end. Simple add .*train_begin()* before the training loop is starting to reset the cooldown and wait timer. Next, add *.epoch_end()* at the end of your training loop.

```python
# Writing a training loop from scratch

# Author: fchollet
# Date created: 2019/03/01
# Last modified: 2020/04/15
# Description: Complete guide to writing low-level training & evaluation loops.


epochs = 15

## Custom Modification: Reset cooldown and wait timer for the callback
reduce_rl_plateau.on_train_begin()

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * 64))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()

 
    print("Validation acc: %.4f" % (float(val_acc),))

    ## Custom Modification: pass epoch and validation loss to the callback 
    reduce_rl_plateau.on_epoch_end(epoch, val_acc)


```
### Training your neural network

You made it! Finally, we can start the training of our neural network! Start the training of the neural network by simply cloning the repository and run the cells in the Jupyter Notebook.

![Reducing the learning rate after 10 epochs of no improving the validation loss.](https://cdn-images-1.medium.com/max/2014/1*3z9FJ29sjSA5TJiqWae-AA.png)*Reducing the learning rate after 10 epochs of no improving the validation loss.*



After 10 epochs when the validation loss is not improving the learning rate will be reduced exponentially!

## Conclusion

By default, Keras provides convenient callbacks built-in callbacks for your training and evaluation loop for the *.fit() *method of a model. But when you write your custom training loop to get a low-level control for training and evaluation it’s not simply possible to use built-in callbacks.

In this tutorial, we implemented the famous reduce learning rate on plateau callback by using its natively implemented callback. Moreover, we modified the callback to reduce the learning rate linearity.

Thanks for reading this article. I continuously want to improve my skill in machine learning. So If you have anything to add or have any ideas for this topic feel free to leave a comment.
