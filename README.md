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

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)