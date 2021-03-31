import tensorflow as tf
import numpy as np

class CustomReduceLRoP():

    """ Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This callback monitors a
        quantity and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.

        Example:
        ```python
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                        patience=5, min_lr=0.001)
        model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```
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
        reduce_exp: reducing the learning rate exponentially
    """





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
                 reduce_exp = False,
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
        self.reduce_exp = reduce_exp
        

        self._reset()

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

    def train_begin(self, logs=None):
        self._reset()

    def epoch_end(self, epoch, loss, logs=None):


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
                if old_lr > self.min_lr:
                    ## Custom modification: Exponential learning Rate
                    if self.reduce_exp == True:
                        new_lr = old_lr * tf.math.exp(-0.1)
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

    def in_cooldown(self):
        return self.cooldown_counter > 0