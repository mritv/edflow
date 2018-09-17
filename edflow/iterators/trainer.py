import tensorflow as tf

from edflow.iterators.model_iterator import HookedModelIterator
from edflow.hooks import LoggingHook, CheckpointHook, RetrainHook
from edflow.hooks import match_frequency
from edflow.project_manager import ProjectManager
from edflow.util import make_linear_var


P = ProjectManager()


class BaseTrainer(HookedModelIterator):
    def __init__(self,
                 trainer_config,
                 root_path,
                 model,
                 hooks=[],
                 hook_freq=100,
                 bar_position=0):
        '''
        Base class for models in this project. Implementations only need to
        define the make_loss_ops method which has to set the attributes
        log_ops, img_ops and return a per submodule loss tensor.
        Args:
            trainer_config (str): Some config file.
            root_path (str): Root directory to store all training outputs.
            model (Model): :class:`Model` to train.
            hooks (list): List of :class:'`Hook`s
            hook_freq (int): Step frequencey at which hooks are evaluated.
            bar_position (int): Used by tqdm to place bars at the right
                position when using multiple Iterators in parallel.
        '''
        self.config = trainer_config
        num_epochs = self.config['num_epochs']
        super().__init__(model, num_epochs, hooks, hook_freq, bar_position)

        self.root = root_path

        # hooks
        hf = hook_freq
        self.log_frequency = match_frequency(hf, self.config["log_freq"])
        self.check_frequency = match_frequency(hf, self.config["ckpt_freq"])
        self.logger.info('Log frequency: {}'.format(self.log_frequency))
        self.logger.info('Check frequency: {}'.format(self.check_frequency))

        # optimizer parameters
        self.initial_lr = self.config["lr"]
        self.lr_decay_begin = self.config["lr_decay_begin"]
        self.lr_decay_end = self.config["lr_decay_end"]

        self.setup()

    def fit(self, *args, **kwargs):
        '''Just renaming self.iterate'''
        return self.iterate(*args, **kwargs)

    def make_loss_ops(self):
        '''Should set attributes for log_ops, img_ops and return dictionary
        of per submodule losses.'''
        raise NotImplemented()

    def get_log_ops(self):
        return self.log_ops

    def get_img_ops(self):
        return self.img_ops

    def setup(self):
        '''Setup all ops needed for step_op as well as hooks.'''
        self.create_train_op()
        log_hook = LoggingHook(scalars=self.log_ops,
                               images=self.img_ops,
                               logs=self.log_ops,
                               root_path=self.root,
                               interval=self.log_frequency)
        check_hook = CheckpointHook(P.checkpoints,
                                    self.get_init_variables(),
                                    modelname=self.model.model_name,
                                    interval=self.check_frequency,
                                    step=self.global_step)
        self.hooks += [log_hook, check_hook]
        self.logger.debug('Set up Trainer')

    def create_train_op(self):
        '''Default optimizer + optimize each submodule'''
        # Optimizer
        self.lr = lr = make_linear_var(self.global_step,
                                       self.lr_decay_begin, self.lr_decay_end,
                                       self.initial_lr, 0.0,
                                       0.0, self.initial_lr)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                                                beta1=0.5,
                                                beta2=0.9)

        # Optimization ops
        losses = self.make_loss_ops()
        opt_ops = dict()
        for k in losses:
            variables = self.get_trainable_variables(k)
            opt_op = self.optimizer.minimize(losses[k],
                                             var_list=variables)
            opt_ops[k] = opt_op
        opt_op = tf.group(*opt_ops.values())
        with tf.control_dependencies([opt_op]):
            train_op = tf.assign_add(self.global_step, 1)
        self.train_op = train_op

        # add log ops
        self.log_ops["global_step"] = self.global_step
        self.log_ops["lr"] = self.lr

    def step_ops(self):
        return self.train_op

    def reset_global_step(self):
        self.hooks += [RetrainHook(self.global_step)]

    def initialize(self, checkpoint_path=None):
        '''Set weights to those stored at checkpoint.

        Args:
            checkpoint_path (str): Path to checkpoint(s).
        '''

        self.init_op = None
        self.checkpoint_path = checkpoint_path

        if checkpoint_path is not None:
            saver = tf.train.Saver(self.get_init_variables())
            saver.restore(self.session, checkpoint_path)
            self.logger.info("Restored model from {}".format(checkpoint_path))
        else:
            self.logger.info('Training from scratch.')

    def get_init_op(self):
        '''If there is a checkpoint do nothing, else initialize all vars.

        Returns:
            tf.op: init op to be run before training.
        '''
        if self.checkpoint_path is None:
            self.init_op = tf.variables_initializer(self.get_init_variables())
            self.logger.info("Initialized model from scratch")
        else:
            self.init_op = tf.no_op()
        return self.init_op

    def get_trainable_variables(self, submodule):
        trainable_variables = [v for v in tf.trainable_variables()
                               if v in self.model.variables]
        return [v for v in trainable_variables if submodule in v.name]

    def get_init_variables(self):
        return [v for v in tf.global_variables() if "__noinit__" not in v.name]