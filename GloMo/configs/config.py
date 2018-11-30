class Config():
    def __init__(self):
        # Network basic settings
        self.BATCH_SIZE = 1
        self.EMBEDDING_DIMS = 512
        self.NUM_CLASSES = 1000
        self.LEARNING_RATE = 0.01
        self.MOMENTUM = 0.9
        self.DEFAULT_OPTIMIZER = 'sgd'
        self.TRAINING_EPOCH = 25
        self.ITERATION_PER_EPOCH = 10000
        self.EVALUATION_PER_EPOCH = True
        self.EVALUATION_RUNS = 500

        self.CHECKPOINT_DIR = '../experiments/ckpts/'
        self.LOG_DIR = '../experiments/logs/'

        # Graph predictor settings
        self.KEY_CNN_LAYERS = QUERY_CNN_LAYERS = 5
        self.KEY_CNN_CONV_LENGTH = 3
        self.QUERY_CNN_CONV_LENGTH = 3
        self.GRAPH_LAYERS = 'every'
        self.GRAPH_BIAS = 0.1
        self.GRAPH_W_UNITS = self.EMBEDDING_DIMS
        self.GRAPH_SCOPE = 'Graph_Predictor'

        # Feature predictor settings
        self.RNN_EACH_LAYERS = 1
        self.USING_RNN_AT_EACH_GRAPH = False
        self.USING_GRAPH_AT = 'every'
        self.FEATURE_SCOPE = 'Feature Predictor'
        self.LINEAR_SHORTCUT_RATE = 1.0
        self.LINEAR_ACTIVATION = 'relu'
        self.LINEAR_BIAS = False
        self.PREDICTION_LENGTH = 10

        self.list_all_member()

    def list_all_member(self):
        for name, value in vars(self).items():
            print('{}={}'.format(name, value))

c = Config()
c.list_all_member()