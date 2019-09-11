import os
import yaml

# General
random_seed = 42

class Config(object):
    def __init__(self):

        self.weights_save_location = "/tmp/" # must have trailing slash
        self.tensorboard_location = "/tmp/logs"

        self.images_location = "/hdd/datasets/TallTimbers/Images"
        self.labels_location = "/hdd/datasets/TallTimbers/Labels1"


        # Hyperparams (NOT USED FOR SEG MODEL)
        self.train_test_split = 0.1
        self.augment_ratio = 0.5
        self.image_size = (224, 224)
        self.input_shape = (self.image_size[0], self.image_size[1], 3) 
        self.samples_per_epoch = 200
        self.total_epoch = 1000
        self.batch_size = 16
        self.test_batch_size = 16
        self.learning_rate = 3.5e-4
        self.dropout = 0.4
        self.l2_constant = 1e-3
        self.mobilenet_alpha = 1.0
        self.include_softmax = True
        self.init_weights = None

        self.training_log_location = os.path.join(self.weights_save_location, "training_log.csv")
        self.config_save_location = os.path.join(self.weights_save_location, "train_config.yaml")
        self.model_vis_save_location = os.path.join(self.weights_save_location, "model.png")


    def serialize(self):
        return yaml.dump(self.__dict__)
