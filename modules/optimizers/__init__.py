from config import Config
import tensorflow as tf

class Optimizers(Config):

    optimizer = None

    def __init__(self, name_opt=None, learning_rate=0.0002, beta_1=0.2, beta_2=0.9999, **kwargs):
        super().__init__()

        if name_opt == 'Adam':
            self.optimizer = self.__adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, **kwargs)
        elif name_opt == "AdamX":
            self.optimizer = self.__adamx(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, **kwargs)
        elif name_opt == "RMSprop":
            self.optimizer = self.__rsmprop(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, **kwargs)
        elif name_opt == "SGD":
            self.optimizer = self.__sgd(learning_rate=learning_rate, **kwargs)
        else:
            raise ValueError("Optimazer not found")

    def __adam(self, learning_rate, beta_1, beta_2, **kwargs):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, **kwargs)
    
    def __adamx(self, learning_rate, beta_1, beta_2, **kwargs):
        return tf.keras.optimizers.Adamax(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, **kwargs)

    def __rsmprop(self, learning_rate, beta_1, beta_2, **kwargs):
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, **kwargs)
    
    def __sgd(self, learning_rate, **kwargs):
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, **kwargs)
