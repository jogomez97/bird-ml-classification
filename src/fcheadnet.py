from tensorflow.keras.layers import Dropout, Flatten, Dense


class FCHeadNet:
    @staticmethod
    def build(base_model, classes, D):
        """
        INPUT => FC => RELU => DO => FC => SOFTMAX
        @args:
            baseModel: body of the network
            classes: total number of classes in the dataset
            D: number of nodes in the fully-connected layer
        """
        # initialize the head model that will be placed on top of the base,
        # then add a FC layer
        head_model = base_model.output
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(D, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)

        # add a softmax layer
        head_model = Dense(classes, activation="softmax")(head_model)

        # return the model
        return head_model
