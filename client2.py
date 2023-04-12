import warnings
import flwr as fl
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils


if __name__ == "__main__":
     # Load dataset data from client1.csv
     (X_train, y_train), (X_test, y_test) = utils.load_data_client2()
     print("X_train :",X_train.shape)
     print("x_test :",X_test.shape)
     # Split train set into 10 partitions and randomly use one for training.
     partition_id = np.random.choice(10)
     (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]
     
     
     # Create a Logistic Regression Model
     model = LogisticRegression(
        solver= 'saga',
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
     
     # Setting initial parameters, akin to model.compile for keras models
     utils.set_initial_params(model)
     #print("model.coef_ :",model.coef_.shape)
     #coef_reshaped = np.transpose(model.coef_)
     #print("coef_reshaped: ",coef_reshaped.shape)
     # Define Flower client
     class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            print("inside client2 fit")
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            print("inside client1 evaluate")
            utils.set_model_params(model, parameters)
            preds = model.predict_proba(X_test)
            all_classes = {'1','0'}
            loss = log_loss(y_test, preds, labels=[1,0])
            #loss_int = int(loss)
            accuracy = model.score(X_test, y_test)
            accuracy_int = float(accuracy)
            return loss, len(X_test), accuracy_int
    # Start Flower client
     fl.client.start_numpy_client(server_address = "localhost:8080",client=FlowerClient(), grpc_max_message_length = 512 * 1024 * 1024)
