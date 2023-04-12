import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import jax.numpy as jnp
from typing import Optional, Tuple, Dict
from typing import Dict
import sys
import numpy as np
from flwr.common import Weights
import pdb


def fit_round(rnd: int) -> Dict:
    #Send number of training rounds to client.
    return {"rnd": rnd}


#def evaluate_round(rnd: int) -> Dict:
    #return {"rnd":rnd}
 #   return rnd

def get_eval_fn(model: LogisticRegression):
    # Return an evaluation function for server-side evaluation.
    #pdb.set_trace()

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_data_client1()

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        print("inside server evaluate")
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        preds = model.predict_proba(X_test)
        #print("preds:",preds)
        loss = log_loss(y_test, preds)
        print("y_test:",y_test)
        #loss_int=int(loss)
        #print("loss_int: ",loss_int)
        accuracy = model.score(X_test, y_test)
        accuracy_int = float(accuracy)
        #print(accuracy_int.dtype)
        res = pd.DataFrame(preds)
        res.index = pd.DataFrame(X_test).index # it's important for comparison
        res.columns = ["prediction", 'real']
        res.to_csv("prediction_results.csv")
        return {"Aggregated Results: loss ":loss}, accuracy_int

    return evaluate

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
      batches) during rounds one to three, then increase to ten local
      evaluation steps.

      """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

# Start Flower server for ten rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression(
        solver= 'saga',
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        #on_evaluate_config_fn= evaluate_round(rnd),
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address = "localhost:8080",
        strategy=strategy,
        config={"num_rounds": 5},
        grpc_max_message_length = 512 * 1024 * 1024,
    )
