import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import optuna

class Focal_Binary_Loss:
    def __init__(self, gamma_indct=2.0, alpha=0.25):
        self.gamma_indct = gamma_indct
        self.alpha = alpha

    def robust_pow(self, num_base, num_pow):
        return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

    def focal_binary_object(self, pred, label):
        gamma_indct = self.gamma_indct
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        g1 = sigmoid_pred * (1 - sigmoid_pred)
        g2 = label + ((-1) ** label) * sigmoid_pred
        g3 = sigmoid_pred + label - 1
        g4 = 1 - label - ((-1) ** label) * sigmoid_pred
        g5 = label + ((-1) ** label) * sigmoid_pred

        grad = (self.alpha * (
            gamma_indct * g3 * self.robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) +
            ((-1) ** label) * self.robust_pow(g5, (gamma_indct + 1))
        ))

        hess_1 = self.robust_pow(g2, gamma_indct) + gamma_indct * ((-1) ** label) * g3 * self.robust_pow(g2, (gamma_indct - 1))
        hess_2 = ((-1) ** label) * g3 * self.robust_pow(g2, gamma_indct) / g4
        
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct + (gamma_indct + 1) * self.robust_pow(g5, gamma_indct)) * g1

        return grad, hess

class CustomXGBClassifier:
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, **params):
        self.params = params
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.model = None
        self.classes_ = np.array([0, 1])  # Assuming binary classification

    def fit(self, X, y, eval_set=None, verbose=True):
        dtrain = xgb.DMatrix(X, label=y)
        evals = []
        if eval_set:
            for i, (X_val, y_val) in enumerate(eval_set):
                evals.append((xgb.DMatrix(X_val, label=y_val), f'eval{i}'))
        
        def focal_loss_obj(preds, dtrain):
            labels = dtrain.get_label()
            focal_loss = Focal_Binary_Loss(gamma_indct=self.focal_gamma, alpha=self.focal_alpha)
            grad, hess = focal_loss.focal_binary_object(preds, labels)
            return grad, hess

        self.model = xgb.train(
            self.params, dtrain, num_boost_round=100, evals=evals,
            obj=focal_loss_obj, verbose_eval=verbose
        )

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        proba = self.model.predict(dtest)
        return np.vstack((1 - proba, proba)).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def get_params(self, deep=True):
        return self.params
