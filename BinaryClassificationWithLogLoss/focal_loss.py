import numpy as np


def focal_loss_with_class_weight(y_true , y_pred , alpha=0.25 ,gamma=2.0 ):
    """
     compute the focal loss for a binary classification problem

    """
    return alpha*-(1-y_pred)**gamma*y_true*np.log(y_pred) - \
        (1-alpha)*(1-y_true)*y_pred**gamma*np.log(1-y_pred)