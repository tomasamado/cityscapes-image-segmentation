""" 
    evaluation module 
"""

import sklearn.metrics as sk_metrics


class BinaryEvaluation(Object):
    """ Binary Evaluation """
    
    def __init__(self, confusion_matrix):
        self.cm = confusion_matrix        
    
    
    # Factory methods 
    @classmethod
    def from_bin_conf_matrix(confusion_matrix):
        """ Create a BinaryEvaluation object given the confussion matrix
        
            Parameters:
                confusion_matrix (np.array) - Binary confusion matrix"""
        return cls(confusion_matrix)
    
    @classmethod
    def from_multilabel_predictions(y_true, y_pred, target_class):
        """ Create a BinaryEvaluation object given the groud-truth labels and the 
            predicted labels. 
        
            Parameters:
                y_true (array-like of shape) - Ground truth (correct) label values
                y_pred (array-like of shape) - Predicted label values
                target_class (int) - the positive class to compare vs. the rest 

            Returs:
                (np.array) - Confusion matrix 2x2
        """
        true_list = torch.zeros(0, dtype=torch.long, device='cpu')
        pred_list = torch.zeros(0, dtype=torch.long, device='cpu')

        for i in range(len(y_true)):
            true_list = torch.cat([true_list, 
                                   (y_true[i] == target_class).long().view(-1).cpu()])
            pred_list = torch.cat([pred_list, 
                                   (y_pred[i] == target_class).long().view(-1).cpu()])    
        cm = sk_metrics.confusion_matrix(true_list.numpy(), pred_list.numpy())
        return cls(cm)
    
    @classmethod
    def from_model(y_true, model, target_class):
        return None
    
    
    # Evaluation metrics 
    def accuracy(self):
        """ Compute the accuracy """
        TN, FP, FN, TP = self.cm.ravel()
        return ((TP + TN) / (TP + TN + FP + FN))
    
    def sensitivity(self):
        """ Compute the sensitivity """
        TN, FP, FN, TP = self.cm.ravel()
        return (TP / (TP + FN))
    
    def specificity(self):
        """ Compute the specificity """
        TN, FP, FN, TP = self.cm.ravel()
        return (TN / (TN+FP))
    
    def dice_coeff(self):
        """ Compute Dice Coefficient """
        TN, FP, FN, TP = self.cm.ravel()
        return (2*TP / (2*TP + FP + FN))
    
    def jaccard_similarity(self):
        """ Compute Jaccard Similarity """
        TN, FP, FN, TP = self.cm.ravel()
        return (TP / (TP + FP + FN))
    
    def f1_score(self):
        """ Compute the F1-score """
        TN, FP, FN, TP = self.cm.ravel()
        return (TP / (TP + 0.5*(FP + FN)))
