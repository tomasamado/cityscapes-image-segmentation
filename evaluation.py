""" 
    evaluation module 
"""

import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix as ml_conf_m


class EvaluationReport:
    """ Evaluation report for a multiclass pixel-level segmentation task """
    
    def __init__(self, confusion_matrix, labels):
        self.cm = confusion_matrix        
        self.labels = np.array(labels)
        self.n_classes = len(labels)
        self.decimal_places = 4

    
    # Factory methods     
    @classmethod
    def from_predictions(cls, y_true, y_pred, labels):
        """ Create a MetricsReport object given the ground-truth labels and the 
            predicted labels. 
        
            Parameters:
                y_true (array-like of shape) - Ground truth (correct) label values
                y_pred (array-like of shape) - Predicted label values
                labels (array-like of shape) - Possible pixel labels 

            Returs:
                (EvaluationReport) - A evaluation report
        """
        true_list = torch.zeros(0, dtype=torch.long, device='cpu')
        pred_list = torch.zeros(0, dtype=torch.long, device='cpu')

        for i in range(len(y_true)):
            true_list = torch.cat([true_list, y_true[i].view(-1).cpu()])
            pred_list = torch.cat([pred_list, y_pred[i].view(-1).cpu()])    
        cm = ml_conf_m(true_list.numpy(), pred_list.numpy(), labels=labels)
        return cls(cm, labels)

    @classmethod
    def from_model(cls, dataloader, model, labels):
        """ Create a MetricsReport object given a dataloader and the model to
             make the predictions.

            Parameters:
                dataloader (torch.utils.data.Dataloader) - Dataloader
                model (torch.nn.Module) - Model to make the predictions
                labels (array-like of shape) - Possible pixel labels

            Returs:
                (EvaluationReport) - A evaluation report
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        true_list = torch.zeros(0, dtype=torch.long, device='cpu')
        pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
        model = model.to(device)

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                true_list = torch.cat([true_list, labels.view(-1).cpu()])
                pred_list = torch.cat([pred_list, preds.view(-1).cpu()])  

        cm = ml_conf_m(true_list.numpy(), pred_list.numpy(), labels=labels)
        return cls(cm, labels)
    
    @classmethod
    def from_model(cls, inputs, y_true, model, labels):
        """ Create a MetricsReport object given the ground-truth labels and the 
            predicted labels. 
        
            Parameters:
                inputs (array-like of shape) - Inputs
                y_true (array-like of shape) - Ground truth (correct) label values
                model (torch.nn.Module) - Model to make the predictions
                labels (array-like of shape) - Possible pixel labels

            Returs:
                (EvaluationReport) - A evaluation report
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        true_list = torch.zeros(0, dtype=torch.long, device='cpu')
        pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
        model = model.to(device)
        
        with torch.no_grad():
            for i, x in enumerate(inputs):
                x = x.to(device)
                output = model(x)
                _, pred = torch.max(output, 1)
                true_list = torch.cat([true_list, y_true[i].view(-1).cpu()])
                pred_list = torch.cat([pred_list, pred.view(-1).cpu()])  
                
        cm = ml_conf_m(true_list.numpy(), pred_list.numpy(), labels=labels)
        return cls(cm, labels)
    
    
    # Evaluation metrics 
    def all_metrics(self, target_class): 
         """ Compute all metrics for a target class in a 
            one vs. rest approach.
            
            Parameters:
                target_class (int) - Target class
                
            Returns: 
                (float) - Accuracy
        """
        return {
            'accuracy': self.accuracy(target_class),
            'sensitivity': self.sensitivity(target_class),
            'specificity': self.specificity(target_class),
            'dice_coeff': self.dice_coeff(target_class),
            'jaccard_sim': self.jaccard_similarity(target_class),
            'f1_score': self.f1_score(target_class)
        }
        
    def accuracy(self, target_class):
        """ Compute the accuracy for a target class in a 
            one vs. rest approach.
            
            Parameters:
                target_class (int) - Target class
                
            Returns: 
                (float) - Accuracy
        """
        assert any(np.isin(self.labels, target_class)), "unknown target class"
        index = np.where(self.labels == target_class)[0][0]
        TN, FP, FN, TP = self.cm[index].ravel()
        acc = (TP + TN) / (TP + TN + FP + FN)
        return round(acc, self.decimal_places)
    
    def sensitivity(self, target_class):
        """ Compute the sensitivity for a target class in a 
            one vs. rest approach.
            
            Parameters:
                target_class (int) - Target class
                
            Returns: 
                (float) - Accuracy
        """
        assert any(np.isin(self.labels, target_class)), "unknown target class"
        index = np.where(self.labels == target_class)[0][0]
        TN, FP, FN, TP = self.cm[index].ravel()
        sens = (TP / (TP + FN))
        return round(sens, self.decimal_places)
    
    
    def specificity(self, target_class):
        """ Compute the specificity for a target class in a 
            one vs. rest approach.
            
            Parameters:
                target_class (int) - Target class
                
            Returns: 
                (float) - Accuracy
        """
        assert any(np.isin(self.labels, target_class)), "unknown target class"
        index = np.where(self.labels == target_class)[0][0]
        TN, FP, FN, TP = self.cm[index].ravel()
        spec = (TN / (TN+FP))
        return round(spec, self.decimal_places)
    
    
    def dice_coeff(self, target_class):
        """ Compute the Dice's coefficient for a target class in a 
            one vs. rest approach.
            
            Parameters:
                target_class (int) - Target class
                
            Returns: 
                (float) - Accuracy
        """
        assert any(np.isin(self.labels, target_class)), "unknown target class"
        index = np.where(self.labels == target_class)[0][0]
        TN, FP, FN, TP = self.cm[index].ravel()
        dc =  (2*TP / (2*TP + FP + FN))
        return round(dc, self.decimal_places)
    
    def jaccard_similarity(self, target_class):
        """ Compute the Jaccards Similarity for a target class in a 
            one vs. rest approach.
            
            Parameters:
                target_class (int) - Target class
                
            Returns: 
                (float) - Accuracy
        """
        assert any(np.isin(self.labels, target_class)), "unknown target class"
        index = np.where(self.labels == target_class)[0][0]
        TN, FP, FN, TP = self.cm[index].ravel()
        js = (TP / (TP + FP + FN))
        return round(js, self.decimal_places)
    
    def f1_score(self, target_class):
        """ Compute the F1-score for a target class in a 
            one vs. rest approach.
            
            Parameters:
                target_class (int) - Target class
                
            Returns: 
                (float) - Accuracy
        """
        assert any(np.isin(self.labels, target_class)), "unknown target class"
        index = np.where(self.labels == target_class)[0][0]
        TN, FP, FN, TP = self.cm[index].ravel()
        f1 =  (TP / (TP + 0.5*(FP + FN)))
        return round(f1, self.decimal_places)
    
