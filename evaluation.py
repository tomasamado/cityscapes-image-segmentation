# -*- coding: utf-8 -*-
""" 
    evaluation module 
"""

import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix as ml_conf_m


class EvaluationReport:
    """ Evaluation report for a multiclass pixel-level segmentation task """
    
    averages = np.array(["binary", "macro", "weighted"])
    metrics = np.array(["accuracy", "sensitivity", "specificity", 
                           "dice_coeff", "jaccard_sim", "f1_score"])
    
    def __init__(self, confusion_matrix, labels, weights=None):
        self.cm = confusion_matrix        
        self.labels = np.array(labels)
        self.n_classes = len(labels)
        self.decimal_places = 4
        self.weights = weights

    
    # Factory methods     
    @classmethod
    def from_predictions(cls, y_true, y_pred, labels, weights=None):
        """ Create a MetricsReport object given the ground-truth labels and the 
            predicted labels. 
        
            Parameters:
                y_true (array-like of shape) - Ground truth (correct) label values
                y_pred (array-like of shape) - Predicted label values
                labels (array-like of shape) - Possible pixel labels 
                weights (array-like of shape, optional) - Weights of the different classes

            Returs:
                (EvaluationReport) - A evaluation report
        """
        true_list = torch.zeros(0, dtype=torch.long, device='cpu')
        pred_list = torch.zeros(0, dtype=torch.long, device='cpu')

        for i in range(len(y_true)):
            true_list = torch.cat([true_list, y_true[i].view(-1).cpu()])
            pred_list = torch.cat([pred_list, y_pred[i].view(-1).cpu()])    
        cm = ml_conf_m(true_list.numpy(), pred_list.numpy(), labels=labels)
        return cls(cm, labels, weights)

    @classmethod
    def from_model1(cls, dataloader, model, labels, weights=None):
        """ Create a MetricsReport object given a dataloader and the model to
             make the predictions.

            Parameters:
                dataloader (torch.utils.data.Dataloader) - Dataloader
                model (torch.nn.Module) - Model to make the predictions
                labels (array-like of shape) - Possible pixel labels
                weights (array-like of shape, optional) - Weights of the different classes

            Returs:
                (EvaluationReport) - A evaluation report
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        true_list = []
        pred_list = []
        model = model.to(device)

        with torch.no_grad():
            i=0
            for inputs, ground_truths in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                pred_list.append(torch.argmax(outputs.detach().cpu(),dim = 1))
                true_list.append(ground_truths)
                i+= 1
                if i == 4:
                    break
        
        true_list = torch.flatten(torch.cat(true_list))
        pred_list = torch.flatten(torch.cat(pred_list))
    
        cm = ml_conf_m(true_list.numpy(), pred_list.numpy(), labels=labels)
        return cls(cm, labels, weights)
    
    @classmethod
    def from_model2(cls, inputs, y_true, model, labels):
        """ Create a MetricsReport object given the ground-truth labels and the 
            predicted labels. 
        
            Parameters:
                inputs (array-like of shape) - Inputs
                y_true (array-like of shape) - Ground truth (correct) label values
                model (torch.nn.Module) - Model to make the predictions
                labels (array-like of shape) - Possible pixel labels
                weights (array-like of shape, optional) - Weights of the different classes

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
    def all_metrics(self, pos_label=1, average="binary"):
        """
            Compute all metrics.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - All metrics
        """
        indices, weights = self._general_metric(pos_label, average)
        return {
            'accuracy': self._accuracy(indices, weights),
            'sensitivity': self._sensitivity(indices, weights),
            'specificity': self._specificity(indices, weights),
            'dice_coeff': self._dice_coeff(indices, weights),
            'jaccard_sim': self._jaccard_similarity(indices, weights),
            'f1_score': self._f1_score(indices, weights)
        }
 
    def accuracy(self, pos_label=1, average="binary"):
        """
            Compute the accuracy.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Accuracy
        """
        indices, weights = self._general_metric(pos_label, average)
        return self._accuracy(indices, weights)
    
    def sensitivity(self, pos_label=1, average="binary"):
        """
            Compute the sensitivity.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Sensitivity
        """
        indices, weights = self._general_metric(pos_label, average)
        return self._sensitivity(indices, weights)    
    
    def specificity(self, pos_label=1, average="binary"):
        """
            Compute the specifity.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Specitifity
        """
        indices, weights = self._general_metric(pos_label, average)
        return self._specificity(indices, weights)
    
    
    def dice_coeff(self, pos_label=1, average="binary"):
        """
            Compute Dice's coefficient.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Dice's coefficient
        """
        indices, weights = self._general_metric(pos_label, average)
        return self._dice_coeff(indices, weights)
    
    def jaccard_similarity(self, pos_label=1, average="binary"):
        """
            Compute Jaccard similarity
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Jaccard similarity
        """
        indices, weights = self._general_metric(pos_label, average)
        return self._jaccard_similarity(indices, weights)
    
    def f1_score(self, pos_label=1, average="binary"):
        """
            Compute Jaccard similarity
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Jaccard similarity
        """
        indices, weights = self._general_metric(pos_label, average)
        return self._f1_score(indices, weights)
    
    def _general_metric(self, pos_label=1, average="binary"):
        """
            Obtain the necessary confusion matrix and weights to compute any metric.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (array-like, array-like) - Indices of the confusion matrices and weights
        """
        assert any(np.isin(EvaluationReport.averages, average)), "unkown 'average' method"
        assert any(np.isin(self.labels, pos_label)), "unknown target class"
        
        indices = []
        weights = []
        if average == "binary":
            index_label = np.where(self.labels == pos_label)[0][0]  
            indices.append(index_label)
            weights.append(1)
        else:
            for label in self.labels:
                index_label = np.where(self.labels == label)[0][0]  
                indices.append(index_label)
            if average == "macro":
                weights = np.ones(len(self.labels))
            else:
                assert self.weights is not None, "no weights have been provided"
                weights = self.weights
        return indices, weights
    
    def _accuracy(self, indices, weights):
        """
            Compute the accuracy.
            
            Parameters:
                indices (array-like shape) - The indices of the confusion matrix to use
                weights (array-like shape) - The weight for each class
                
            Returns:
                (float) - Accuracy
        """
        accuracies = []
        for i in indices:
            TN, FP, FN, TP = self.cm[i].ravel()
            acc_i = (TP + TN) / (TP + TN + FP + FN)
            accuracies.append(acc_i)
        acc = np.average(accuracies, weights=weights)
        return round(acc, self.decimal_places)
    
    def _sensitivity(self, indices, weights):
        """
            Compute the sensitivity.
            
            Parameters:
                indices (array-like shape) - The indices of the confusion matrix to use
                weights (array-like shape) - The weight for each class
                
            Returns:
                (float) - sensitivity
        """
        sensitivities = []
        for i in indices:
            TN, FP, FN, TP = self.cm[i].ravel()
            sens_i = (TP / (TP + FN))
            sensitivities.append(sens_i)
        sen = np.average(sensitivities, weights=weights)
        return round(sen, self.decimal_places)
    
    
    def _specificity(self, indices, weights):
        """
            Compute the specificity.
            
            Parameters:
                indices (array-like shape) - The indices of the confusion matrix to use
                weights (array-like shape) - The weight for each class
                
            Returns:
                (float) - Specificity
        """
        specificities = []
        for i in indices:
            TN, FP, FN, TP = self.cm[i].ravel()     
            spec_i = (TN / (TN+FP))
            specificities.append(spec_i)
        spec = np.average(specificities, weights=weights)
        return round(spec, self.decimal_places)
    
    
    def _dice_coeff(self, indices, weights):
        """
            Compute the Dice's coefficient.
            
            Parameters:
                indices (array-like shape) - The indices of the confusion matrix to use
                weights (array-like shape) - The weight for each class
                
            Returns:
                (float) - Dice's coefficient
        """
        dice_coeffs = []
        for i in indices:
            TN, FP, FN, TP = self.cm[i].ravel()   
            dc_i =  (2*TP / (2*TP + FP + FN))
            dice_coeffs.append(dc_i)
        dc = np.average(dice_coeffs, weights=weights)
        return round(dc, self.decimal_places)
    
    def _jaccard_similarity(self, indices, weights):
        """
            Compute the Jaccard similarity
            
            Parameters:
                indices (array-like shape) - The indices of the confusion matrix to use
                weights (array-like shape) - The weight for each class
                
            Returns:
                (float) - Jaccard similarity
        """
        jaccard_sims = []
        for i in indices:
            TN, FP, FN, TP = self.cm[i].ravel()   
            js_i = (TP / (TP + FP + FN))
            jaccard_sims.append(js_i)
        js = np.average(jaccard_sims, weights=weights)
        return round(js, self.decimal_places)
    
    def _f1_score(self, indices, weights):
        """
            Compute the F1-score
            
            Parameters:
                indices (array-like shape) - The indices of the confusion matrix to use
                weights (array-like shape) - The weight for each class
                
            Returns:
                (float) - F1-score
        """
        f1_scores = []
        for i in indices:
            TN, FP, FN, TP = self.cm[i].ravel()   
            f1_i =  (TP / (TP + 0.5*(FP + FN)))
            f1_scores.append(f1_i)
        f1 = np.average(f1_scores, weights=weights)   
        return round(f1, self.decimal_places) 



