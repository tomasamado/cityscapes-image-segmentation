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
    metrics = {"accuracy", "sensitivity", "specificity", 
               "dice_coeff", "jaccard_sim", "f1_score"}
    
    def __init__(self, confusion_matrix, labels, weights=None):
        self._cm = confusion_matrix    
        self._weights = weights
        self._decimal_places = 4    
        self._labels = np.array(labels)
        self._n_classes = len(labels)
       
    
    # Factory methods     
    @classmethod
    def from_predictions(cls, ground_truths, predictions, labels, weights=None):
        """ Create a MetricsReport object given the ground-truth labels and the 
            predicted labels. 
        
            Parameters:
                ground_truths (array-like of shape) - Ground-truth (correct) label values
                predictions (array-like of shape) - Predicted label values
                labels (array-like of shape) - Possible pixel labels 
                weights (array-like of shape, optional) - Weights of the different 
                classes

            Returs:
                (EvaluationReport) - A evaluation report
        """  
        cm = ml_conf_m(ground_truths, predictions, labels=labels)
        return cls(cm, labels, weights)

    @classmethod
    def from_model(cls, dataloader, model, labels, weights=None):
        """ Create a MetricsReport object given a dataloader and the model to
             make the predictions.

            Parameters:
                dataloader (torch.utils.data.Dataloader) - Dataloader
                model (torch.nn.Module) - Model to make the predictions
                labels (array-like of shape) - Possible pixel labels
                weights (array-like of shape, optional) - Weights of the different 
                classes

            Returs:
                (EvaluationReport) - A evaluation report
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        true_list = []
        pred_list = []

        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            for inputs, ground_truths in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim = 1).detach().cpu()
                pred_list.append(preds)
                true_list.append(ground_truths)
        true_list = torch.flatten(torch.cat(true_list))
        pred_list = torch.flatten(torch.cat(pred_list))
        
        cm = ml_conf_m(true_list.detach().cpu().numpy(), pred_list.detach().cpu().numpy(), labels=labels)
        return cls(cm, labels, weights)    
        
    # Properties 
    @property
    def weights(self):
        return self._weights
    
    @property
    def decimal_places(self):
        return self._decimal_places
    
    @property
    def labels(self):
        return self._labels
    
    # Setters
    @weights.setter
    def weights(self, weights):
        assert len(weights) == len(self._labels), "invalid number of weights"
        self._weights = weights
        
    @decimal_places.setter
    def decimal_places(self, decimal_places):
        assert decimal_places >= 1, "the minimimum number of decimal places is 1"
        self._decimal_places = decimal_places
        
    
    # Evaluation metrics     
    def confusion_matrix(self, pos_label=1):
        """ Return the confusion matrix of a certain label.
        
            Parameters:
                pos_label(int) - 
                
            Returns:
                (numpy.array) - Confusion matrix
        """
        assert any(np.isin(self._labels, pos_label)), "unknown target class"
        index_label = np.where(self._labels == pos_label)[0][0]  
        return self._cm[index_label]
    
    def get_metrics(self, metrics="all", pos_label=1, average="binary"):
        """
            Compute a set of metrics.
            
            Parameters:
                metrics (any subset of {"all", "accuracy", "sensitivity", "specificity", 
                "dice_coeff", "jaccard_sim", "f1_score"'}, default="all") - Metrics to
                be computed.                           
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - All metrics
        """    
        assert metrics == "all" or set(metrics).issubset(EvaluationReport.metrics), "invalid list of metrics"
        indices, weights = self._general_metric(pos_label, average)
        report = dict()
        include_all =  (metrics == "all")

        metrics = np.array(metrics)
        if include_all or any(np.isin(metrics, "accuracy")):
            report["accuracy"] = self._accuracy(indices, weights)
        if include_all or any(np.isin(metrics, "sensitivity")):
            report["sensitivity"] = self._sensitivity(indices, weights)
        if include_all or any(np.isin(metrics, "specificity")):
            report["specificity"] = self._specificity(indices, weights)
        if include_all or any(np.isin(metrics, "dice_coeff")):
            report["dice_coeff"] = self._dice_coeff(indices, weights)
        if include_all or any(np.isin(metrics, "jaccard_sim")):
            report["jaccard_sim"] = self._jaccard_similarity(indices, weights)
        if include_all or any(np.isin(metrics, "f1_score")):
            report["f1_score"] = self._f1_score(indices, weights)

        return report
            
 
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
        assert any(np.isin(self._labels, pos_label)), "unknown target class"
        
        indices = []
        weights = []
        if average == "binary":
            index_label = np.where(self._labels == pos_label)[0][0]  
            indices.append(index_label)
            weights.append(1)
        else:
            for label in self._labels:
                index_label = np.where(self._labels == label)[0][0]  
                indices.append(index_label)
            if average == "macro":
                weights = np.ones(len(self._labels))
            else:
                assert self._weights is not None, "no weights have been provided"
                weights = self._weights
        return np.array(indices), np.array(weights)
    
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
            TN, FP, FN, TP = self._cm[i].ravel()
            acc_i = (TP + TN) / (TP + TN + FP + FN)
            accuracies.append(acc_i)
        acc = np.average(accuracies, weights=weights)
        return round(acc, self._decimal_places)
    
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
            TN, FP, FN, TP = self._cm[i].ravel()
            sens_i = (TP / (TP + FN))
            sensitivities.append(sens_i)
        sen = np.average(sensitivities, weights=weights)
        return round(sen, self._decimal_places)
    
    
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
            TN, FP, FN, TP = self._cm[i].ravel()
            spec_i = (TN / (TN+FP))
            specificities.append(spec_i)
        spec = np.average(specificities, weights=weights)
        return round(spec, self._decimal_places)
    
    
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
            TN, FP, FN, TP = self._cm[i].ravel() 
            dc_i =  (2*TP / (2*TP + FP + FN))
            dice_coeffs.append(dc_i)
        dc = np.average(dice_coeffs, weights=weights)
        return round(dc, self._decimal_places)
    
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
            TN, FP, FN, TP = self._cm[i].ravel()
            js_i = (TP / (TP + FP + FN))
            jaccard_sims.append(js_i)
        js = np.average(jaccard_sims, weights=weights)
        return round(js, self._decimal_places)
    
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
            TN, FP, FN, TP = self._cm[i].ravel() 
            f1_i =  (TP / (TP + 0.5*(FP + FN)))
            f1_scores.append(f1_i)
        f1 = np.average(f1_scores, weights=weights)   
        return round(f1, self._decimal_places) 



