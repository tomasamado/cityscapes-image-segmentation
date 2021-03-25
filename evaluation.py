# -*- coding: utf-8 -*-
""" 
    evaluation module 
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix as cm_sklearn


class EvaluationReport:
    """ Evaluation report for a multiclass pixel-level segmentation task """
    
    averages = np.array(["binary", "macro", "weighted"])
    metrics = {"accuracy", "sensitivity", "specificity", "dice_coeff", 
                   "jaccard_sim", "precision", "recall", "f1_score"}
    
    def __init__(self, confusion_matrix, labels, weights=None):
        self._cm = confusion_matrix    
        self._decimal_places = 4    
        self._labels = np.array(labels)
        self._n_classes = len(labels)
        self._weights = weights
        
        if self._weights is None:
            total = 0
            self._weights = np.zeros(len(labels))
            for label in self._labels:
                c = np.where(self._labels == label)[0][0]
                nc = sum(self._cm[c,:])
                total += nc
                self._weights[c] = nc
            self._weights = self._weights / total
            
    @classmethod
    def from_predictions(cls, ground_truths, predictions, labels, weights=None):
        """ Create a MetricsReport object given the ground-truth labels and the 
            predicted labels. 
        
            Parameters:
                ground_truths (array-like of shape) - Ground-truth (correct) label values
                predictions (array-like of shape) - Predicted label values
                labels (array-like of shape) - Possible pixel labels 
                weights (array-like of shape, optional) - Weights of the different classes

            Returs:
                (EvaluationReport) - A evaluation report
        """  
        cm = cm_sklearn(ground_truths, predictions, labels=labels)
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
                pred_list.append(torch.argmax(outputs.detach().cpu(), dim = 1))
                true_list.append(ground_truths)
                
        true_list = torch.flatten(torch.cat(true_list))
        pred_list = torch.flatten(torch.cat(pred_list))
        cm = cm_sklearn(true_list.detach(), pred_list.detach(), labels=labels)
        
        return cls(cm, labels, weights)    
          
    @property
    def weights(self):
        return self._weights
    
    @property
    def decimal_places(self):
        return self._decimal_places
    
    @property
    def labels(self):
        return self._labels
    
    @weights.setter
    def weights(self, weights):
        assert len(weights) == len(self._labels), "invalid number of weights"
        self._weights = weights
        
    @decimal_places.setter
    def decimal_places(self, decimal_places):
        assert decimal_places >= 1, "the minimimum number of decimal places is 1"
        self._decimal_places = decimal_places
        
        
    @staticmethod
    def _TN(TN, FP, FN, TP):
        return TN
    
    @staticmethod
    def _FP(TN, FP, FN, TP):
        return FP
    
    @staticmethod
    def _FN(TN, FP, FN, TP):
        return FN
    
    @staticmethod
    def _TP(TN, FP, FN, TP):
        return TP
    
    @staticmethod
    def _accuracy(TN, FP, FN, TP):
        return (TP + TN) / (TP + TN + FP + FN + 1e-7)
    
    @staticmethod
    def _precision(TN, FP, FN, TP):
        return (TP / (TP + FP + 1e-7))
    
    @staticmethod
    def _recall(TN, FP, FN, TP):
        return (TP / (TP + FN + 1e-7))

    @staticmethod
    def _specificity(TN, FP, FN, TP):
        return (TN / (TN + FP + 1e-7))
    
    @staticmethod
    def _jaccard_similarity(TN, FP, FN, TP):
        return (TP / (TP + FP + FN + 1e-7))
    
    @staticmethod
    def _f1_score(TN, FP, FN, TP):
        return (2*TP / (2*TP + FP + FN + 1e-7))
    
    
    def _weighted_sum(self, values, weights, normalize=False):
        """ Compute a weighted sum """
        value = np.average(values, weights=weights)
        return round(value, self._decimal_places)
    
    def _confusion_matrix(self, c):
        """ Obtain the confusion matrix for class c """
        TP = self._cm[c, c]
        FN = sum(self._cm[c,:]) - self._cm[c,c]
        FP = sum(self._cm[:,c]) - self._cm[c,c]
        TN = sum(sum(self._cm)) - TP - FN - FP
        return np.array([[TN, FP], [FN, TP]])
    
    def _general_metric(self, metric, pos_label=1, average="binary"):
        """
            Obtain the necessary confusion matrix and weights to compute any metric.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (array-like, array-like) - Indices of the confusion matrices and weights
        """
        assert any(np.isin(EvaluationReport.averages, average)), "unknown 'average' method"
        assert any(np.isin(self._labels, pos_label)), "unknown target class '{}'".format(pos_label)
        
        indices = []
        weights = []
        scores = []
        
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
                weights = self._weights
                
        indices = np.array(indices)         
        for i in indices:
            TN, FP, FN, TP = self._confusion_matrix(i).ravel()
            score_i =  metric(TN, FP, FN, TP) 
            scores.append(score_i)
        return self._weighted_sum(scores, weights)
         
        
    def confusion_matrix(self, pos_label=1):
        """ Return the confusion matrix of a certain label.
        
            Parameters:
                pos_label(int) - 
                
            Returns:
                (numpy.array) - Confusion matrix
        """
        assert any(np.isin(self._labels, pos_label)), "unknown target class '{}'".format(pos_label)
        c = np.where(self._labels == pos_label)[0][0]  
        return self._confusion_matrix(c)
    
    
    def get_metrics(self, metrics="all", pos_label=1, average="binary"):
        """
            Compute a set of metrics.
            
            Parameters:
                metrics (any subset of {"all", "accuracy", "sensitivity", "specificity", 
                "dice_coeff", "jaccard_sim", "precision", "recall", "f1_score"'}, default="all") - 
                Metrics to be computed.                           
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’, ’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - All metrics
        """    
        assert metrics == "all" or set(metrics).issubset(EvaluationReport.metrics), "invalid list of metrics"
        report = dict()
        include_all =  (metrics == "all")

        metrics = np.array(metrics)
        if include_all or any(np.isin(metrics, "accuracy")):
            report["accuracy"] = self.accuracy(pos_label=pos_label, average=average)
        if include_all or any(np.isin(metrics, "sensitivity")):
            report["sensitivity"] = self.sensitivity(pos_label=pos_label, average=average)
        if include_all or any(np.isin(metrics, "specificity")):
            report["specificity"] = self.specificity(pos_label=pos_label, average=average)
        if include_all or any(np.isin(metrics, "dice_coeff")):
            report["dice_coeff"] = self.dice_coeff(pos_label=pos_label, average=average)
        if include_all or any(np.isin(metrics, "jaccard_sim")):
            report["jaccard_sim"] = self.jaccard_similarity(pos_label=pos_label, average=average)
        if include_all or any(np.isin(metrics, "precision")):
            report["precision"] = self.precision(pos_label=pos_label, average=average)
        if include_all or any(np.isin(metrics, "recall")):
            report["recall"] = self.recall(pos_label=pos_label, average=average)
        if include_all or any(np.isin(metrics, "f1_score")):
            report["f1_score"] = self.f1_score(pos_label=pos_label, average=average)

        return report
                
    def TN(self, pos_label=1, average="binary"):
        """ Compute the True Negatives.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - True Negatives
        """
        return self._general_metric(EvaluationReport._TN, pos_label, average)   

    def FP(self, pos_label=1, average="binary"):
        """ Compute the False Positives.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - False Positives
        """
        return self._general_metric(EvaluationReport._FP, pos_label, average)    

    def FN(self, pos_label=1, average="binary"):
        """ Compute the False Negatives.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - False Negatives
        """
        return self._general_metric(EvaluationReport._FN, pos_label, average)    

    def TP(self, pos_label=1, average="binary"):
        """ Compute the True Positives.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - True Positives
        """
        return self._general_metric(EvaluationReport._TP, pos_label, average)   

    def accuracy(self, pos_label=1, average="binary"):
        """ Compute the accuracy in a per-class basis.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Accuracy
        """
        return self._general_metric(EvaluationReport._accuracy, pos_label, average)   

    def precision(self, pos_label=1, average="binary"):
        """ Compute the precision.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - precision
        """
        return self._general_metric(EvaluationReport._precision, pos_label, average)   

    def recall(self, pos_label=1, average="binary"):
        """ Compute the Recall.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Recall
        """
        return self._general_metric(EvaluationReport._recall, pos_label, average)  

    def sensitivity(self, pos_label=1, average="binary"):
        """ Compute the sensitivity.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Sensitivity
        """
        return self._general_metric(EvaluationReport._recall, pos_label, average) 
    
    def specificity(self, pos_label=1, average="binary"):
        """ Compute the specifity.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Specitifity
        """
        return self._general_metric(EvaluationReport._specificity, pos_label, average)
    
    
    def dice_coeff(self, pos_label=1, average="binary"):
        """ Compute Dice's coefficient.
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Dice's coefficient
        """
        return self._general_metric(EvaluationReport._f1_score, pos_label, average)
    
    def jaccard_similarity(self, pos_label=1, average="binary"):
        """ Compute Jaccard similarity
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Jaccard similarity
        """
        return self._general_metric(EvaluationReport._jaccard_similarity, pos_label, average)
    
    def f1_score(self, pos_label=1, average="binary"):
        """ Compute Jaccard similarity
            
            Parameters:
                pos_label (str or int, default=1) - The class to report if average='binary' 
                and the data is binary.
                average ({‘macro’,’weighted’, ‘binary’}, default='binary') - 
                
            Returns:
                (float) - Jaccard similarity
        """
        return self._general_metric(EvaluationReport._f1_score, pos_label, average)
