#!/usr/bin/env python3

import sys
import os
import numpy as np
import h5py

class Score:
    """Class used to compute scores
    After creating an instance of this class, call score_record() for
    each record being tested.  To calculate scores for a particular
    record, call record_auprc() and record_auroc().  After scoring all
    records, call gross_auprc() and gross_auroc() to obtain the scores
    for the database as a whole.
    """

    def __init__(self, input_digits=None):
        """Initialize a new scoring buffer.

        If 'input_digits' is given, it is the number of decimal digits
        of precision used in input probability values.
        """
        if input_digits is None:
            input_digits = 3
        self._scale = 10**input_digits
        self._pos_values = np.zeros(self._scale + 1, dtype=np.int64)
        self._neg_values = np.zeros(self._scale + 1, dtype=np.int64)
        self._record_auc = {}

    def score_record(self, truth, predictions, record_name=None):
        """Add results for a given record to the buffer.

        'truth' is a vector of arousal values: zero for non-arousal
        regions, positive for target arousal regions, and negative for
        unscored regions.

        'predictions' is a vector of probabilities produced by the
        classification algorithm being tested.  This vector must be
        the same length as 'truth', and each value must be between 0
        and 1.

        If 'record_name' is specified, it can be used to obtain
        per-record scores afterwards, by calling record_auroc() and
        record_auprc().
        """
        # Check if length is correct
        if len(predictions) != len(truth):
            raise ValueError("length of 'predictions' does not match 'truth'")

        # Compute the histogram of all input probabilities
        b = self._scale + 1
        r = (-0.5 / self._scale, 1.0 + 0.5 / self._scale)
        all_values = np.histogram(predictions, bins=b, range=r)[0]

        # Check if input contains any out-of-bounds or NaN values
        # (which are ignored by numpy.histogram)
        if np.sum(all_values) != len(predictions):
            raise ValueError("invalid values in 'predictions'")

        # Compute the histogram of probabilities within arousal regions
        pred_pos = predictions[truth > 0]
        pos_values = np.histogram(pred_pos, bins=b, range=r)[0]

        # Compute the histogram of probabilities in non-arousal regions,
        # given the above
        
        neg_values = all_values - pos_values

        self._pos_values += pos_values
        self._neg_values += neg_values

        if record_name is not None:
            self._record_auc[record_name] = self._auc(pos_values, neg_values)

    def _auc(self, pos_values, neg_values):
        # Calculate areas under the ROC and PR curves by iterating
        # over the possible threshold values.

        # At the minimum threshold value, all samples are classified as
        # positive, and thus TPR = 1 and TNR = 0.
        tp = np.sum(pos_values)
        fp = np.sum(neg_values)
        tn = fn = 0
        tpr = 1
        tnr = 0
        if tp == 0 or fp == 0:
            # If either class is empty, scores are undefined.
            return (float('nan'), float('nan'))
        ppv = float(tp) / (tp + fp)
        auroc = 0
        auprc = 0

        # As the threshold increases, TP decreases (and FN increases)
        # by pos_values[i], while TN increases (and FP decreases) by
        # neg_values[i].
        for (n_pos, n_neg) in zip(pos_values, neg_values):
            tp -= n_pos
            fn += n_pos
            fp -= n_neg
            tn += n_neg
            tpr_prev = tpr
            tnr_prev = tnr
            ppv_prev = ppv
            tpr = float(tp) / (tp + fn)
            tnr = float(tn) / (tn + fp)
            if tp + fp > 0:
                ppv = float(tp) / (tp + fp)
            else:
                ppv = ppv_prev
            auroc += (tpr_prev - tpr) * (tnr + tnr_prev) * 0.5
            auprc += (tpr_prev - tpr) * ppv_prev
        return (auroc, auprc)

    def gross_auroc(self):
        """Compute the area under the ROC curve.

        The result will be NaN if none of the records processed so far
        contained any target arousals.
        """
        return self._auc(self._pos_values, self._neg_values)[0]

    def gross_auprc(self):
        """Compute the area under the precision-recall curve.

        The result will be NaN if none of the records processed so far
        contained any target arousals.
        """
        return self._auc(self._pos_values, self._neg_values)[1]

    def record_auroc(self, record_name):
        """Compute the area under the ROC curve for a single record.

        The result will be NaN if the record did not contain any
        target arousals.

        The given record must have previously been processed by
        calling score_record().
        """
        return self._record_auc[record_name][0]

    def record_auprc(self, record_name):
        """Compute the area under the PR curve for a single record.

        The result will be NaN if the record did not contain any
        target arousals.

        The given record must have previously been processed by
        calling score_record().
        """
        return self._record_auc[record_name][1]