'''Serving decision threshold. Reviewed 2026-08-23: recall is prioritized over
precision for fraud detection (missing fraud costs more than reviewing false
positives). See train.py's optimal_threshold_diagnostic for the F1-optimal
 alternative (0.857) that was considered and rejected for this reason. '''

DECISION_THRESHOLD = 0.50
