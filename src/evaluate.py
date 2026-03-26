from sklearn.metrics import recall_score, make_scorer

specificity_scorer = make_scorer(recall_score, pos_label=0)

scoring = {
	'auc_pr' : 'average_precision',
	'auc_roc' : 'roc_auc',
	'recall' : 'recall',
	'specificity' : specificity_scorer
}
