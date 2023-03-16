evaluation-complete:
	clear
	python network/evaluate.py -t complete -p ../Evaluation-Data/complete/

evaluation-classification:
	clear
	python network/evaluate.py -t classification -p ../Evaluation-Data/classification/

evaluation-detection:
	clear
	python network/evaluate.py -t detection -p ../Evaluation-Data/detection/