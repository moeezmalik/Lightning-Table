evaluation-complete:
	clear
	python network/evaluate.py -t complete -p evaluation-data/complete/

evaluation-classification:
	clear
	python network/evaluate.py -t classification -p evaluation-data/classification/

evaluation-detection:
	clear
	python network/evaluate.py -t detection -p evaluation-data/detection/