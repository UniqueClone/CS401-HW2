run:
	python3 main.py

setup: requirements.txt
	pip install -r requirements.txt

train:
	python3 train.py
	python3 main.py
