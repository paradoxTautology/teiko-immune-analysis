.PHONY: setup pipeline dashboard

# Install all project dependencies.
setup:
	python3 -m pip install -r requirements.txt

# Build the database, load the data (Part 1), and generate all Part 2-4 outputs.
pipeline:
	python3 load_data.py
	python3 analysis.py

# Launch the interactive dashboard (local server).
dashboard:
	python3 -m streamlit run app.py
