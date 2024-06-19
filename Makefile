# Define the Python interpreter
PYTHON := python3

# packages to install
install:
	$(PYTHON) -m pip install pandas;



# Define the target for running the script
run:
	$(PYTHON) main.py