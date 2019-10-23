init:
	# Global install with caching
	sudo -H pip3 install -r requirements.txt

test:
	# -x --pdb: drop to PDB on first failure, then end test session
	# -q: less verbose output
	python3 -m pytest -q -x --pdb

testNoDebug:
	# -q: less verbose output
	python3 -m pytest -q

.PHONY: init test