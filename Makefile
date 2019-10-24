init:
	# Global install with caching
	sudo -H pip3 install -r requirements.txt
	sudo -H pip3 install -e gym-watershed

test:
	# -x --pdb: drop to PDB on first failure, then end test session
	# -q: less verbose output
	python3 -m pytest -q -x --pdb

testNoDebug:
	# -q: less verbose output
	python3 -m pytest -q

push:
	git push origin master
	git push gitlab master

.PHONY: init test