init:
	# Global install with caching
	sudo -H pip3 install -r requirements.txt
	sudo -H pip3 install -e gym-watershed

updateEnv:
	# Global install with caching
	pip3.7 install --editable gym-watershed

test:
	# -x --pdb: drop to PDB on first failure, then end test session
	# -q: less verbose output
	python3 -m pytest -q -x --pdb

testVerbose:
	# -x --pdb: drop to PDB on first failure, then end test session
	python3 -m pytest -x --pdb

testNoDebug:
	# -q: less verbose output
	python3 -m pytest -q

testNoDebugVerbose:
	# -q: less verbose output
	python3 -m pytest

push:
	git push origin master
	git push gitlab master
	git push github master

docs:
	pdoc3 --html src --output-dir docs --force
	pdoc3 --html gym_watershed --output-dir gym-watershed/docs --force

.PHONY: init test docs