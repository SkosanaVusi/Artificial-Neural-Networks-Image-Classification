install: venv
	source env/bin/activate; pip3 install -Ur requirements.txt
venv:
	test -d env || virtualenv -p python3 venv
clean:
	rm -rf venv
	find -iname "*.pyc" -delete