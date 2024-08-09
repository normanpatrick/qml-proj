SHELL := /bin/bash

.PHONY: help

help:
	@echo "----------------------------------------------------"
	@echo "test-all          : run all test cases"
	@echo "process-all       : pre-process all images"
	@echo "----------------------------------------------------"

process-all:
	python pre_process.py

t1:
	TIGTEST=1 python test.py

t2:
	TIGTEST=2 python test.py

t3:
	TIGTEST=3 python test.py

t4:
	TIGTEST=4 python test.py

t5:
	TIGTEST=5 python test.py

t6:
	TIGTEST=6 python test.py

debug:
ifneq ("$(wildcard $(PATH_TO_FILE))","")
	export FILE_EXISTS=1
	@echo "File exists = ${FILE_EXISTS}"
else
	export FILE_EXISTS=0
	@echo "File exists = ${FILE_EXISTS}"
endif

# removed t3 since it overwrites data json file
test-all: t1 t2 t4 t5 t6
	@echo "You may have to restore the metadata json file"
	@echo "Done!"
