.PHONY: all clean

all:
	pip3 install .

clean:
	rm -rf build dist *.egg-info
	rm -rf src/*.egg-info
