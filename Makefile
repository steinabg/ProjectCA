default:
	python setup.py build_ext --inplace

setup:
	mkdir -p Data
	mkdir -p Config
	mkdir -p Bathymetry
	touch ./Config/configs.txt
	python setup.py build_ext --inplace