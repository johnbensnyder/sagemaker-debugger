#!/usr/bin/env bash

VERSION = '0.2.1'

for FRAMEWORK in tensorflow mxnet pytorch
do
    CAPITALIZED_FRAMEWORK=`echo "$FRAMEWORK" | tr '[a-z]' '[A-Z]'`
	TORNASOLE_WITH_$CAPITALIZED_FRAMEWORK=1 python setup.py bdist_wheel --universal
	# aws s3 cp dist/tornasole-$VERSION-py2.py3-none-any.whl s3://tornasole-binaries-use1/tornasole_$FRAMEWORK/py3/
done