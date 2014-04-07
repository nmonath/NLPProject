#!/bin/bash
echo "setting the classpath to clearnlp"
export CLEARNLP=tools/clearnlp/clearnlp-lib-2.0.2.1
export CLASSPATH=$CLEARNLP/args4j-2.0.23.jar:$CLEARNLP/guava-14.0.1.jar:$CLEARNLP/hppc-0.5.2.jar:$CLEARNLP/jregex1.2_01.jar:$CLEARNLP/log4j-1.2.17.jar:$CLEARNLP/clearnlp-2.0.2.jar:.
echo "classpath has been set"

echo "testing clearnlp"
java com.clearnlp.run.Version
echo "end of testing"

echo "testing our code"
python TestCode.py
echo "end testing our code"