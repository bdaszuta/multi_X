#!/bin/sh

######################################################
# Automatically run unit tests associated with project
#
# Requires:
#  inotify-tools
#
# Usage:
#  Run from base of repository
######################################################
MONITOR_DIRECTORY=$PWD                             # just use the current dir
BASE_FILENAME=$PWD/multi_SWSH/tests/run_tests.py   # file containing unit tests
TESTING_DIR=$PWD/multi_SWSH/tests
PYTHONPATH=$PWD                                    # required to locate files

echo "===="
echo "Monitoring $MONITOR_DIRECTORY"
echo "Automatically running unit tests with: $BASE_FILENAME"
echo "PYTHONPATH=$PYTHONPATH"
echo "TESTING_DIR=$TESTING_DIR"
echo "===="

export PYTHONPATH

# main loop here
while true
do
    inotifywait --timefmt '%d/%m/%y %H:%M:%S' --format '%T %w %f' \
                --exclude '.*(\.pyc|~|\.sh|\.py#)|(.*#.*)' \
                --quiet \
	            --recursive \
	            -r -e close_write ${MONITOR_DIRECTORY} \
	| while read date time dir file; do
	STATUS="Modification observed @ ${time} on ${date}..."
	echo "$STATUS"
	sleep 1  # allow oper. completion

        # run unit tests
        # cd $TESTING_DIR
        pytest
	# python $BASE_FILENAME
        # cd ..
    done
done
