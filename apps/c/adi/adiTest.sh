#!/bin/bash

#exit if some variables are not initialized
set -o nounset
#exit on error
set -o errexit

#change the max error
EPSILON=0.0000001


#functions --------------------------
BLACK="\e[30m"
RED="\e[31m"
GREEN="\e[32m"
YELLOW="\e[33m"
BLUE="\e[34m"
PURPLE="\e[35m"
DEFAULT="\e[39m"

printColored() {
	echo -e "$1$2$DEFAULT"
}

Indent() {
	 echo "$1" | sed -e 's/^/    /'
}

Exiting() {
	Indent "$(printColored $RED "EXITING . . .")"
	exit 1
}

function pause(){
   read -p "$*"
}

FileExistsOrDie(){

	FILE="$1"
	#check if the executable exists (opencal not built? was BUILD_OPENCAL_TESTS cmake option used?
	INFO_FILE=`file $FILE`
	DATA_FILE=`date -r $FILE`
	NUM_ROWS=`wc -l < $FILE`
	NUM_COLS=`head -n 1 $FILE | grep -o " " | wc -l`
	printColored $YELLOW "$INFO_FILE"
	Indent "$(printColored $YELLOW  " $NUM_ROWS rows,$NUM_COLS columns - $DATA_FILE")"

	if [ ! -f "$FILE" ] ; then
		printColored $RED "FATAL ERROR- FILE $FILE does not exists"
		printColored $RED $OUT
		
		Exiting
	fi
}

Md5CumulativeTest() {
	#md5sum CUMULATIVE
	ORIGINAL="$1"
	OPS="$2"
	#check if two output files exists
	FileExistsOrDie "$ORIGINAL"
	FileExistsOrDie "$OPS"

	MD5ORIGINAL="$(cat "$ORIGINAL"   | md5sum)"
	MD5OPS="$(cat "$OPS"	   | md5sum)"

	RES="OK"
	if [[ "$MD5ORIGINAL" != "$MD5OPS" ]]; then
		Indent "$(Indent "$(printColored $RED "MD5 CUMULATIVE FAILED.")")"
		#Exiting
	else
		Indent "$(Indent "$(printColored $GREEN "MD5 CUMULATIVE OK.")")"
	fi

}


CompareMatrixEpsilon() {
	ORIGINAL="$1"
	OPS="$2"
	base=`basename $ORIGINAL`
	OUT="$(perl compareMatrix.pl $ORIGINAL $OPS $EPSILON" ")"
	ISOK="$(echo $OUT | cut -c 1,2)"
	if [[ "$ISOK" == "OK" ]]; then
		Indent "$(Indent "$(printColored $GREEN "NUMERICAL COMPARISON(e=$EPSILON): $base $OUT")")"
	else
		Indent "$(Indent "$(printColored $RED "NUMERICAL COMPARISON(e=$EPSILON): $base $OUT")")"
	fi
}

ORIGINAL="$1"
OPS="$2"



Md5CumulativeTest $ORIGINAL $OPS

CompareMatrixEpsilon $ORIGINAL $OPS