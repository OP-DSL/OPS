#!/bin/sh
#
# Call precommit hook scripts.
$OPS_INSTALL_PATH/../scripts/pre-commit-white_spaces.sh
rc=$?;if [ $rc != 0 ]; then echo "White spaces Found - Aborting Commit";exit $rc; fi;
$OPS_INSTALL_PATH/../scripts/pre-commit-check_formatting.sh
