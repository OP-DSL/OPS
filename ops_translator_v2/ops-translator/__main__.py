import os
import json

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from target import Target
from util import getVersion

def main(argv=None) -> None:

    #Build arg parser
    parser = ArgumentParser(prog="ops-translator")

    #argument declariations
    parser.add_argument("-V", "--version", help="Version", action="version", version=getVersion()) #this needs version tag
    parser.add_argument("-v", "--verbose", help="Verbose", action="store_true")
    parser.add_argument("-d", "--dump", help="JSON sotre dump", action="store_true")
    parser.add_argument("-o", "--out", help="Output directory", type=isDirPath)
    parser.add_argument("-c", "--config", help="Target configuration", type=json.loads, default="{}")
    parser.add_argument("-soa", "--force_soa", help="Force structs of arrays", action="store_true")

    parser.add_argument("--suffix", help="Add a suffix to genreated program translations", default="")

    parser.add_argument("-I", help="Add to include directories", type=isDirPath, action="append", nargs=1, default=[])
    parser.add_argument("-i", help="Add to include files", type=isFilePath, action="append", nargs=1, default=[])
    parser.add_argument("-D", help="Add to preprocessor defines", action="append", nargs=1, default=[])

    parser.add_argument("--file_paths", help="Input OPS sources", type="isFilePath", nargs="+")

    target_names = [taget.name for target in Target.all()] #TODO: implement Target Findable class
    parser.add_argument("-t", "--target", help="Code-gereration target", type=str, action="append", nargs=1, choices=target_names, default=[])


def isDirPath(path):
    if os.path.isdir(path):
        return path
    else:
        raise ArgumentTypeError("Invalid directory path: {path}")

def isFilePath(path):
    if os.path.isfile(path):
        return path
    else:
        raise ArgumentTypeError("Invalid file: {path}")

    

if __name__ == "__main__":
    main()