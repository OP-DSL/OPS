import os
import json
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path

#custom implementation imports
from language import Lang
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

    parser.add_argument("--file_paths", help="Input OPS sources", type=isFilePath, nargs="+")

    target_names = [target.name for target in Target.all()] #TODO: implement Target Findable class
    parser.add_argument("-t", "--target", help="Code-gereration target", type=str, action="append", nargs=1, choices=target_names, default=[])

    #invoking arg parser
    args = parser.parse_args(argv)

    if os.environ.get("OPS_AUTO_SOA") is not None:
        args.force_soa = True

    file_parents = [Path(file_path).parent for file_path in args.file_paths]

    if args.out is None:
        args.out = file_parents[0]

    #checking includes of OPS
    if os.environ.get("OPS_INSTALL_PATH") is not None:
        ops_install_path = Path(os.environ.get("OPS_INSTALL_PATH"))
        args.I = [[str(ops_install_path/"include")]] + args.I
    else:
        script_parents = list(Path(__file__).resolve().parents)
        if len(script_parents) >= 3 and script_parents[2].stem == "OPS":
            args.I = [[str(script_parents[2].joinpath("ops/c/include"))]] + args.I

    args.I = [[str(file_parent)] for file_parent in file_parents] + args.I

    # Collect the set of ile extensions 
    extentions = {str(Path(file_path).suffix)[1:] for file_path in args.file_paths}

    if not extentions:
        exit("Missing file extensions, unable to determine target language.")
    elif len(extentions) > 1:
        exit("Varying file extensions, unable to determine target language.")
    else:
        [extention] = extentions

    lang = Lang.find(extention)

    if lang is None:
        exit(f"Unknown file extention: {extention}")

    
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