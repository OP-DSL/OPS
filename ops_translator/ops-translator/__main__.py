import dataclasses
import os
import json
import re
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import datetime
from pathlib import Path
import logging

#custom implementation imports
import cpp
import fortran

from jinja_utils import env
from language import Lang
from ops import OpsError, Type
from scheme import Scheme
from cpp.schemes import CppHLS
from store import Application, ParseError
from target import Target
from util import getVersion, safeFind

def main(argv=None) -> None:

    #Build arg parser
    parser = ArgumentParser(prog="ops-translator")

    #argument declariations
    parser.add_argument("-V", "--version", help="Version", action="version", version=getVersion()) #this needs version tag
    parser.add_argument("-v", "--verbose", help="Verbose", action="store_true")
    parser.add_argument("-g", "--debug", help="Debug", action="store_true")
    parser.add_argument("-f", "--logfile", help="Logfile Name", default="ops_translator_run.log")
    parser.add_argument("-d", "--dump", help="JSON store dump", action="store_true")
    parser.add_argument("-o", "--out", help="Output directory", type=isDirPath)
    parser.add_argument("-c", "--config", help="Target configuration", type=json.loads, default="{}")
    parser.add_argument("-soa", "--force_soa", help="Force structs of arrays", action="store_true")

    parser.add_argument("--suffix", help="Add a suffix to genreated program translations", default="_ops")

    parser.add_argument("-I", help="Add to include directories", type=isDirPath, action="append", nargs=1, default=[])
    parser.add_argument("-i", help="Add to include files", type=isFilePath, action="append", nargs=1, default=[])
    parser.add_argument("-D", help="Add to preprocessor defines", action="append", nargs=1, default=[])

    parser.add_argument("--file_paths", help="Input OPS sources", type=isFilePath, nargs="+")

    target_names = [target.name for target in Target.all()]
    parser.add_argument("-t", "--target", help="Code-gereration target", type=str, action="append", nargs=1, choices=target_names, default=[])

    parser.add_argument("-fpga", "--fpga", help="Generate program for FPGA vitis HLS", action="store_true")
    
    #invoking arg parser
    args = parser.parse_args(argv)

    #setting logger
    if (args.debug):
        logging.basicConfig(filename=args.logfile, level=logging.DEBUG)
    elif (args.verbose):
        logging.basicConfig(filename=args.logfile, level=logging.INFO)
    else:
        logging.basicConfig(filename=args.logfile, level=logging.WARNING)
    
    if os.environ.get("OPS_AUTO_SOA") is not None:
        args.force_soa = True
        logging.warning("OPS FORCE SOA set")
    else:
        logging.warning("OPS FORCE SOA not set")

    file_parents = [Path(file_path).parent for file_path in args.file_paths]

    if args.out is None:
        args.out = file_parents[0]
        logging.warning("output location is not set selecting default path: %s", str(args.out.resolve()))

    #checking includes of OPS
    if os.environ.get("OPS_INSTALL_PATH") is not None:
        ops_install_path = Path(os.environ.get("OPS_INSTALL_PATH"))
        args.I = [[str(ops_install_path/"include")]] + args.I
        logging.info("detected OPS_INSTALL_PATH: %s", str(ops_install_path.resolve()))
    else:
        script_parents = list(Path(__file__).resolve().parents)
        if len(script_parents) >= 3 and script_parents[2].stem == "OPS":
            ops_install_path = script_parents[2].joinpath("ops/c")
            logging.info("detected OPS_INSTALL_PATH: %s", str(ops_install_path.resolve()))
            args.I = [[str(ops_install_path/"include")]] + args.I

    args.I = [[str(file_parent)] for file_parent in file_parents] + args.I

    # Collect the set of file extensions 
    extensions = {str(Path(file_path).suffix)[1:] for file_path in args.file_paths}

    if not extensions:
        logging.error("Missing file extensions, unable to determine target language.")
        exit("Missing file extensions, unable to determine target language.")
    elif len(extensions) > 1:
        logging.error("Varying file extensions, unable to determine target language.")
        exit("Varying file extensions, unable to determine target language.")
    else:
        [extension] = extensions

    lang = Lang.find(extension)

    if lang is None:
        exit(f"Unknown file extension: {extension}")

    Type.set_formatter(lang.formatType)

    if len(args.target) == 0:
        args.target = [[target_name] for target_name in target_names]

    try:
        app = parse(args, lang)
    except ParseError as e:
        exit(e)

    # TODO: Make sure SOA is applicable to OPS
    # if args.force_soa:
    #     for program in app.programs:
    #         for loop in program.loops:
    #             loop.dats = [dataclasses.replace(dat, soa=True) for dat in loop.dats]

    if args.verbose:
        print()
        print(app)
        logging.debug("App: \n %s", str(app))

    # Validation phase
    try: 
        validate(args, lang, app)
    except OpsError as e:
        logging.error("parsed application validation failed with exception: %e", str(e))
        exit(e)

    # Generate program translations
    app_consts = app.consts()
    for i, program in enumerate(app.programs, 1):
        logging.info("Generating code for program: %s", str(program.path))
        include_dirs = set([Path(dir) for [dir] in args.I])
        defines = [define for [define] in args.D]

        if (args.fpga):
            logging.warning("only FPGA vitis HLS mode selected")
            source = lang.translateProgram(program, include_dirs, defines, app_consts, args.force_soa, True)
        else:   
            source = lang.translateProgram(program, include_dirs, defines, app_consts, args.force_soa)

        if not args.force_soa and program.soa_val:
            args.force_soa = program.soa_val

        new_file = os.path.splitext(os.path.basename(program.path))[0]
        ext = os.path.splitext(os.path.basename(program.path))[1]
        
        if (args.fpga):
            new_path = Path(args.out, f"{new_file}{args.suffix}_hls{ext}")
        else:
            new_path = Path(args.out, f"{new_file}{args.suffix}{ext}")
        logging.info("   writing generated code to: %s", str(new_path.resolve()))
        
        with open(new_path, "w") as new_file:
            new_file.write(f"\n{lang.com_delim} Auto-generated at {datetime.now()} by ops-translator\n")
            new_file.write(source)

            if args.verbose:
                print(f"Translated program {i} of {len(args.file_paths)}: {new_path}")
                logging.info(f"Translated program {i} of {len(args.file_paths)}: {new_path}")


    # Generating code for targets
    for [target] in args.target:
        target = Target.find(target)

        # Applying user defined configs to the target config
        for key in target.config:
            if key in args.config and key in target.config:
                target.config[key] = args.config[key]

        logging.info("Found target: %s", str(target))
        
        scheme = Scheme.find((lang, target))

        if not scheme:
            if args.verbose:
                print(f"No scheme register for {lang}/{target}")
                logging.warning(f"No scheme register for {lang}/{target}")
            continue

        if args.verbose:
            print(f"Translation scheme: {scheme}")
            logging.info(f"Translation scheme: {scheme}")
        

        codegen(args, scheme, app, target.config, args.force_soa)
        
        if target.name == "hls":
            codegenHLSDevice(args, scheme, app, target.config, args.force_soa)

        if args.verbose:
            print(f"Translation completed: {scheme}")


def parse(args: Namespace, lang: Lang) -> Application:
    app = Application()

    # Collect the include directories
    include_dirs = set([Path(dir) for [dir] in args.I])
    defines = [define for [define] in args.D]

    # Parse the input files
    for i, raw_path in enumerate(args.file_paths, 1):
        if args.verbose:
            print(f"Parseing file {i} of {len(args.file_paths)}: {raw_path}")

        # Parse the program
        program = lang.parseProgram(Path(raw_path), include_dirs, defines)
        app.programs.append(program)

    # for item in app.loops():
    #     loop = item[0]
    #     KernelEntity = app.findEntities(loop.kernel)
        
    #     print(f"Found Kernel Entity: {KernelEntity[0]}")
    return app

def validate(args: Namespace, lang: Lang, app: Application) -> None:
    # Run sementic checks on the application
    app.validate(Lang)

    if args.dump:
        store_path = Path(args.out, "store.json")
        serializer = lambda o: getattr(o, "__dict__", "unserializable")
        logging.info("Application dump enabled. Dumping file to: %s", store_path.resolve())
        
        # Write application dump
        with open(store_path, "w") as file:
            file.write(json.dumps(app, default=serializer, indent=4))

        if args.verbose:
            print("Dumped store: ", store_path.resolve(), end="\n\n")


def codegen(args: Namespace, scheme: Scheme, app: Application, target_config: dict, force_soa: bool = False) -> None:
    # Collect the paths of the generated files
    include_dirs = set([Path(dir) for [dir] in args.I])
    defines = [define for [define] in args.D]

    # Generate loop hosts
    for i, (loop, program) in enumerate(app.uniqueLoops(), 1):
        # Generate loop host source
        source, extension = scheme.genLoopHost(include_dirs, defines, env, loop, program, app, i, force_soa)

        new_source = re.sub(r'\n\s*\n', '\n\n', source)

        # From output files path
        path = None
        if scheme.lang.kernel_dir:
            if scheme.target.name == "hls":
                Path(args.out, scheme.target.name, "host", "kernel_wrappers").mkdir(parents=True, exist_ok=True)
                path = Path(args.out, scheme.target.name, "host", "kernel_wrappers", f"{loop.kernel}_kernel.{extension}")                
            else:
                Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)
                path = Path(args.out, scheme.target.name, f"{loop.kernel}_kernel.{extension}")
        else:
            if scheme.target.name == "hls":
                path = Path(args.out,f"{loop.kernel}_{scheme.target.name}_kernel_wrapper.{extension}")
            else:
                path = Path(args.out,f"{loop.kernel}_{scheme.target.name}_kernel.{extension}")

        # Write the gernerated source file
        with open(path, "w") as file:
            file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by ops-translator\n")
            file.write(new_source)

            if args.verbose:
                print(f"Generated loop host {i} of {len(app.uniqueLoops())}: {path}")

    # Gernerate master kernel file
    if scheme.master_kernel_template is not None:
        user_types_name = f"user_types.{scheme.lang.include_ext}"
        user_types_candidates = [Path(dir, user_types_name) for dir in include_dirs]
        user_types_file = safeFind(user_types_candidates, lambda p: p.is_file())

        source, name = scheme.genMasterKernel(env, app, user_types_file, target_config, force_soa)

        new_source = re.sub(r'\n\s*\n', '\n\n', source)

        path = None

        if scheme.lang.kernel_dir:
            if scheme.target.name == "hls":
                Path(args.out, scheme.target.name, "host", "kernel_wrappers").mkdir(parents=True, exist_ok=True)
                path = Path(args.out, scheme.target.name, "host", "kernel_wrappers", name)
            else:
                Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)
                path = Path(args.out, scheme.target.name, name)

        else:
            path = Path(args.out, name)

        with open(path, "w") as file:
            file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by ops-translator\n")
            file.write(new_source)

            if args.verbose:
                print(f"Generated master kernel file: {path}")
                

#TODO: Add a generic target flag to target class "kernel_device_translation"
def codegenHLSDevice(args: Namespace, scheme: Scheme, app: Application, target_config: dict, force_soa: bool = False) -> None:
    
    defines = [define for [define] in args.D]

    #Generate common_config
    source, extension = scheme.genConfigDevice(env, target_config)
    new_source = re.sub(r'\n\s*\n', '\n\n', source)
    
    # From output files path
    path = None
    if scheme.lang.kernel_dir:
        Path(args.out, scheme.target.name, "device", "include").mkdir(parents=True, exist_ok=True)
        path = Path(args.out, scheme.target.name, "device", "include", f"common_config.{extension}")                
    else:
        path = Path(args.out,f"{scheme.target.name}_common_config.{extension}")

    # Write the gernerated source file
    with open(path, "w") as file:
        file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by ops-translator\n")
        file.write(new_source)

        if args.verbose:
            print(f"Generated Device common_config.hpp")

    #Generate stencil device definitions
    for program in app.programs:
        for stencil in program.stencils:
            source, extension = scheme.genStencilDecl(env, target_config, stencil)
            new_source = re.sub(r'\n\s*\n', '\n\n', source)
            
            # From output files path
            path = None
            if scheme.lang.kernel_dir:
                Path(args.out, scheme.target.name, "device", "include").mkdir(parents=True, exist_ok=True)
                path = Path(args.out, scheme.target.name, "device", "include", f"stencil_{stencil.stencil_ptr}.{extension}")                
            else:
                path = Path(args.out,f"stencil_{scheme.target.name}_{stencil.stencil_ptr}.{extension}")

            # Write the gernerated source file
            with open(path, "w") as file:
                file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by ops-translator\n")
                file.write(new_source)

                if args.verbose:
                    print(f"Generated Device stencil_{stencil.stencil_ptr}.hpp")
                
    #Generate loop device
    #if scheme.target.name == "hls":
    for i, (loop, program) in enumerate(app.uniqueLoops(), 1):
        # Generate loop host source
        [(datamov_inc_source, datamov_inc_extension),
         (datamov_src_source, datamov_src_extension),
         (kernel_inc_source, kernel_inc_extension),
         (kernel_src_source, kernel_src_extension)] = scheme.genLoopDevice(env, loop, program, app, target_config, i)

        datamov_inc_source = re.sub(r'\n\s*\n', '\n\n', datamov_inc_source)
        datamov_src_source = re.sub(r'\n\s*\n', '\n\n', datamov_src_source)
        kernel_inc_source = re.sub(r'\n\s*\n', '\n\n', kernel_inc_source)
        kernel_src_source = re.sub(r'\n\s*\n', '\n\n', kernel_src_source)
        
        # datamover include
        path = None
        if scheme.lang.kernel_dir:
            Path(args.out, scheme.target.name, "device", "include").mkdir(parents=True, exist_ok=True)
            path = Path(args.out, scheme.target.name, "device", "include", f"datamover_{loop.kernel}.{datamov_inc_extension}")                
        else:
            path = Path(args.out,f"{loop.kernel}_{scheme.target.name}_datamover.{datamov_inc_extension}")

        logging.debug(f"writing datamover include for: {loop.kernel} to {path}")
        
        # Write the gernerated datamover include file
        with open(path, "w") as file:
            file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by ops-translator\n")
            file.write(datamov_inc_source)

            if args.verbose:
                print(f"Generated loop device datamover inclue {i} of {len(app.uniqueLoops())}: {path}")

        #datamover src
        path = None
        if scheme.lang.kernel_dir:
            Path(args.out, scheme.target.name, "device", "src").mkdir(parents=True, exist_ok=True)
            path = Path(args.out, scheme.target.name, "device", "src", f"datamover_{loop.kernel}.{datamov_src_extension}")                
        else:
            path = Path(args.out,f"{loop.kernel}_{scheme.target.name}_datamover.{datamov_src_extension}")

        logging.debug(f"writing datamover src for: {loop.kernel} to {path}")
        
        # Write the gernerated source file
        with open(path, "w") as file:
            file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by ops-translator\n")
            file.write(datamov_src_source)

            if args.verbose:
                print(f"Generated loop device datamover src {i} of {len(app.uniqueLoops())}: {path}")
                
        #kernel inc
        path = None
        if scheme.lang.kernel_dir:
            Path(args.out, scheme.target.name, "device", "include").mkdir(parents=True, exist_ok=True)
            path = Path(args.out, scheme.target.name, "device", "include", f"kernel_{loop.kernel}.{kernel_inc_extension}")                
        else:
            path = Path(args.out,f"{loop.kernel}_{scheme.target.name}_kernel.{kernel_inc_extension}")

        logging.debug(f"writing kernel: {loop.kernel} include to {path}")
        
        # Write the gernerated source file
        with open(path, "w") as file:
            file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by ops-translator\n")
            file.write(kernel_inc_source)

            if args.verbose:
                print(f"Generated loop device kernel include {i} of {len(app.uniqueLoops())}: {path}")
                
        #kernel src
        path = None
        if scheme.lang.kernel_dir:
            Path(args.out, scheme.target.name, "device", "src").mkdir(parents=True, exist_ok=True)
            path = Path(args.out, scheme.target.name, "device", "src", f"kernel_{loop.kernel}.{datamov_src_extension}")                
        else:
            path = Path(args.out,f"{loop.kernel}_{scheme.target.name}_kernel.{datamov_src_extension}")

        # Write the gernerated source file
        with open(path, "w") as file:
            file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by ops-translator\n")
            file.write(kernel_src_source)

            if args.verbose:
                print(f"Generated loop device kernel src {i} of {len(app.uniqueLoops())}: {path}")
                
    
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
