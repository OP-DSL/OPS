
import pcpp
import sys
from store import ParseError, Location
from typing import List, Any
from dataclasses import dataclass

@dataclass
class isl_directive:
    dirtoken: Any
    arg_toks: List[Any]
    
    def __str__(self) -> str:
        return f"isl_dir: {self.dirtoken.source}:{self.dirtoken.lineno}, dirtoken: {self.dirtoken}, args: {self.arg_toks}"


class Preprocessor(pcpp.Preprocessor):
    def __init__(self, lexer=None):
        print("Custom Preprocessor Enabled")
        super(Preprocessor, self).__init__(lexer)
        self.__iter_parloop_directives = []
        self.line_directive = None

    def on_comment(self, tok: str) -> bool:
        return True

    def on_error(self, file: str, line: int, msg: str) -> None:
        loc = Location(file, line, 0)
        raise ParseError(msg, loc)

    def on_include_not_found(self, is_malformed, is_system_include, curdir, includepath) -> None:
        if is_system_include:
            raise pcpp.OutputDirective(pcpp.Action.IgnoreAndPassThrough)

        super().on_include_not_found(is_malformed, is_system_include, curdir, includepath)
        
    def on_directive_unknown(self ,directive, toks, ifpassthru, precedingtoks):
                
        if toks[0].value == "ISL":
            self.__iter_parloop_directives.append(isl_directive(directive, toks[1:]))
            print("[PREPROC_DEBUG] %s:%d ISL directive: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks[1:])), file = sys.stderr)
            return True
        else:
            print("[PREPROC_DEBUG] %s:%d unknown directive: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks)), file = sys.stderr)
            
        # This section is part of original hook
        if directive.value == 'error':
            print("%s:%d error: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks)), file = sys.stderr)
            self.return_code += 1
            return True
        elif directive.value == 'warning':
            print("%s:%d warning: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks)), file = sys.stderr)
            return True
        return None
    
    def get_isl_directives(self) -> Any:
        return self.__iter_parloop_directives