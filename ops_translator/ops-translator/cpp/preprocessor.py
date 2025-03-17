
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

    def get_lineno(self):
        return self.dirtoken.lineno
        
    def get_isl_name(self):
        name_tok = self.arg_toks[0]
        return name_tok.value[1:-1]
    
    def get_max_iter_param(self):
        param = ''.join(tok.value for tok in self.arg_toks[1:])
        return param

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
        raise ParseError("[PREPORC] " + msg, loc)

    def on_include_not_found(self, is_malformed, is_system_include, curdir, includepath) -> None:
        if is_system_include:
            raise pcpp.OutputDirective(pcpp.Action.IgnoreAndPassThrough)

        super().on_include_not_found(is_malformed, is_system_include, curdir, includepath)
        
    def clean_args(self, precedingtoks):
        cleaned_precedingtoks = []
        
        for tok in precedingtoks:
            if not tok.type == "CPP_WS":
                cleaned_precedingtoks.append(tok)
        return cleaned_precedingtoks
            
    def on_directive_unknown(self ,directive, toks, ifpassthru, precedingtoks):
                
        if toks[0].value == "ISL":
            cleaned_args =   self.clean_args(toks[1:])
            
            if len(cleaned_args) < 2:
                self.on_error(directive.source, directive.lineno, f"error in ISL pragma. it got #{len(cleaned_args)} prameters. It should have the name and the total iteration as parameters")
                
            self.__iter_parloop_directives.append(isl_directive(directive,cleaned_args))
            print("[PREPROC_DEBUG] %s:%d ISL directive: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks[1:])), file = sys.stderr)
            return True
        else:
            print("[PREPROC_DEBUG] %s:%d unknown directive: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks)), file = sys.stderr)
            
        # This section is part of original hook
        if directive.value == 'error':
            print("%s:%d error: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks)), file = sys.stderr)
            self.on_error(directive.source, directive.lineno, f"error directive wit values: {''.join(tok.value for tok in toks)}")
            self.return_code += 1
            return True
        elif directive.value == 'warning':
            print("%s:%d warning: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks)), file = sys.stderr)
            return True
        return None
    
    def get_isl_directives(self) -> Any:
        return self.__iter_parloop_directives