
import pcpp
import sys
from store import ParseError, Location

class Preprocessor(pcpp.Preprocessor):
    def __init__(self, lexer=None):
        print("Custom Preprocessor Enabled")
        super(Preprocessor, self).__init__(lexer)
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
        print("Call custom unknown directive")
        print("%s:%d unknown directive: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks)), file = sys.stderr)
        if directive.value == 'error':
            print("%s:%d error: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks)), file = sys.stderr)
            self.return_code += 1
            return True
        elif directive.value == 'warning':
            print("%s:%d warning: %s" % (directive.source,directive.lineno,''.join(tok.value for tok in toks)), file = sys.stderr)
            return True
        return None