import sys
import traceback

import logging
log = logging.getLogger()

def stacktraces():
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n#\n# ThreadID: %s" % threadId)
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))

    log.error("\n".join(code))

