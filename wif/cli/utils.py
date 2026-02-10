import sys
from argparse import Action

from wif import __version__
from wif.utils import show_default_params

class _VersionAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(__version__)
        sys.exit(0)

def add_util_args(parser, sections=None):
    class _ShowDefaultParamsAction(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            show_default_params(sections)
            sys.exit(0)

    parser.register('action', 'show_default_params', _ShowDefaultParamsAction)
    parser.register('action', 'version', _VersionAction)
    parser.add_argument("--default_params", nargs=0, action='show_default_params',
                        help="display defaults param toml file")
    parser.add_argument("--version", nargs=0, action='version', help="display version")
