

def _get_prefix(cls):
    from test.base import Policy
    from test.base import Baseline
    from test.base import Algorithm

    if hasattr(cls.__init__, '_autoargs_prefix'):
        return cls.__init__._autoargs_prefix
    elif issubclass(cls, Algorithm):
        return 'algo_'
    elif issubclass(cls, Baseline):
        return 'baseline_'
    elif issubclass(cls, Policy):
        return 'policy_'
    else:
        return ""


def _get_info(cls_or_fn):
    if isinstance(cls_or_fn, type):
        if hasattr(cls_or_fn.__init__, '_autoargs_info'):
            return cls_or_fn.__init__._autoargs_info
        return {}
    else:
        if hasattr(cls_or_fn, '_autoargs_info'):
            return cls_or_fn._autoargs_info
        return {}


def _t_or_f(s):
    ua = str(s).upper()
    if ua == 'TRUE'[:len(ua)]:
        return True
    elif ua == 'FALSE'[:len(ua)]:
        return False
    else:
        raise ValueError('Unrecognized boolean value: %s' % s)


def add_args(_):
    def _add_args(cls, parser):
        args_info = _get_info(cls)
        prefix_ = _get_prefix(cls)
        for args_name, arg_info in args_info.items():
            type = args_info['type']
            if type == bool:
                type = _t_or_f

            parser.add_argument(
                '--' + prefix_ + args_name,
                help=arg_info['help'],
                choices=args_info['choices'],
                type=type,
                nargs=arg_info['nargs'])
    return _add_args