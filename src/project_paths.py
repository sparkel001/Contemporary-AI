from pathlib import Path, PurePosixPath


def portable_path(value):
    """Accept archived Windows separators on any host."""
    return Path(str(value).replace(chr(92), '/'))


def posix_path(value):
    return PurePosixPath(str(value).replace(chr(92), '/')).as_posix()
