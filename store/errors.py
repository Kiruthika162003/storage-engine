from __future__ import annotations

# One exception per thing that can go wrong, arranged so a caller can catch a category.
#
# The split that matters is between a refusal and a corruption. A refusal is an answer: the key
# is not there, the transaction conflicted, the batch is too large. A corruption is a statement
# that something on disk is not what it was written as, and no caller can handle it by retrying.
# Catching StorageError catches both, which is almost never what anybody wants.


class StorageError(Exception):
    """Anything this package raises."""


class Refused(StorageError):
    """The engine understood the request and will not do it."""


class NotFound(Refused):
    """No live value for that key."""


class Conflict(Refused):
    """A transaction wrote a key somebody else wrote after it started."""


class ReadOnly(Refused):
    """A write arrived at something that only reads."""


class TooLarge(Refused):
    """A key, value or batch is past a limit the format cannot express."""


class Closed(Refused):
    """The engine has been shut down."""


class Corrupt(StorageError):
    """Something read back is not what was written."""


class BadChecksum(Corrupt):
    """A block or record failed its checksum."""


class TornWrite(Corrupt):
    """A record was cut short, which is what a crash mid write looks like."""


class BadFormat(Corrupt):
    """A file does not have the shape this version can read."""


class MissingFile(Corrupt):
    """The manifest names a file that is not there."""


class ConfigError(StorageError):
    """A setting that cannot be honoured, raised before anything is written."""
