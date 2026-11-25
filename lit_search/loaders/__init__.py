"""Loaders for bulk corpora held on disk rather than fetched per query.

A loader is not a connector.  Connectors talk to a remote service and are
paced against its quota; a loader reads a file the operator downloaded, and its
honesty problem is different in kind.  A remote source can be asked how many
records match; a snapshot cannot be asked anything, and it is stale from the
moment it is written.

The one rule shared with the connectors: a loader states what it consulted, and
a reader of the result can always find the snapshot's date and version in it.
"""
