"""node id generation"""

from __future__ import annotations

from uuid import uuid4


def new_node_id() -> str:
    """return a new unique node id"""
    return uuid4().hex
