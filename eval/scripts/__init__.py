"""Scripted callers: the eight v1 calls plus thirty-two generated ones, ten per language class."""

from .generated import generate
from .v1 import V1_CALLS

CALLS = {**V1_CALLS, **generate()}
KIND_OF = {cid: cid.split("-")[0] for cid in CALLS}
