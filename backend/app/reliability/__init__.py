"""Reliability Lab: run one input through an existing pipeline repeatedly and compare the results.

normalize.py  turns one repetition's ordinary outputs into a comparable record.
metrics.py    computes consistency, accuracy, step, detection, and timing figures from those records.
learning.py   compares procedures learned independently from one demonstration.
service.py    creates, reads, cancels, and labels batches.
executor.py   launches repetitions into free slots and runs their jobs on a bounded lane.
"""
