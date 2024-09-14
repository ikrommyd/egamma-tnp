from __future__ import annotations


def extra_filter(events, extra_filter_arg1, extra_filter_arg2):
    print("I'm an extra filter")
    print(f"extra_filter_arg1: {extra_filter_arg1}")
    print(f"extra_filter_arg2: {extra_filter_arg2}")
    return events
