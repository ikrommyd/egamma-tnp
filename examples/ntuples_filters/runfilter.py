from __future__ import annotations


def runfilter(events):
    dataset = events.metadata["dataset"]
    if "before" in dataset or "after" in dataset:
        runnumber = dataset.split("_")[-1]

    if "before" in dataset:
        if runnumber == "FPix":
            return events[events.run < 382799]
        elif runnumber == "HCAL":
            return events[(events.run < 383129) & (events.run > 382799)]
        elif runnumber == "MD":
            return events[events.run < 384918]

    elif "after" in dataset:
        if runnumber == "FPix":
            return events[(events.run > 382799) & (events.run < 383129)]
        elif runnumber == "HCAL":
            return events[events.run > 383219]
        elif runnumber == "MD":
            return events[events.run > 384918]

    else:
        return events
