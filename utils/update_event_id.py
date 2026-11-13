def update_event_id(events, event_id):
    # events has to mne events object
    # event_id has to be dict
    return {key: code for key, code in event_id.items() if code in events[:, 2]}