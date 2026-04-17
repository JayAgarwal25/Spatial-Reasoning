VALID_RELATIONS = {
    'left_of', 'right_of', 'above', 'below', 'on', 'under', 'inside', 'surrounding',
    'overlapping', 'in_front_of', 'behind', 'near', 'contains', 'none'
}

SYNONYMS = {
    'left': 'left_of', 'right': 'right_of', 'top': 'above', 'bottom': 'below', 'beneath': 'under',
    'within': 'inside', 'around': 'surrounding', 'overlap': 'overlapping', 'overlapping': 'overlapping',
    'front': 'in_front_of', 'back': 'behind', 'close_to': 'near', 'next_to': 'near', 'beside': 'near',
    'contains': 'contains', 'contain': 'contains', 'none': 'none'
}


def parse_relation_text(text: str) -> str:
    if not text:
        return 'none'
    t = text.strip().lower().replace(' ', '_').replace('-', '_')
    if t in VALID_RELATIONS:
        return t
    for k, v in SYNONYMS.items():
        if k in t:
            return v
    return 'none'
