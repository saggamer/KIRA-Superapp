"""Broad-edit sampling and leakage checks; category tags are lexical hints."""
import hashlib
import re

CATEGORIES = ('addition', 'removal', 'replacement', 'color', 'text', 'appearance', 'spatial')

def edit_category(prompt):
    text = prompt.casefold()
    for category, pattern in (
        ('removal', r'\b(remove|delete|erase|eliminate|without)\b'),
        ('addition', r'\b(add|insert|include|put|place)\b'),
        ('text', r'\b(text|words?|letters?|written|writing|reads?|called|named|says?|saying|label|number)\b'),
        ('replacement', r'\b(replace|instead|turn .+ into|change .+ (into|to) (a|an|the))\b'),
        ('color', r'\b(colou?r|red|blue|green|yellow|purple|pink|orange|black|white|brown|grey|gray)\b'),
        ('spatial', r'\b(move|left|right|above|below|raise|lower|rotate|upside|closer|farther|bigger|smaller|size)\b'),
    ):
        if re.search(pattern, text):
            return category
    return 'appearance'

def historical_split(group):
    number = int(hashlib.sha256(str(group).encode()).hexdigest()[:8], 16) % 10
    return 'test' if number == 0 else 'valid' if number == 1 else 'train'

def select_balanced(rows, count, minimum=8):
    """Round-robin actual examples, rejecting missing category coverage."""
    buckets = {category: [] for category in CATEGORIES}
    seen = set()
    for row in rows:
        if row['record_id'] in seen:
            continue
        seen.add(row['record_id'])
        buckets[row['category']].append(row)
    if any(len(bucket) < minimum for bucket in buckets.values()):
        raise ValueError('Insufficient broad edit coverage: ' + str({k:len(v) for k,v in buckets.items()}))
    selected = []
    while len(selected) < count:
        previous = len(selected)
        for category in CATEGORIES:
            if buckets[category] and len(selected) < count:
                selected.append(buckets[category].pop(0))
        if len(selected) == previous:
            raise ValueError('Insufficient unique edit examples')
    if any(sum(r['category'] == c for r in selected) < minimum for c in CATEGORIES):
        raise ValueError('Count too small for required category coverage')
    return selected

def verify_split_identities(splits):
    """Group and canonical decoded image identity must be disjoint."""
    identities = {}
    for name, rows in splits.items():
        groups, pixels = set(), set()
        for row in rows:
            groups.add(str(row['group_id']))
            pixels.update(row['pixel_hashes'])
        identities[name] = (groups, pixels)
    names = list(identities)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if identities[left][0] & identities[right][0] or identities[left][1] & identities[right][1]:
                raise ValueError(f'Split leakage: {left}/{right}')
    return True
