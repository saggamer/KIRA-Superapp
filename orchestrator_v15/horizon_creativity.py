"""Fixed development briefs and a transparent visual-review rubric for Horizon 1."""
RUBRIC = {
    'brief_fidelity': 'Number of the four listed constraints clearly visible (zero to four)',
    'concept_integration': '0 unrelated; 1 decorative collage; 2 partly combined; 3 integrated concept; 4 strongly integrated and specific',
    'visual_coherence': '0 broken; 1 major artifacts; 2 visible issues; 3 coherent with minor issues; 4 visually coherent',
    'composition': '0 unreadable; 1 confused; 2 adequate; 3 clear focal hierarchy; 4 particularly clear composition',
}

BRIEFS = [
    {'id':'greenhouse', 'group':'new-greenhouse',
     'prompt':'A surreal greenhouse suspended between two stone islands in a misty forest, spiraling glass petals and copper roots, dawn watercolor.',
     'criteria':['Recognizable enclosed glass greenhouse with plants inside', 'Suspended between two stone islands',
                 'Spiraling glass petals and copper roots integrated into structure', 'Misty forest and dawn watercolor style']},
    {'id':'nautilus', 'group':'new-nautilus',
     'prompt':'A fantastical forest building shaped like an unfolding paper nautilus, warm interior lights, mossy stone terraces at dusk, architectural illustration.',
     'criteria':['Recognizable building with visible rooms or entrance', 'Spiral nautilus-shell form, rather than a radial star',
                 'Warm interior lights and mossy stone terraces', 'Forest dusk architectural illustration']},
    {'id':'observatory', 'group':'creative-observatory-v1',
     'prompt':'A ceramic teapot redesigned as a tiny working astronomical observatory, telescope through the spout, open domed lid, star-map glaze, studio product photograph.',
     'criteria':['Recognizable ceramic teapot', 'Telescope through the spout', 'Open domed lid and star-map glaze',
                 'Coherent observatory product photograph']},
    {'id':'moth-library', 'group':'creative-moth-library-v1',
     'prompt':'A paper-cut illustration of a tiny librarian riding a mechanical moth through a flooded library; bookshelves rise from water, warm lanterns, teal and amber palette.',
     'criteria':['Tiny librarian visibly riding a mechanical moth', 'Flooded library with bookshelves rising from water',
                 'Layered paper-cut illustration', 'Warm lanterns and teal/amber palette']},
    {'id':'desert-boot', 'group':'creative-desert-boot-v1',
     'prompt':'A transparent rain boot that contains a tiny desert ecosystem with dunes and a flowering cactus; condensation beads on the outside, clean studio product photograph.',
     'criteria':['Recognizable transparent rain boot', 'Dunes and flowering cactus inside the boot',
                 'Condensation on outside surface', 'Clean studio product photograph']},
    {'id':'pear-station', 'group':'creative-pear-station-v1',
     'prompt':'An isometric miniature train station built into a sliced pear, platforms around the seed chamber, one small blue train, warm wood and green peel, clay model photograph.',
     'criteria':['Recognizable sliced pear', 'Station platforms integrated around the seed chamber',
                 'Exactly one small blue train', 'Isometric clay miniature with wood and green peel']},
]


def validate_review(criteria, scores):
    if len(criteria) != 4 or any(type(value) is not bool for value in criteria):
        raise ValueError('Review each of the four visible brief constraints explicitly')
    if set(scores) != set(RUBRIC) or any(type(value) is not int or not 0 <= value <= 4 for value in scores.values()):
        raise ValueError('Each rubric dimension requires an integer from zero to four')
    if scores['brief_fidelity'] != sum(criteria):
        raise ValueError('Brief-fidelity score must agree with the visible constraint decisions')
    # These are transparent agent/human decisions, not model benchmark metrics.
    return {'instruction':all(criteria),
            'creativity':scores['concept_integration'] >= 3 and scores['visual_coherence'] >= 3,
            'quality_mean':sum(scores.values()) / len(scores)}
