"""Shared artifact exclusions for planning and the native execution evidence gate."""
import re


def no_web_requested(request):
    return bool(re.search(
        r'\b(?:offline|no web|without (?:the )?(?:web|internet))\b|'
        r'\b(?:do not|don\'t|never)\s+(?:browse|search (?:the )?(?:web|internet))\b',
        str(request), re.I))


def no_images_requested(request):
    return bool(re.search(
        r'\b(?:no|without)\s+(?:images?|photos?|visuals?|pictures?|illustrations?|artwork)\b|\btext[ -]only\b|'
        r'\b(?:do not|don\'t|never)\s+(?:(?:browse|search|download)\s+or\s+)?'
        r'(?:add|include|use|generate)\s+(?:any\s+)?(?:images?|photos?|visuals?|pictures?|illustrations?|artwork)\b',
        str(request), re.I))
