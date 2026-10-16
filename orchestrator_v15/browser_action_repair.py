"""Repair narrow browser proposal syntax using only the latest observed controls."""
import re

from orchestrator_v15.contracts import parse_block


def repair_observed_action(answer, controller):
    parsed = parse_block(answer)
    if not parsed or parsed[0] != 'BROWSER_TOOL' or not controller.observations:
        return answer
    fields = dict(parsed[1])
    latest = controller.observations[-1]
    elements = [item for item in latest.get('elements', []) if isinstance(item, dict)]
    action = fields.get('ACTION', '').lower()
    if (action == 'act' and fields.get('OPERATION', '').lower() == 'click' and
            fields.get('TARGET_ROLE', '').lower() == 'link' and fields.get('TARGET_NAME') and
            isinstance(latest.get('elements'), list) and not any(
                item.get('role') == 'link' and
                str(item.get('name', '')).casefold() == fields['TARGET_NAME'].casefold()
                for item in elements)):
        # A model may name a link from a prior observed index after opening
        # another source page. Revisit only its exact recorded href.
        urls = {str(item.get('href')) for observation in controller.observations[-4:]
                for item in observation.get('elements', []) if isinstance(item, dict)
                and item.get('role') == 'link' and item.get('href') and
                str(item.get('name', '')).casefold() == fields['TARGET_NAME'].casefold()}
        if len(urls) == 1:
            return '[BROWSER_TOOL]\nACTION: navigate\nURL: ' + urls.pop() + '\n[/BROWSER_TOOL]'
    if action in {'click', 'fill', 'select_option'} or (
            action == 'act' and not fields.get('OPERATION') and
            fields.get('ROLE', '').lower() in {'button', 'link'} and
            fields.get('NAME') and not fields.get('VALUE')):
        # Only translate a compact model dialect when the target is unique in
        # the most recent observation. Never infer a tab or an effect target.
        role = fields.get('ROLE') or fields.get('TARGET_ROLE', '')
        name = fields.get('NAME') or fields.get('TARGET_NAME', '')
        if not role or not name or (fields.get('ROLE') and fields.get('TARGET_ROLE') and
                                    fields['ROLE'] != fields['TARGET_ROLE']) or (
                fields.get('NAME') and fields.get('TARGET_NAME') and
                fields['NAME'] != fields['TARGET_NAME']):
            return answer
        if fields.get('TABID') and fields['TABID'] != latest.get('tabId'):
            return answer
        matches = [item for item in elements if
                   str(item.get('role', '')).casefold() == role.casefold() and
                   str(item.get('name', '')).casefold() == name.casefold()]
        if len(matches) != 1:
            return answer
        operation = action if action != 'act' else 'click'
        if operation in {'fill', 'select_option'} and not fields.get('VALUE'):
            return answer
        fields = {'ACTION': 'act', 'OPERATION': operation,
                  'TARGET_ROLE': matches[0]['role'],
                  'TARGET_NAME': matches[0]['name'],
                  **({'VALUE': fields['VALUE']} if fields.get('VALUE') else {}),
                  **({'TABID': fields['TABID']} if fields.get('TABID') else {})}
        return '[BROWSER_TOOL]\n' + '\n'.join(
            f'{key}: {value}' for key, value in fields.items()) + '\n[/BROWSER_TOOL]'
    if action != 'act':
        return answer
    operation = fields.get('OPERATION', '').lower()
    if operation not in {'click', 'fill', 'select_option'}:
        return answer
    if operation == 'select_option' and fields.get('TARGET_NAME') and fields.get('VALUE'):
        controls = [item for item in elements if item.get('role') == 'combobox' and
                    str(item.get('name', '')).casefold() == fields['TARGET_NAME'].casefold()]
        if len(controls) == 1:
            options = controls[0].get('options') or []
            values = {str(option.get('value')) for option in options if isinstance(option, dict)}
            if fields['VALUE'] not in values:
                matching = [option for option in options if isinstance(option, dict) and
                            str(option.get('label', '')).casefold() == fields['VALUE'].casefold()]
                if len(matching) == 1:
                    fields['VALUE'] = str(matching[0]['value'])
    target = fields.get('TARGET', '').strip()
    if target and not any(fields.get(key) for key in ('TARGET_NAME', 'SELECTOR', 'INDEX')):
        selector = re.fullmatch(r'([a-zA-Z][\w-]*)\[name=[\'\"]([^\'\"]+)[\'\"]\]', target)
        if selector:
            role, name = selector.groups()
            matches = [item for item in elements if item.get('role', '').casefold() == role.casefold()
                       and item.get('name', '').casefold() == name.casefold()]
        else:
            matches = [item for item in elements if item.get('name', '').casefold() == target.casefold()]
        if len(matches) != 1 and operation != 'select_option':
            return answer
        if len(matches) == 1:
            fields.pop('TARGET', None)
            fields['TARGET_NAME'] = matches[0]['name']
            fields['TARGET_ROLE'] = matches[0]['role']
    if operation == 'select_option' and not fields.get('VALUE'):
        named = fields.get('TARGET_NAME', '')
        controls = [item for item in elements if item.get('role') == 'combobox' and
                    (not named or item.get('name', '').casefold() == named.casefold())]
        if len(controls) != 1:
            return answer
        options = controls[0].get('options') or []
        requested = [option for option in options if option.get('label') and re.search(
            r'\b' + re.escape(option['label']) + r'\b', controller.request, re.I)]
        if len(requested) != 1:
            return answer
        fields.pop('TARGET', None)
        fields.pop('SELECTOR', None)
        fields.pop('INDEX', None)
        fields['TARGET_NAME'] = controls[0]['name']
        fields['TARGET_ROLE'] = 'combobox'
        fields['VALUE'] = requested[0]['value']
    if not fields.get('TARGET_NAME') and not any(fields.get(key) for key in ('INDEX', 'SELECTOR')):
        return answer
    return '[BROWSER_TOOL]\n' + '\n'.join(f'{key}: {value}' for key, value in fields.items()) + '\n[/BROWSER_TOOL]'
