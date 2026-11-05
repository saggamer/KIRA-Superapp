"""Model-authored tool selection. Plans select a workflow, never authorize effects."""
import json

ROUTES = frozenset({'browser', 'mcp', 'control', 'answer'})
SYSTEM = '''Choose the workflow needed to complete the user's request. Return only JSON:
{"route":"browser","purpose":"Look up current public information."}
Choose exactly one route: browser, mcp, control or answer. Choose browser for current public information, researching websites, page interaction or saved browser history. Choose mcp for retrieving data from a connected app. Choose control for local files, installed apps, coding workers or explicitly requested connector configuration. Choose answer when no tool is needed. Select by the goal, not a required keyword. An ordinary cafe recommendation needs current web evidence even when the user does not say browser. A greeting or rewriting supplied text does not need web search. A request to use an app does not request connector installation. The plan grants no permissions and executes nothing. Do not include private reasoning, tool calls, invented connection names or additional fields.
Examples:
User: Which cafes are worth visiting this week?
Plan: {"route":"browser","purpose":"Check current cafe information."}
User: Rewrite this sentence: search me the best cafe.
Plan: {"route":"answer","purpose":"Rewrite the supplied sentence."}
User: What does my charger reminder in the connected notes app say?
Plan: {"route":"mcp","purpose":"Read the requested reminder."}
User: Which apps are installed on this computer?
Plan: {"route":"control","purpose":"Inspect the local app inventory."}'''


def parse_plan(text):
    if not isinstance(text, str) or len(text) > 1500:
        raise ValueError('Invalid tool plan')
    value = text.strip()
    if value.startswith('```json\n') and value.endswith('\n```'):
        value = value[8:-4]
    plan = json.loads(value)
    if (not isinstance(plan, dict) or set(plan) != {'route', 'purpose'}
            or not isinstance(plan['route'], str) or plan['route'] not in ROUTES
            or not isinstance(plan['purpose'], str) or not 1 <= len(plan['purpose']) <= 180
            or any(char in plan['purpose'] for char in '\n\r[]<>')):
        raise ValueError('Invalid tool plan schema')
    return plan


def adapter_channel(plan):
    return {'browser': 'browser', 'mcp': 'mcp', 'control': 'control', 'answer': 'control'}[plan['route']]
