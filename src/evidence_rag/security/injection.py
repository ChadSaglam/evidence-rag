"""Conservative heuristic prompt-injection detection for untrusted documents."""

import hashlib
import re
from dataclasses import dataclass
from re import Pattern

from evidence_rag.schemas import InjectionMatch, SecurityAssessment


@dataclass(frozen=True)
class InjectionRule:
    rule_id: str
    pattern: Pattern[str]


def rule(rule_id: str, expression: str) -> InjectionRule:
    return InjectionRule(rule_id, re.compile(expression, re.IGNORECASE | re.UNICODE))


DEFAULT_RULES = (
    rule(
        "en-ignore-previous-instructions",
        r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+instructions\b",
    ),
    rule(
        "de-ignore-previous-instructions",
        r"\bignorier\w*\s+(?:alle\s+)?(?:vorherigen|bisherigen)\s+anweisungen\b",
    ),
    rule(
        "fr-ignore-previous-instructions",
        r"\bignorez\s+(?:toutes\s+)?les\s+instructions\s+précédentes\b",
    ),
    rule(
        "tr-ignore-previous-instructions",
        r"\bönceki\s+talimatları\s+yok\s+say\b",
    ),
    rule(
        "system-prompt-exfiltration",
        r"\b(?:reveal|print|show|expose)\s+(?:the\s+)?system\s+prompt\b",
    ),
    rule("role-override", r"\byou\s+are\s+now\s+(?:an?|the)\b"),
    rule(
        "secret-exfiltration",
        r"\b(?:send|upload|exfiltrate)\s+[^\n.!?]{0,80}\b(?:api[\s_-]?key|password|secret\s+token)\b",
    ),
    rule("fake-system-tag", r"<\s*system\s*>"),
)


class InjectionDetector:
    """Match documented attack patterns without executing document instructions."""

    def __init__(self, rules=DEFAULT_RULES) -> None:
        self.rules = tuple(rules)

    def assess(self, text: str) -> SecurityAssessment:
        matches = [
            InjectionMatch(
                rule_id=rule.rule_id,
                start_char=match.start(),
                end_char=match.end(),
            )
            for rule in self.rules
            for match in rule.pattern.finditer(text)
        ]
        matches.sort(key=lambda item: (item.start_char, item.end_char, item.rule_id))
        return SecurityAssessment(
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            flagged=bool(matches),
            matches=matches,
        )
