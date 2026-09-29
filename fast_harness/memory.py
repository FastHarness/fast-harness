import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Statistics:
    successes: float = 0
    failures: float = 0
    total: int = 0
    clean_streak: int = 0
    ever_failed: bool = False


@dataclass(frozen=True)
class Match:
    stage: str | None
    distance: float
    reason: str


def distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or len(a) != len(b):
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def feature_json(features: tuple[float, ...]) -> str:
    if not features or not all(math.isfinite(x) for x in features):
        raise ValueError('Features must be nonempty and finite')
    return json.dumps(features, separators=(',', ':'), allow_nan=False)


class Memory:
    def __init__(self, path: str | Path, namespace: str, decay: float = 0.98,
                 examples_per_stage: int = 128, error_merge_radius: float = 0.04):
        if (not namespace.strip() or not 0 < decay <= 1 or examples_per_stage < 1
                or not math.isfinite(error_merge_radius) or error_merge_radius < 0):
            raise ValueError('Invalid memory configuration')
        self.namespace, self.decay = namespace, decay
        self.examples_per_stage = examples_per_stage
        self.error_merge_radius = error_merge_radius
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (version INTEGER NOT NULL);
            INSERT INTO metadata SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM metadata);
            CREATE TABLE IF NOT EXISTS stats (
                namespace TEXT, stage TEXT, successes REAL, failures REAL,
                total INTEGER, clean_streak INTEGER, ever_failed INTEGER,
                PRIMARY KEY(namespace, stage));
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY, namespace TEXT, stage TEXT,
                fingerprint TEXT, features TEXT,
                UNIQUE(namespace, stage, fingerprint));
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, namespace TEXT, episode TEXT, step INTEGER,
                kind TEXT, stage TEXT, outcome TEXT, evidence TEXT,
                UNIQUE(namespace, episode, step, kind));
            CREATE TABLE IF NOT EXISTS errors (
                namespace TEXT, case_id TEXT, stage TEXT, features TEXT,
                error_type TEXT, count INTEGER, evidence TEXT,
                recovered INTEGER DEFAULT 0, recovery_evidence TEXT DEFAULT '',
                PRIMARY KEY(namespace, case_id));
            CREATE TABLE IF NOT EXISTS spans (
                namespace TEXT, stage TEXT, span_ewma REAL, samples INTEGER,
                PRIMARY KEY(namespace, stage));
        ''')
        if self.db.execute('SELECT version FROM metadata').fetchone()[0] != 1:
            self.db.close()
            raise ValueError('Unsupported memory schema; use a new database')

    def stages(self) -> tuple[str, ...]:
        return tuple(row[0] for row in self.db.execute(
            'SELECT DISTINCT stage FROM examples WHERE namespace=? ORDER BY stage',
            (self.namespace,)))

    def learn_stage(self, stage: str, features: tuple[float, ...]) -> None:
        encoded = feature_json(features)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        with self.db:
            self.db.execute('''INSERT OR REPLACE INTO examples
                (namespace, stage, fingerprint, features) VALUES (?, ?, ?, ?)''',
                (self.namespace, stage, fingerprint, encoded))
            self.db.execute('''DELETE FROM examples WHERE namespace=? AND stage=?
                AND id NOT IN (SELECT id FROM examples WHERE namespace=? AND stage=?
                ORDER BY id DESC LIMIT ?)''',
                (self.namespace, stage, self.namespace, stage, self.examples_per_stage))

    def match(self, features: tuple[float, ...], threshold: float, margin: float) -> Match:
        feature_json(features)
        nearest: dict[str, float] = {}
        for row in self.db.execute('SELECT stage, features FROM examples WHERE namespace=?',
                                   (self.namespace,)):
            value = distance(features, tuple(json.loads(row['features'])))
            nearest[row['stage']] = min(value, nearest.get(row['stage'], math.inf))
        ranked = sorted(nearest, key=nearest.get)
        if not ranked or nearest[ranked[0]] > threshold:
            return Match(None, nearest[ranked[0]] if ranked else math.inf, 'unknown_context')
        if len(ranked) > 1 and nearest[ranked[1]] - nearest[ranked[0]] < margin:
            return Match(None, nearest[ranked[0]], 'ambiguous_stage')
        return Match(ranked[0], nearest[ranked[0]], 'matched')

    def statistics(self, stage: str) -> Statistics:
        row = self.db.execute('SELECT * FROM stats WHERE namespace=? AND stage=?',
                              (self.namespace, stage)).fetchone()
        return Statistics() if row is None else Statistics(
            row['successes'], row['failures'], row['total'], row['clean_streak'],
            bool(row['ever_failed']))

    def record(self, *, episode: str, step: int, kind: str, stage: str,
               features: tuple[float, ...], outcome: str, evidence: str,
               student: bool, error_type: str = '') -> str | None:
        if outcome not in ('ok', 'error', 'unknown'):
            raise ValueError('Invalid observed outcome')
        encoded = feature_json(features)
        case_id = hashlib.sha256(f'{stage}|{encoded}|{error_type}'.encode()).hexdigest()[:24]
        with self.db:
            cursor = self.db.execute('''INSERT OR IGNORE INTO events
                (namespace, episode, step, kind, stage, outcome, evidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (self.namespace, episode, step, kind, stage, outcome, evidence))
            if not cursor.rowcount:
                return None
            if student and outcome != 'unknown':
                old = self.statistics(stage)
                good = outcome == 'ok'
                self.db.execute('INSERT OR REPLACE INTO stats VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (self.namespace, stage, old.successes * self.decay + int(good),
                     old.failures * self.decay + int(not good), old.total + 1,
                     old.clean_streak + 1 if good else 0, old.ever_failed or not good))
            if outcome == 'error':
                nearest = sorted((distance(features, tuple(json.loads(row['features']))), row['case_id'])
                    for row in self.db.execute(
                        'SELECT features, case_id FROM errors WHERE namespace=? AND stage=? AND error_type=?',
                        (self.namespace, stage, error_type)))
                if nearest and nearest[0][0] <= self.error_merge_radius:
                    case_id = nearest[0][1]
                if not student:
                    old = self.statistics(stage)
                    self.db.execute('INSERT OR REPLACE INTO stats VALUES (?, ?, ?, ?, ?, ?, ?)',
                        (self.namespace, stage, old.successes, old.failures, old.total, 0, True))
                self.db.execute('''INSERT INTO errors
                    (namespace, case_id, stage, features, error_type, count, evidence)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(namespace, case_id) DO UPDATE SET
                    count=count+1, evidence=excluded.evidence''',
                    (self.namespace, case_id, stage, encoded, error_type, evidence))
                return case_id
        return None

    def errors(self, stage: str | None = None, limit: int = 12) -> tuple[dict, ...]:
        query = 'SELECT * FROM errors WHERE namespace=?'
        values: list = [self.namespace]
        if stage is not None:
            query += ' AND stage=?'
            values.append(stage)
        rows = self.db.execute(query + ' ORDER BY count DESC, case_id LIMIT ?', (*values, limit))
        return tuple({key: row[key] for key in (
            'case_id', 'stage', 'error_type', 'count', 'evidence', 'recovered',
            'recovery_evidence')} for row in rows)

    def near_error(self, stage: str, features: tuple[float, ...], threshold: float) -> bool:
        return any(distance(features, tuple(json.loads(row[0]))) <= threshold
                   for row in self.db.execute(
                       'SELECT features FROM errors WHERE namespace=? AND stage=?',
                       (self.namespace, stage)))

    def recovered(self, case_ids: set[str], evidence: str) -> None:
        with self.db:
            for case_id in case_ids:
                self.db.execute('''UPDATE errors SET recovered=recovered+1,
                    recovery_evidence=? WHERE namespace=? AND case_id=?''',
                    (evidence, self.namespace, case_id))

    def record_span(self, stage: str, chunks: int) -> None:
        """chunk table: EWMA of how many chunks a contiguous run of <stage> took. Drives the commit
        horizon so a proven subtask can run ~its typical span without a mid-subtask forced review.
        Capped downstream, so an occasional recovery-inflated span is harmless."""
        if not isinstance(chunks, int) or chunks < 1:
            return
        row = self.db.execute('SELECT span_ewma, samples FROM spans WHERE namespace=? AND stage=?',
                              (self.namespace, stage)).fetchone()
        if row is None:
            ewma, samples = float(chunks), 1
        else:
            ewma, samples = row['span_ewma'] * 0.7 + chunks * 0.3, row['samples'] + 1
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO spans VALUES (?, ?, ?, ?)',
                            (self.namespace, stage, ewma, samples))

    def span(self, stage: str) -> float | None:
        """EWMA chunk-span recorded for <stage>, or None if never recorded."""
        row = self.db.execute('SELECT span_ewma FROM spans WHERE namespace=? AND stage=?',
                              (self.namespace, stage)).fetchone()
        return row['span_ewma'] if row else None

    def close(self) -> None:
        self.db.close()
