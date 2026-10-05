"""Offline regression and invalid-input checks. Run: python3 -m unittest -v test_reproduce_table"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import reproduce_table as table


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads(table.DATA.read_bytes())

    def test_archive_counts_and_all_published_rates(self):
        report = table.summarize(self.records)
        self.assertEqual(report['conference_records'], 3464)
        self.assertEqual(report['scenarios'], 222)
        self.assertEqual(report['excluded_records'], 12)
        self.assertTrue(all(r['matches_published'] for r in report['table']))
        self.assertEqual([r['cells'][0]['n'] for r in report['table']], [222, 222, 222, 200])
        self.assertEqual([c['leaked'] for c in report['table'][0]['cells']], [184, 212, 203, 201])

    def test_order_does_not_change_report(self):
        self.assertEqual(table.summarize(self.records), table.summarize(list(reversed(self.records))))

    def test_smoke_labels_do_not_change_conference_rates(self):
        rows = copy.deepcopy(self.records)
        for row in rows:
            if row['model'] == table.SMOKE:
                row['leaked'] = not row['leaked']
        self.assertEqual(table.summarize(self.records), table.summarize(rows))

    def test_rejects_bad_structure(self):
        for mutation in ('duplicate', 'missing', 'boolean', 'model', 'condition', 'domain'):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(self.records)
                if mutation == 'duplicate': rows.append(rows[0].copy())
                elif mutation == 'missing': rows.pop(0)
                elif mutation == 'boolean': rows[0]['leaked'] = 'false'
                elif mutation == 'model': rows[0]['model'] = 'unknown'
                elif mutation == 'condition': rows[0]['cond'] = 'C_TH'
                elif mutation == 'domain': rows[0]['vert'] = 'unknown'
                with self.assertRaises(ValueError): table.summarize(rows)

    def test_check_rejects_changed_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'changed.json'
            path.write_bytes(table.DATA.read_bytes() + b'\n')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(table.main(['--input', str(path), '--check']), 1)

    def test_cli_works_outside_repo_without_writing_data(self):
        before = hashlib.sha256(table.DATA.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, '-I', str(Path(table.__file__).resolve()), '--check'],
                                 cwd=tmp, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('Published Table III match: yes', run.stdout)
        self.assertEqual(hashlib.sha256(table.DATA.read_bytes()).hexdigest(), before)
        self.assertEqual(before, table.SHA256)


if __name__ == '__main__':
    unittest.main()
