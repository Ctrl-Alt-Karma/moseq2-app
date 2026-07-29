"""
Tests for the flip-classifier processing record.

Applying the flip classifier rewrites frames in place, and a 180 degree
rotation is its own inverse, so a second application silently undoes the
first. The processing record is what makes that impossible -- including in the
case that actually happens in a lab, where a long run is interrupted partway
through and someone re-runs it.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from os.path import join
from unittest import TestCase

import h5py
import numpy as np

from moseq2_app.flip.widget import (FLIP_RECORD_PATH, FLIP_RECORD_JOURNAL_PATH,
                                    FLIP_RECORD_UPDATE_ATTR, LEGACY_FLIP_ATTR,
                                    EXTRACTION_FLIPS_PATH,
                                    EXTRACTION_PROVENANCE_PATHS,
                                    FlipProcessingError, _sha256,
                                    read_extraction_provenance,
                                    read_flip_record, write_flip_record)


class TestFlipRecord(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = join(self.tmp, 'session.h5')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _h5(self, mode='w'):
        return h5py.File(self.path, mode)

    def test_fresh_file_reads_as_not_started(self):
        with self._h5() as f:
            f.create_dataset('frames', data=np.zeros((4, 8, 8), dtype='uint8'))
            self.assertEqual(read_flip_record(f), {'state': 'not_started'})

    def test_legacy_attribute_reads_as_complete(self):
        with self._h5() as f:
            f.attrs[LEGACY_FLIP_ATTR] = True
            record = read_flip_record(f)
        self.assertEqual(record['state'], 'complete')
        self.assertTrue(record['legacy'])

    def test_legacy_attribute_false_reads_as_not_started(self):
        with self._h5() as f:
            f.attrs[LEGACY_FLIP_ATTR] = False
            self.assertEqual(read_flip_record(f)['state'], 'not_started')

    def test_round_trip_preserves_every_field(self):
        record = {
            'state': 'complete',
            'classifier_path': '/some/clf.pkl',
            'classifier_sha256': 'a' * 64,
            'chunk_size': 4000,
            'chunk_overlap': 100,
            'smoothing': 51,
            'frame_path': 'frames',
            'n_frames': 900,
            'frames_corrected': 900,
            'frames_flipped': 137,
            'started': '2026-07-29T00:00:00',
            'finished': '2026-07-29T00:05:00',
        }
        with self._h5() as f:
            write_flip_record(f, record)
        with self._h5('r') as f:
            self.assertEqual(read_flip_record(f), record)

    def test_writing_a_record_twice_replaces_it(self):
        with self._h5() as f:
            write_flip_record(f, {'state': 'in_progress', 'n_frames': 10})
            write_flip_record(f, {'state': 'complete', 'n_frames': 10})
        with self._h5('r') as f:
            self.assertEqual(read_flip_record(f)['state'], 'complete')

    def test_legacy_attribute_tracks_completion(self):
        """Older apps must refuse every state where frames may be touched."""
        with self._h5() as f:
            write_flip_record(f, {'state': 'in_progress'})
            self.assertTrue(f.attrs[LEGACY_FLIP_ATTR])
            write_flip_record(f, {'state': 'failed'})
            self.assertTrue(f.attrs[LEGACY_FLIP_ATTR])
            write_flip_record(f, {'state': 'complete'})
            self.assertTrue(f.attrs[LEGACY_FLIP_ATTR])

    def test_fault_injection_never_fails_open(self):
        """Lightweight exception-hook coverage; this is not process death."""
        stages = (
            'after_sentinel_flush',
            'after_inactive_slot_clear',
            'after_new_slot_create',
            'after_new_slot_flush',
        )
        for stage in stages:
            path = join(self.tmp, stage + '.h5')
            with h5py.File(path, 'w') as f:
                write_flip_record(f, {'state': 'in_progress', 'version': 1})

            def inject(current, fail_at=stage):
                if current == fail_at:
                    raise RuntimeError('fault at ' + current)

            with self.assertRaisesRegex(RuntimeError, stage):
                with h5py.File(path, 'a') as f:
                    write_flip_record(
                        f,
                        {'state': 'failed', 'version': 2},
                        fault_inject=inject,
                    )

            with h5py.File(path, 'r') as f:
                # The legacy reader's only signal must stay latched, and the new
                # reader must never interpret an interrupted update as pristine.
                self.assertTrue(f.attrs[LEGACY_FLIP_ATTR], stage)
                self.assertNotEqual(read_flip_record(f)['state'], 'not_started')
                self.assertTrue(
                    f.attrs.get(FLIP_RECORD_UPDATE_ATTR, False),
                    stage,
                )
                self.assertTrue(
                    FLIP_RECORD_PATH in f or FLIP_RECORD_JOURNAL_PATH in f,
                    stage,
                )

    def test_abrupt_process_death_at_every_checkpoint_is_fail_closed(self):
        stages = (
            'after_sentinel_flush',
            'after_inactive_slot_clear',
            'after_new_slot_create',
            'after_new_slot_flush',
            'after_sentinel_clear',
        )
        child = r'''
import os
import signal
import sys
import h5py
from moseq2_app.flip.widget import write_flip_record
path, stage = sys.argv[1:3]
def inject(current):
    if current == stage:
        if os.name == "posix":
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(99)
with h5py.File(path, "a") as handle:
    write_flip_record(
        handle,
        {"state": "complete", "frames_corrected": 10},
        fault_inject=inject,
    )
'''
        for stage in stages:
            path = join(self.tmp, 'abrupt-' + stage + '.h5')
            with h5py.File(path, 'w') as f:
                write_flip_record(f, {
                    'state': 'in_progress',
                    'frames_corrected': 0,
                })

            result = subprocess.run(
                [sys.executable, '-c', child, path, stage],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0, stage)

            # Parent reopen proves HDF5 readability after real unclean child
            # termination. Slot persistence can vary by HDF5/platform; the
            # reader semantics may not.
            with h5py.File(path, 'r') as f:
                self.assertTrue(f.attrs[LEGACY_FLIP_ATTR], stage)
                record = read_flip_record(f)
                if stage == 'after_sentinel_clear':
                    self.assertFalse(
                        f.attrs.get(FLIP_RECORD_UPDATE_ATTR, False), stage)
                    self.assertEqual(record['state'], 'complete', stage)
                else:
                    self.assertTrue(
                        f.attrs.get(FLIP_RECORD_UPDATE_ATTR, False), stage)
                    self.assertEqual(record['state'], 'failed', stage)

    def test_initial_record_fault_leaves_both_readers_latched(self):
        def inject(stage):
            if stage == 'after_sentinel_flush':
                raise RuntimeError(stage)

        with self.assertRaisesRegex(RuntimeError, 'after_sentinel_flush'):
            with self._h5() as f:
                write_flip_record(
                    f,
                    {'state': 'in_progress'},
                    fault_inject=inject,
                )

        with self._h5('r') as f:
            self.assertTrue(f.attrs[LEGACY_FLIP_ATTR])
            self.assertEqual(read_flip_record(f)['state'], 'failed')
            self.assertIn('interrupted', read_flip_record(f)['error'])

    def test_fault_after_sentinel_clear_leaves_new_record_durable(self):
        with self._h5() as f:
            write_flip_record(f, {'state': 'in_progress', 'version': 1})

            def inject(stage):
                if stage == 'after_sentinel_clear':
                    raise RuntimeError(stage)

            with self.assertRaisesRegex(RuntimeError, 'after_sentinel_clear'):
                write_flip_record(
                    f,
                    {'state': 'complete', 'version': 2},
                    fault_inject=inject,
                )

        with self._h5('r') as f:
            self.assertFalse(f.attrs.get(FLIP_RECORD_UPDATE_ATTR, False))
            self.assertTrue(f.attrs[LEGACY_FLIP_ATTR])
            self.assertEqual(read_flip_record(f), {
                'state': 'complete',
                'version': 2,
            })

    def test_corrupt_record_reads_as_failed_not_untouched(self):
        """A truncated record means a crash mid-write, not a pristine file."""
        with self._h5() as f:
            f.create_dataset(FLIP_RECORD_PATH, data='{"state": "in_prog')
        with self._h5('r') as f:
            record = read_flip_record(f)
        self.assertEqual(record['state'], 'failed')
        self.assertIn('corrupt', record['error'])

    def test_non_mapping_json_reads_as_corrupt_not_attribute_error(self):
        for payload in ('null', '7', '"text"', '["item"]'):
            with self.subTest(payload=payload):
                path = join(self.tmp, hashlib.sha256(
                    payload.encode()).hexdigest() + '.h5')
                with h5py.File(path, 'w') as f:
                    f.create_dataset(FLIP_RECORD_PATH, data=payload)
                with h5py.File(path, 'r') as f:
                    record = read_flip_record(f)
                self.assertEqual(record['state'], 'failed')
                self.assertIn('corrupt', record['error'])

    def test_mapping_json_remains_valid(self):
        with self._h5() as f:
            f.create_dataset(
                FLIP_RECORD_PATH,
                data=json.dumps({'state': 'complete', 'frames_corrected': 4}))
        with self._h5('r') as f:
            record = read_flip_record(f)
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(record['frames_corrected'], 4)

    def test_sentinel_diagnostics_never_auto_trust_slots(self):
        cases = (
            ('none', []),
            ('failed', [('failed', 1)]),
            ('complete', [('complete', 2)]),
            ('conflicting', [('failed', 3), ('complete', 3)]),
        )
        for name, slots in cases:
            path = join(self.tmp, 'sentinel-' + name + '.h5')
            with h5py.File(path, 'w') as f:
                f.attrs[FLIP_RECORD_UPDATE_ATTR] = True
                for (state, sequence), slot_path in zip(
                        slots, (FLIP_RECORD_PATH, FLIP_RECORD_JOURNAL_PATH)):
                    f.create_dataset(slot_path, data=json.dumps({
                        'state': state,
                        '_journal_sequence': sequence,
                    }))
            with h5py.File(path, 'r') as f:
                record = read_flip_record(f)
            self.assertEqual(record['state'], 'failed', name)
            self.assertIn('unknown', record['error'], name)
            self.assertEqual(len(record['diagnostic_slots']), len(slots), name)
            diagnostic_states = {
                item.get('state') for item in record['diagnostic_slots']
            }
            self.assertEqual(
                diagnostic_states, {state for state, _ in slots}, name)

    def test_conflicting_same_sequence_slots_fail_closed(self):
        with self._h5() as f:
            f.create_dataset(FLIP_RECORD_PATH, data=json.dumps({
                'state': 'failed', '_journal_sequence': 7,
            }))
            f.create_dataset(FLIP_RECORD_JOURNAL_PATH, data=json.dumps({
                'state': 'complete', '_journal_sequence': 7,
            }))
        with self._h5('r') as f:
            record = read_flip_record(f)
        self.assertEqual(record['state'], 'failed')
        self.assertIn('conflict', record['error'])
        self.assertEqual(len(record['diagnostic_slots']), 2)

    def test_record_without_state_reads_as_failed(self):
        with self._h5() as f:
            f.create_dataset(FLIP_RECORD_PATH, data=json.dumps({'n_frames': 10}))
        with self._h5('r') as f:
            self.assertEqual(read_flip_record(f)['state'], 'failed')

    def test_sha256_matches_hashlib(self):
        import hashlib
        payload = b'not really a classifier'
        clf = join(self.tmp, 'clf.pkl')
        with open(clf, 'wb') as f:
            f.write(payload)
        self.assertEqual(_sha256(clf), hashlib.sha256(payload).hexdigest())

    def test_sha256_of_missing_file_is_none(self):
        self.assertIsNone(_sha256(join(self.tmp, 'nope.pkl')))

    def test_extraction_provenance_uses_canonical_path_order(self):
        self.assertEqual(
            EXTRACTION_PROVENANCE_PATHS,
            ('metadata/extraction/pipeline', 'metadata/pipeline'),
        )

    def test_identical_dual_extraction_provenance_is_accepted(self):
        record = {'package': 'moseq2-extract', 'policies': {'x': 'same'}}
        with self._h5() as f:
            for path in EXTRACTION_PROVENANCE_PATHS:
                f.create_dataset(path, data=json.dumps(record))
            self.assertEqual(read_extraction_provenance(f), record)

    def test_conflicting_dual_extraction_provenance_is_refused(self):
        with self._h5() as f:
            f.create_dataset(
                EXTRACTION_PROVENANCE_PATHS[0],
                data=json.dumps({
                    'package': 'moseq2-extract',
                    'policies': {'x': 'first'},
                }),
            )
            f.create_dataset(
                EXTRACTION_PROVENANCE_PATHS[1],
                data=json.dumps({
                    'package': 'moseq2-extract',
                    'policies': {'x': 'second'},
                }),
            )
            with self.assertRaisesRegex(
                    FlipProcessingError, 'Conflicting extraction provenance'):
                read_extraction_provenance(f)

    def test_corrupt_extraction_provenance_is_refused(self):
        with self._h5() as f:
            f.create_dataset(
                EXTRACTION_PROVENANCE_PATHS[0], data='{"policies": ')
            with self.assertRaisesRegex(
                    FlipProcessingError, 'Corrupt extraction provenance'):
                read_extraction_provenance(f)


class TestFlipRecordGatesReprocessing(TestCase):
    """
    End-to-end behaviour of `apply_flip_classifier`, with the classifier itself
    stubbed out so the test is about the record, not about sklearn.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clf = join(self.tmp, 'clf.pkl')
        with open(self.clf, 'wb') as f:
            f.write(b'stub classifier')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _session(self, name, n_frames=20, existing_record=None, legacy=False):
        path = join(self.tmp, name)
        with h5py.File(path, 'w') as f:
            # each frame is a distinct constant image, so a 180 degree rotation
            # is detectable by value even though the images are symmetric
            frames = np.arange(n_frames, dtype='uint8')[:, None, None] * np.ones((1, 4, 4), dtype='uint8')
            f.create_dataset('frames', data=frames)
            f.create_dataset('scalars/angle', data=np.zeros(n_frames))
            if existing_record is not None:
                write_flip_record(f, existing_record)
            if legacy:
                f.attrs[LEGACY_FLIP_ATTR] = True
        return path

    def _widget(self, sessions):
        from moseq2_app.flip.widget import FlipClassifierWidget
        widget = FlipClassifierWidget.__new__(FlipClassifierWidget)
        widget.sessions = sessions
        return widget

    def _run(self, widget, flips_returned, **kwargs):
        """Run apply_flip_classifier with get_flips stubbed."""
        import moseq2_app.flip.widget as mod
        original = mod.get_flips
        mod.get_flips = lambda batch, flip_file=None, smoothing=None: flips_returned(batch)
        try:
            widget.apply_flip_classifier(self.clf, verbose=False, **kwargs)
        finally:
            mod.get_flips = original

    def test_successful_run_writes_a_complete_record(self):
        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})

        # flip exactly the first frame of each batch
        def flips(batch):
            out = np.zeros(len(batch), dtype=bool)
            out[0] = True
            return out

        self._run(widget, flips, chunk_size=20)

        with h5py.File(path, 'r') as f:
            record = read_flip_record(f)
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(record['n_frames'], 20)
        self.assertEqual(record['frames_corrected'], 20)
        self.assertEqual(record['frames_flipped'], 1)
        self.assertEqual(record['classifier_sha256'], _sha256(self.clf))
        self.assertEqual(record['chunk_size'], 20)
        self.assertIn('started', record)
        self.assertIn('finished', record)

    def test_refuses_session_with_extraction_time_flip_correction(self):
        path = self._session('already-corrected.h5', n_frames=20)
        with h5py.File(path, 'a') as f:
            f.create_dataset(
                EXTRACTION_FLIPS_PATH,
                data=np.asarray([False, True] + [False] * 18, dtype=bool),
            )
            frames_before = f['frames'][:].copy()
            angles_before = f['scalars/angle'][:].copy()

        widget = self._widget({'a': path})
        with self.assertRaisesRegex(
                FlipProcessingError, 'extraction-time flip correction'):
            self._run(
                widget,
                lambda batch: np.ones(len(batch), dtype=bool),
                chunk_size=20,
            )

        with h5py.File(path, 'r') as f:
            np.testing.assert_array_equal(frames_before, f['frames'][:])
            np.testing.assert_array_equal(angles_before, f['scalars/angle'][:])
            self.assertNotIn(FLIP_RECORD_PATH, f)
            self.assertNotIn(FLIP_RECORD_JOURNAL_PATH, f)
            self.assertFalse(f.attrs.get(LEGACY_FLIP_ATTR, False))

    def test_completed_session_is_skipped_not_reflipped(self):
        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})

        def flips(batch):
            out = np.zeros(len(batch), dtype=bool)
            out[0] = True
            return out

        self._run(widget, flips, chunk_size=20)
        with h5py.File(path, 'r') as f:
            after_first = f['frames'][:].copy()
            angles_first = f['scalars/angle'][:].copy()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self._run(widget, flips, chunk_size=20)

        with h5py.File(path, 'r') as f:
            after_second = f['frames'][:]
            angles_second = f['scalars/angle'][:]

        np.testing.assert_array_equal(after_first, after_second)
        np.testing.assert_array_equal(angles_first, angles_second)
        self.assertTrue(any('already applied' in str(w.message) for w in caught))

    def test_in_progress_record_is_a_hard_refusal(self):
        path = self._session('a.h5', existing_record={
            'state': 'in_progress', 'n_frames': 20, 'frames_corrected': 7,
            'started': '2026-07-29T00:00:00'})
        widget = self._widget({'a': path})

        with h5py.File(path, 'r') as f:
            before = f['frames'][:].copy()

        with self.assertRaises(FlipProcessingError) as ctx:
            self._run(widget, lambda b: np.ones(len(b), dtype=bool), chunk_size=20)

        message = str(ctx.exception)
        self.assertIn('in_progress', message)
        self.assertIn(path, message)
        # the user needs to know how far it got and what to do
        self.assertIn('7 of 20', message)
        self.assertIn('re-extract', message.lower())

        # and nothing may have been touched
        with h5py.File(path, 'r') as f:
            np.testing.assert_array_equal(before, f['frames'][:])
            self.assertEqual(read_flip_record(f)['state'], 'in_progress')

    def test_failed_record_is_a_hard_refusal(self):
        path = self._session('a.h5', existing_record={
            'state': 'failed', 'error': 'disk full', 'n_frames': 20,
            'frames_corrected': 12})
        widget = self._widget({'a': path})

        with h5py.File(path, 'r') as f:
            before = f['frames'][:].copy()

        with self.assertRaises(FlipProcessingError) as ctx:
            self._run(widget, lambda b: np.ones(len(b), dtype=bool), chunk_size=20)

        self.assertIn('disk full', str(ctx.exception))
        self.assertIn('re-extract', str(ctx.exception).lower())

        with h5py.File(path, 'r') as f:
            np.testing.assert_array_equal(before, f['frames'][:])

    def test_legacy_processed_session_is_skipped(self):
        path = self._session('a.h5', legacy=True)
        widget = self._widget({'a': path})

        with h5py.File(path, 'r') as f:
            before = f['frames'][:].copy()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            self._run(widget, lambda b: np.ones(len(b), dtype=bool), chunk_size=20)

        with h5py.File(path, 'r') as f:
            np.testing.assert_array_equal(before, f['frames'][:])

    def test_crash_mid_run_leaves_a_failed_record(self):
        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})

        calls = {'n': 0}

        def flips(batch):
            calls['n'] += 1
            if calls['n'] == 2:
                raise RuntimeError('classifier exploded')
            out = np.zeros(len(batch), dtype=bool)
            out[0] = True
            return out

        with self.assertRaises(RuntimeError):
            self._run(widget, flips, chunk_size=10)

        with h5py.File(path, 'r') as f:
            record = read_flip_record(f)
        self.assertEqual(record['state'], 'failed')
        self.assertIn('classifier exploded', record['error'])
        # the first batch was committed before the crash, so the record must
        # show a partial correction rather than claiming nothing happened
        self.assertEqual(record['frames_corrected'], 10)
        self.assertIn('failed_at', record)

    def test_keyboard_interrupt_also_leaves_a_failed_record(self):
        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})

        def flips(batch):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self._run(widget, flips, chunk_size=10)

        with h5py.File(path, 'r') as f:
            self.assertEqual(read_flip_record(f)['state'], 'failed')

    def test_session_crashed_then_rerun_is_refused_not_double_flipped(self):
        """The whole point: a crash followed by a re-run must not undo work."""
        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})

        calls = {'n': 0}

        def flips(batch):
            calls['n'] += 1
            if calls['n'] == 2:
                raise RuntimeError('boom')
            return np.ones(len(batch), dtype=bool)

        with self.assertRaises(RuntimeError):
            self._run(widget, flips, chunk_size=10)

        with h5py.File(path, 'r') as f:
            after_crash = f['frames'][:].copy()
            angles_after_crash = f['scalars/angle'][:].copy()

        # the re-run must refuse rather than flip the first 10 frames back
        with self.assertRaises(FlipProcessingError):
            self._run(widget, lambda b: np.ones(len(b), dtype=bool), chunk_size=10)

        with h5py.File(path, 'r') as f:
            np.testing.assert_array_equal(after_crash, f['frames'][:])
            np.testing.assert_array_equal(angles_after_crash, f['scalars/angle'][:])

    def test_completed_session_skips_and_later_sessions_still_run(self):
        """A finished session is a normal skip, not a reason to abort the batch."""
        done = self._session('done.h5', n_frames=20)
        with h5py.File(done, 'a') as f:
            write_flip_record(f, {'state': 'complete', 'n_frames': 20})
        todo = self._session('todo.h5', n_frames=20)
        widget = self._widget({'done': done, 'todo': todo})

        with h5py.File(done, 'r') as f:
            done_before = f['frames'][:].copy()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self._run(widget, lambda b: np.ones(len(b), dtype=bool), chunk_size=20)

        with h5py.File(done, 'r') as f:
            np.testing.assert_array_equal(done_before, f['frames'][:])
        with h5py.File(todo, 'r') as f:
            self.assertEqual(read_flip_record(f)['state'], 'complete')
        self.assertTrue(any('already applied' in str(w.message) for w in caught))

    def test_complete_record_carries_the_full_field_set(self):
        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})
        self._run(widget, lambda b: np.zeros(len(b), dtype=bool), chunk_size=10,
                  chunk_overlap=2, smoothing=31)

        with h5py.File(path, 'r') as f:
            record = read_flip_record(f)

        for field in ('package', 'version', 'git_sha', 'state',
                      'classifier_path', 'classifier_filename', 'classifier_sha256',
                      'chunk_size', 'chunk_overlap', 'smoothing', 'frame_path',
                      'n_frames', 'frames_corrected', 'frames_flipped',
                      'started', 'finished', 'preview_requested', 'preview_written',
                      'source_extraction_policies', 'source_extraction_policy_sha256'):
            self.assertIn(field, record)

        self.assertEqual(record['package'], 'moseq2-app')
        self.assertEqual(record['classifier_filename'], 'clf.pkl')
        self.assertEqual(record['chunk_overlap'], 2)
        self.assertEqual(record['smoothing'], 31)
        self.assertFalse(record['preview_requested'])
        self.assertIsNone(record['preview_written'])

    def test_classifier_hash_changes_when_classifier_bytes_change(self):
        path_a = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path_a})
        self._run(widget, lambda b: np.zeros(len(b), dtype=bool), chunk_size=20)
        with h5py.File(path_a, 'r') as f:
            sha_a = read_flip_record(f)['classifier_sha256']

        # retrain: same path, different bytes
        with open(self.clf, 'wb') as f:
            f.write(b'stub classifier v2')

        path_b = self._session('b.h5', n_frames=20)
        widget = self._widget({'b': path_b})
        self._run(widget, lambda b: np.zeros(len(b), dtype=bool), chunk_size=20)
        with h5py.File(path_b, 'r') as f:
            sha_b = read_flip_record(f)['classifier_sha256']

        self.assertNotEqual(sha_a, sha_b)
        self.assertEqual(sha_b, _sha256(self.clf))

    def test_extraction_provenance_is_snapshotted_but_not_modified(self):
        policies = {'scalar_px_to_mm': 'pinhole', 'area_units': 'mm2-xy-product'}
        extraction = {'package': 'moseq2-extract', 'version': '1.2.0',
                      'git_sha': 'abc123', 'policies': policies}
        path = self._session('a.h5', n_frames=20)
        with h5py.File(path, 'a') as f:
            f.create_dataset('metadata/extraction/pipeline', data=json.dumps(extraction))

        widget = self._widget({'a': path})
        self._run(widget, lambda b: np.zeros(len(b), dtype=bool), chunk_size=20)

        with h5py.File(path, 'r') as f:
            # the extraction record must come back byte-identical
            raw = f['metadata/extraction/pipeline'][()]
            if isinstance(raw, bytes):
                raw = raw.decode()
            self.assertEqual(json.loads(raw), extraction)

            record = read_flip_record(f)

        self.assertEqual(record['source_extraction_policies'], policies)
        expected = hashlib.sha256(
            json.dumps(policies, sort_keys=True).encode()).hexdigest()
        self.assertEqual(record['source_extraction_policy_sha256'], expected)

    def test_unstamped_extraction_snapshots_as_none(self):
        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})
        self._run(widget, lambda b: np.zeros(len(b), dtype=bool), chunk_size=20)
        with h5py.File(path, 'r') as f:
            record = read_flip_record(f)
        self.assertIsNone(record['source_extraction_policies'])
        self.assertIsNone(record['source_extraction_policy_sha256'])

    def test_preview_failure_still_completes_frames_but_is_recorded(self):
        """Frame correction is the deliverable; the preview movie is not."""
        import moseq2_app.flip.widget as mod

        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})

        def boom(*args, **kwargs):
            raise AttributeError('ffmpeg not found')

        original = mod.write_frames_preview
        mod.write_frames_preview = boom
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                self._run(widget, lambda b: np.zeros(len(b), dtype=bool),
                          chunk_size=10, write_movie=True)
        finally:
            mod.write_frames_preview = original

        with h5py.File(path, 'r') as f:
            record = read_flip_record(f)

        # frames were fully corrected despite the preview failing
        self.assertEqual(record['state'], 'complete')
        self.assertEqual(record['frames_corrected'], 20)
        # ...and the preview failure is on the record, not just in a warning
        self.assertTrue(record['preview_requested'])
        self.assertFalse(record['preview_written'])
        self.assertIn('ffmpeg not found', record['preview_error'])
        self.assertTrue(any('preview movie' in str(w.message) for w in caught))

    def test_preview_is_finalized_before_complete_record(self):
        import moseq2_app.flip.widget as mod

        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})
        written_states = []

        class Pipe:
            returncode = 0

            def communicate(self):
                self.was_finalized = True
                self.assert_no_complete_yet()
                return b'', b''

            def assert_no_complete_yet(self):
                assert 'complete' not in written_states

        pipe = Pipe()
        pipe.was_finalized = False
        original_preview = mod.write_frames_preview
        original_record = mod.write_flip_record

        def preview(*args, **kwargs):
            return pipe

        def record(f, value, *args, **kwargs):
            written_states.append(value['state'])
            return original_record(f, value, *args, **kwargs)

        mod.write_frames_preview = preview
        mod.write_flip_record = record
        try:
            self._run(
                widget,
                lambda batch: np.zeros(len(batch), dtype=bool),
                chunk_size=10,
                write_movie=True,
            )
        finally:
            mod.write_frames_preview = original_preview
            mod.write_flip_record = original_record

        self.assertTrue(pipe.was_finalized)
        self.assertEqual(written_states, ['in_progress', 'complete'])
        with h5py.File(path, 'r') as f:
            record = read_flip_record(f)
        self.assertTrue(record['preview_written'])
        self.assertIsNone(record['preview_error'])

    def test_preview_nonzero_exit_is_recorded_before_completion(self):
        import moseq2_app.flip.widget as mod

        path = self._session('a.h5', n_frames=20)
        widget = self._widget({'a': path})

        class Pipe:
            returncode = 7

            def communicate(self):
                return b'', b'encoder exploded'

        pipe = Pipe()
        original = mod.write_frames_preview
        mod.write_frames_preview = lambda *args, **kwargs: pipe
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                self._run(
                    widget,
                    lambda batch: np.zeros(len(batch), dtype=bool),
                    chunk_size=10,
                    write_movie=True,
                )
        finally:
            mod.write_frames_preview = original

        with h5py.File(path, 'r') as f:
            record = read_flip_record(f)
        self.assertEqual(record['state'], 'complete')
        self.assertFalse(record['preview_written'])
        self.assertIn('status 7', record['preview_error'])
        self.assertIn('encoder exploded', record['preview_error'])
        self.assertTrue(any('finalize flipped movie' in str(w.message)
                            for w in caught))

    def test_preview_is_finalized_on_mid_batch_exception(self):
        path = self._session('preview-exception.h5', n_frames=20)
        widget = self._widget({'a': path})

        class Pipe:
            returncode = 0

            def __init__(self):
                self.communicated = False

            def communicate(self):
                self.communicated = True
                return b'', b''

        pipe = Pipe()
        calls = {'n': 0}

        def flips(batch):
            calls['n'] += 1
            if calls['n'] == 2:
                raise RuntimeError('classifier exploded')
            return np.zeros(len(batch), dtype=bool)

        import moseq2_app.flip.widget as mod
        original_preview = mod.write_frames_preview
        mod.write_frames_preview = lambda *args, **kwargs: pipe
        try:
            with self.assertRaisesRegex(RuntimeError, 'classifier exploded'):
                self._run(
                    widget,
                    flips,
                    chunk_size=10,
                    write_movie=True,
                )
        finally:
            mod.write_frames_preview = original_preview

        self.assertTrue(pipe.communicated)
        with h5py.File(path, 'r') as f:
            record = read_flip_record(f)
        self.assertEqual(record['state'], 'failed')
        self.assertIn('classifier exploded', record['error'])
