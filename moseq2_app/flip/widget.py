import h5py
import json
import pickle
import hashlib
import warnings
import datetime
import panel as pn
import holoviews as hv
from tqdm import tqdm
import numpy as np
from pathlib import Path
import ruamel.yaml as yaml
from collections import defaultdict
from moseq2_extract.extract.proc import get_flips
from moseq2_extract.util import gen_batch_sequence
from moseq2_extract.io.video import write_frames_preview


#: h5 dataset holding the flip-classifier processing record.
FLIP_RECORD_PATH = 'metadata/processing/flip_classifier'

#: Second slot for crash-safe record replacement. Readers select the valid
#: record with the highest journal sequence, so one durable slot always remains
#: while the other is replaced.
FLIP_RECORD_JOURNAL_PATH = FLIP_RECORD_PATH + '_journal'

#: h5 attribute used before the structured record existed. Still honoured when
#: reading so files processed by the older code are not reprocessed.
LEGACY_FLIP_ATTR = 'flip_classifier_applied'

#: Durable fail-closed sentinel set before a record update begins and cleared
#: only after the new slot has been flushed.
FLIP_RECORD_UPDATE_ATTR = 'flip_classifier_record_update_in_progress'


# Workaround for HoloViews/Bokeh _buffers parameter issue
# Bokeh 2.4.2 includes '_buffers' in messages but HoloViews 1.14.7 
# doesn't filter it before passing to Param, causing ValueError
def _patch_holoviews_stream_update():
    """Patch HoloViews Stream._set_stream_parameters to handle _buffers error."""
    from holoviews.streams import Stream
    
    # Store original method
    _original_set_stream_parameters = Stream._set_stream_parameters
    
    def _patched_set_stream_parameters(self, **kwargs):
        """Wrapper that catches and handles _buffers parameter error."""
        try:
            # Try with all parameters first (might work in some version combos)
            return _original_set_stream_parameters(self, **kwargs)
        except ValueError as e:
            # Only intervene if the error is specifically about _buffers
            if '_buffers' in str(e) and 'is not a parameter' in str(e):
                # Remove _buffers and retry
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != '_buffers'}
                return _original_set_stream_parameters(self, **filtered_kwargs)
            # Re-raise any other ValueError
            raise
    
    # Replace the method with our patched version
    Stream._set_stream_parameters = _patched_set_stream_parameters

# Apply the patch when this module is imported
_patch_holoviews_stream_update()


class FlipProcessingError(RuntimeError):
    """
    Raised when a session's flip-classifier state makes it unsafe to touch.

    The case this exists for is an interrupted run: part of the session has been
    rotated and part has not, and nothing in the file distinguishes the two, so
    there is no correct way to continue.
    """


def _sha256(path):
    """Return the SHA-256 of a file, or None if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(str(path), 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _app_version_and_sha():
    """Best-effort version and git commit of the installed moseq2-app."""
    try:
        from moseq2_app import __version__ as version
    except Exception:
        version = 'unknown'
    try:
        import subprocess
        sha = subprocess.check_output(
            ['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = 'unknown'
    return version, sha


def read_extraction_provenance(f):
    """
    Read the extraction's own provenance record, without modifying it.

    Args:
    f (h5py.File): open extraction h5.

    Returns:
    provenance (dict): the extraction provenance, or None if unstamped.
    """
    for candidate in ('metadata/extraction/pipeline', 'metadata/pipeline'):
        if candidate in f:
            raw = f[candidate][()]
            if isinstance(raw, bytes):
                raw = raw.decode()
            try:
                return json.loads(raw)
            except ValueError:
                return None
    return None


def _extraction_policy_snapshot(f):
    """
    Snapshot the extraction policies this flip run was applied on top of.

    Returns both the policy map and a hash of it, so a later reader can tell
    whether two flip-corrected sessions sat on the same extraction semantics
    without having to compare the maps by hand.
    """
    provenance = read_extraction_provenance(f)
    if not provenance:
        return None, None
    policies = provenance.get('policies', {})
    canonical = json.dumps(policies, sort_keys=True)
    return policies, hashlib.sha256(canonical.encode()).hexdigest()


def read_flip_record(f):
    """
    Read the flip-classifier processing record from an open h5 file.

    Args:
    f (h5py.File): open extraction h5.

    Returns:
    record (dict): the processing record. Files written before the structured
        record existed are reported as ``{'state': 'complete', 'legacy': True}``
        if they carry the old boolean attribute, and ``{'state': 'not_started'}``
        otherwise, so callers never have to special-case the old format.
    """
    if f.attrs.get(FLIP_RECORD_UPDATE_ATTR, False):
        return {
            'state': 'failed',
            'error': 'flip processing record update was interrupted',
        }

    valid = []
    corrupt = []
    for path in (FLIP_RECORD_PATH, FLIP_RECORD_JOURNAL_PATH):
        if path not in f:
            continue
        raw = f[path][()]
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            record = json.loads(raw)
            sequence = int(record.pop('_journal_sequence', 0))
            valid.append((sequence, record))
        except (TypeError, ValueError):
            corrupt.append(path)

    if valid:
        _, record = max(valid, key=lambda item: item[0])
        record.setdefault('state', 'failed')
        return record

    if corrupt:
        # A truncated record means the process died mid-write; treat the file as
        # being in an unknown state rather than as untouched.
        return {
            'state': 'failed',
            'error': 'processing record is corrupt: {}'.format(', '.join(corrupt)),
        }

    if f.attrs.get(LEGACY_FLIP_ATTR, False):
        return {'state': 'complete', 'legacy': True,
                'note': 'processed before structured records existed; the '
                        'classifier and settings used are not recorded'}

    return {'state': 'not_started'}


def write_flip_record(f, record, fault_inject=None):
    """
    Write (or replace) the flip-classifier processing record in an open h5 file.

    Updates use a two-slot journal. The legacy Boolean safety latch and an
    update sentinel are flushed first; then only the inactive record slot is
    replaced. The sentinel is cleared only after the new slot is durable. At
    every interruption point, an older app sees the Boolean as true and this
    reader sees either the prior valid slot or a fail-closed interrupted update.

    Args:
    f (h5py.File): open extraction h5, opened for writing.
    record (dict): JSON-serialisable processing record.
    fault_inject (callable): test-only hook called after each durable update
        stage with its stage name.
    """

    def checkpoint(stage):
        if fault_inject is not None:
            fault_inject(stage)

    # Older app versions understand only this Boolean. It must be true for
    # in_progress, failed, and complete because frames may have been touched in
    # every one of those states.
    f.attrs[LEGACY_FLIP_ATTR] = True
    f.attrs[FLIP_RECORD_UPDATE_ATTR] = True
    f.flush()
    checkpoint('after_sentinel_flush')

    slots = (FLIP_RECORD_PATH, FLIP_RECORD_JOURNAL_PATH)
    latest_path = None
    latest_sequence = -1
    for path in slots:
        if path not in f:
            continue
        try:
            raw = f[path][()]
            if isinstance(raw, bytes):
                raw = raw.decode()
            existing = json.loads(raw)
            sequence = int(existing.get('_journal_sequence', 0))
        except (TypeError, ValueError):
            continue
        if sequence > latest_sequence:
            latest_path = path
            latest_sequence = sequence

    target_path = (
        FLIP_RECORD_JOURNAL_PATH
        if latest_path == FLIP_RECORD_PATH
        else FLIP_RECORD_PATH
    )
    if target_path in f:
        del f[target_path]
        f.flush()
    checkpoint('after_inactive_slot_clear')

    stored = dict(record)
    stored['_journal_sequence'] = latest_sequence + 1
    f.create_dataset(target_path, data=json.dumps(stored))
    checkpoint('after_new_slot_create')
    f.flush()
    checkpoint('after_new_slot_flush')

    del f.attrs[FLIP_RECORD_UPDATE_ATTR]
    f.flush()
    checkpoint('after_sentinel_clear')


def check_flip_state(key, path, record):
    """
    Decide what to do with a session given its processing record.

    A completed session is skipped -- that is a normal, safe outcome, and the
    remaining sessions should still be processed. A session left mid-run is not:
    its frames are partly rotated and partly not, with no record of where the
    boundary was, so continuing would produce a file whose orientation is wrong
    in an unknown subset of frames. That is a hard failure, because the only
    correct responses are to restore a backup or re-extract from raw data.

    Args:
    key (str): display name of the session.
    path (str): path to the session's h5, quoted in the error.
    record (dict): the record returned by :func:`read_flip_record`.

    Returns:
    process (bool): True to process the session, False to skip it.

    Raises:
    FlipProcessingError: if the session was left in an unusable state.
    """
    state = record.get('state')

    if state == 'not_started':
        return True

    if state == 'complete':
        clf = record.get('classifier_sha256')
        detail = f' (classifier {clf[:12]})' if clf else ''
        warnings.warn(
            f'Flip classifier already applied to {key}{detail}; skipping. '
            'Rotating frames is its own inverse, so re-running would undo the '
            'existing correction.')
        return False

    if state == 'in_progress':
        corrected = record.get('frames_corrected')
        total = record.get('n_frames')
        extent = (f'{corrected} of {total} frames had been committed'
                  if corrected is not None and total is not None
                  else 'the number of frames already committed was not recorded')
        raise FlipProcessingError(
            f'{key} ({path}) has a flip-classifier run marked in_progress, '
            f'started {record.get("started", "at an unrecorded time")}. A '
            'previous run was interrupted partway through rewriting the frames '
            f'({extent}), so part of the session is rotated and part is not, '
            'and the file does not record where the boundary was.\n'
            'This cannot be resumed or re-run safely. Restore this session from '
            'a backup, or re-extract it from the raw depth data and apply the '
            'classifier to the fresh output.')

    raise FlipProcessingError(
        f'{key} ({path}) has a failed flip-classifier run recorded: '
        f'{record.get("error", "no detail recorded")}. '
        f'{record.get("frames_corrected", "an unrecorded number of")} frames had '
        'been committed when it stopped, so the session is partially corrected.\n'
        'Restore this session from a backup, or re-extract it from the raw depth '
        'data before retrying.')


def _extraction_complete(file_path: Path):
    config = yaml.safe_load(file_path.read_text())
    return config['complete']


def _find_extractions(data_path: str):
    files = Path(data_path).glob('**/*.h5')
    files = sorted(f for f in files if _extraction_complete(f.with_suffix('.yaml')))
    if len(set([f.name for f in files])) < len(files):
        files = {f.parent.name + '/' + f.name: f for f in files}
    else:
        files = {f.name: f for f in files}
    return files


class FlipClassifierWidget:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.sessions = _find_extractions(data_path)
        if len(self.sessions) == 0:
            raise ValueError(
                f"No completed extractions found in '{data_path}'. "
                "Make sure you have run the 'aggregate results' step in the "
                "extraction notebook and that input_dir points to the folder "
                "containing your extracted .h5 files (e.g. the "
                "aggregate_results/ directory)."
            )
        # self.selected_frame_ranges_dict = {k: [] for k in self.path_dict}
        self.selected_frame_ranges_dict = defaultdict(list)
        self.curr_total_selected_frames = 0

        self.session_select_dropdown = pn.widgets.Select(options=list(self.sessions), name='Session', value=list(self.sessions)[0])
        self.frame_num_slider = pn.widgets.IntSlider(name='Current Frame', start=0, end=1000, step=1, value=1)
        self.start_button = pn.widgets.Button(name='Start Range', button_type='primary')
        self.face_left_button = pn.widgets.Button(name='Facing Left', button_type='success', width=140, visible=False)
        self.face_right_button = pn.widgets.Button(name='Facing Right', button_type='success', width=140, visible=False)
        self.selected_ranges = pn.widgets.MultiSelect(name='Selected Ranges', options=[])
        self.delete_selection_button = pn.widgets.Button(name='Delete Selection', button_type='danger')
        self.curr_total_label = pn.pane.Markdown(f'Current Total Selected Frames: {self.curr_total_selected_frames}')

        self.facing_info = pn.pane.Markdown('To finish selection, click on direction the animal is facing', visible=False)
        self.facing_row = pn.Row(self.face_left_button, self.face_right_button)
        self.range_box = pn.Column(
            self.curr_total_label,
            pn.pane.Markdown('#### Selected Correct Frame Ranges'),
            self.selected_ranges,
            self.delete_selection_button
        )

        self.forward_button = pn.widgets.Button(name='Forward', button_type='primary', width=142)
        self.backward_button = pn.widgets.Button(name='Backward', button_type='primary', width=142)

        self.frame_advancer_row = pn.Row(self.backward_button, self.forward_button)

        self.widgets = pn.Column(
            self.session_select_dropdown,
            self.frame_num_slider,
            self.start_button,
            self.frame_advancer_row,
            self.facing_info,
            self.facing_row,
            self.range_box,
            width=325
        )

        self.frame_display = hv.DynamicMap(self.display_frame, streams=[self.frame_num_slider.param.value]).opts(
            frame_width=400, frame_height=400, aspect='equal', xlim=(0, 1), ylim=(0, 1)
        )

        self.start_button.on_click(self.start_stop_frame_range)
        self.face_left_button.on_click(lambda event: self.facing_range_callback(event, True))
        self.face_right_button.on_click(lambda event: self.facing_range_callback(event, False))
        self.delete_selection_button.on_click(self.on_delete_selection_clicked)

        self.forward_button.on_click(self.advance_frame)
        self.backward_button.on_click(self.rewind_frame)

        self.session_select_dropdown.param.watch(self.changed_selected_session, 'value')
        self.session_select_dropdown.value = list(self.sessions)[0]

    def display_frame(self, value):
        if hasattr(self, 'frames'):
            frame = self.frames[value]
        else:
            frame = None
        # set bounds for the image
        return hv.Image(frame, bounds=(0, 0, 1, 1)).opts(cmap="cubehelix")

    def advance_frame(self, event):
        if self.frame_num_slider.value < self.frame_num_slider.end:
            self.frame_num_slider.value += 1
    
    def rewind_frame(self, event):
        if self.frame_num_slider.value > 0:
            self.frame_num_slider.value -= 1

    def start_stop_frame_range(self, event):
        if self.start_button.name == 'Start Range':
            self.start = self.frame_num_slider.value
            self.start_button.name = 'Cancel Select'
            self.start_button.button_type = 'danger'
            self.face_left_button.visible = True
            self.face_right_button.visible = True
            self.facing_info.visible = True
            self.facing_info.object = f'To finish selection, click on direction the animal is facing. Start: {self.start}'
        else:
            self.start_button.name = 'Start Range'
            self.start_button.button_type = 'primary'
            self.face_left_button.visible = False
            self.face_right_button.visible = False
            self.facing_info.visible = False

    def facing_range_callback(self, event, left):
        self.stop = self.frame_num_slider.value
        if self.stop > self.start:
            self.update_state_on_selected_range(left)
            self.face_left_button.visible = False
            self.face_right_button.visible = False
            self.facing_info.visible = False
            self.start_button.name = 'Start Range'
            self.start_button.button_type = 'primary'

    def update_state_on_selected_range(self, left):
        selected_range = range(self.start, self.stop)

        beginning = 'L' if left else 'R'
        display_selected_range = f'{beginning} - {selected_range} - {self.session_select_dropdown.value}'

        self.curr_total_selected_frames += len(selected_range)
        self.curr_total_label.object = f'Current Total Selected Frames: {self.curr_total_selected_frames}'

        self.selected_frame_ranges_dict[self.session_select_dropdown.value].append((left, selected_range))

        self.selected_ranges.options = self.selected_ranges.options + [display_selected_range]
        self.selected_ranges.value = []

        self.save_frame_ranges()

    def on_delete_selection_clicked(self, event):
        selected_range = self.selected_ranges.value
        if selected_range:
            vals = selected_range[0].split(' - ')
            delete_key = vals[2]
            direction = vals[0] == 'L'
            range_to_delete = eval(vals[1])

            to_drop = (direction, range_to_delete)
            self.selected_frame_ranges_dict[delete_key].remove(to_drop)

            self.curr_total_selected_frames -= len(range_to_delete)
            self.curr_total_label.object = f'Current Total Selected Frames: {self.curr_total_selected_frames}'

            l = list(self.selected_ranges.options)
            l.remove(selected_range[0])

            self.selected_ranges.options = l
            self.selected_ranges.value = []

    def changed_selected_session(self, event):
        if self.start_button.name == 'Cancel Select':
            self.start_button.name = 'Start Range'
            self.start_button.button_type = 'primary'

        self.frame_num_slider.value = 1
        self.widgets.loading = True

        with h5py.File(self.sessions[event.new], mode='r') as f:
            self.frame_num_slider.end = f['frames'].shape[0] - 1
            self.frames = f['frames'][()]

        self.widgets.loading = False

        self.frame_num_slider.value = 0

    def show(self):
        return pn.Row(self.widgets, self.frame_display)

    def save_frame_ranges(self):
        with open(self.training_data_path, 'wb') as f:
            pickle.dump((self.sessions, dict(self.selected_frame_ranges_dict)), f)
            
    def apply_flip_classifier(self, clf_path, chunk_size=4000, chunk_overlap=0,
                              smoothing=51, frame_path='frames', fps=30,
                              write_movie=False, verbose=True):
        """
        Apply a trained flip classifier on previously extracted data to flip the mice to the correct orientation.

        This rewrites the frames in place, and rotating a frame by 180 degrees is
        its own inverse, so applying the classifier twice silently undoes it. To
        make that impossible, each session carries a processing record at
        ``metadata/processing/flip_classifier`` recording which classifier (by
        SHA-256) and settings were used, and one of three states:

        * ``in_progress`` -- written before the first frame is touched. If the
          process dies here, the record survives and marks the session as
          partially corrected.
        * ``complete`` -- every frame was classified and committed.
        * ``failed`` -- the run raised, was interrupted, or finished with frames
          left unclassified.

        The record also carries the package name, version and best-effort git
        SHA, the classifier's filename and SHA-256, the chunk/overlap/smoothing
        settings, frame counts, whether the preview movie was written, and a
        snapshot (plus hash) of the extraction policies the correction was
        applied on top of. The extraction's own provenance record is read but
        never modified: it describes how the frames were produced, which is
        still true afterwards.

        Only a session with no record at all (or one written by an older version
        that used the ``flip_classifier_applied`` attribute) is processed. A
        completed session is skipped and the rest continue; a session left
        ``in_progress`` or ``failed`` raises, because its frames are partly
        rotated with no record of where the boundary lies.

        Args:
        clf_path (str): path to the pickled flip classifier to apply.
        chunk_size (int): size of frame chunks to process in batches.
        chunk_overlap (int): number of frames to overlap between chunks to improve classification precision between chunks.
        smoothing (int): kernel size of the applied median filter on the flip classifier results
        frame_path (str): h5 path of the frame stack to correct.
        fps (int): frame rate used when writing the optional preview movie.
        write_movie (bool): write a preview movie of the corrected frames.
        verbose (bool): displays the tqdm progress bars for each session.

        Raises:
        FlipProcessingError: if a session was left ``in_progress`` or ``failed``
            by an earlier run. Such a file is partly rotated with no record of
            where the boundary was, so it must be restored or re-extracted.
        Exception: anything raised while correcting frames is re-raised after a
            ``failed`` record has been written to the session's h5.

        Note:
        This does not provide atomic rollback. Frames are rewritten in place as
        each batch completes, so an interrupted run leaves the file partially
        corrected. What the record guarantees is that such a file is detected
        and refused, not that it can be repaired.
        """
        video_pipe = None
        clf_sha256 = _sha256(clf_path)
        app_version, app_sha = _app_version_and_sha()
        # NOTE: this used to be `simplefilter('ignore', UserWarning)` for the
        # whole loop, which also swallowed this function's own diagnostics --
        # the "already applied, skipping" message included, so a skipped session
        # was indistinguishable from a processed one. The suppression is now
        # applied only around the third-party call it was there for
        # (`get_flips`, which is what emits the sklearn/joblib noise).
        with warnings.catch_warnings():
            for key, path in tqdm(self.sessions.items(), desc='Flipping extracted sessions...'):
                # Open h5 file to stream and correct/update stored frames and scalar angles.
                path = str(path)
                with h5py.File(path, mode='a') as f:
                    output_movie = path.replace('.h5', '_flipped.mp4')

                    # Rotating frames and adding pi to the angle are their own
                    # inverses, so re-running this would silently undo a previous
                    # correction. Anything other than a clean not_started state
                    # means the frames are already modified, or modified to an
                    # unknown extent, and must not be touched again.
                    if not check_flip_state(key, path, read_flip_record(f)):
                        continue

                    frames = f[frame_path]

                    started = datetime.datetime.now().isoformat()
                    # snapshot, but never modify, the extraction's own provenance:
                    # it records how the frames were produced, which stays true
                    # after the flip correction
                    source_policies, source_policy_sha256 = _extraction_policy_snapshot(f)

                    def make_record(state, **extra):
                        """Build a processing record with the settings of this run."""
                        record = {
                            'package': 'moseq2-app',
                            'version': app_version,
                            'git_sha': app_sha,
                            'state': state,
                            'classifier_path': str(clf_path),
                            'classifier_filename': Path(str(clf_path)).name,
                            'classifier_sha256': clf_sha256,
                            'chunk_size': int(chunk_size),
                            'chunk_overlap': int(chunk_overlap),
                            'smoothing': int(smoothing),
                            'frame_path': frame_path,
                            'n_frames': int(len(frames)),
                            'started': started,
                            'source_extraction_policies': source_policies,
                            'source_extraction_policy_sha256': source_policy_sha256,
                        }
                        record.update(extra)
                        return record

                    # Claim the file before modifying a single frame. If this
                    # process dies mid-session the record stays 'in_progress',
                    # which is what stops the next run from treating a half
                    # corrected session as untouched and flipping it again.
                    write_flip_record(f, make_record('in_progress'))
                    # process every frame; this previously passed len(frames)-1
                    # and so never classified the final frame
                    frame_batches = list(gen_batch_sequence(len(frames), chunk_size, chunk_overlap))
                    n_batches = len(frame_batches)

                    # frames already corrected in this run. Overlapping batches
                    # share frames, and re-rotating one would undo its correction.
                    committed = np.zeros(len(frames), dtype=bool)
                    make_movie = write_movie
                    n_flipped = 0
                    # Preview success is UNKNOWN until the subprocess has been
                    # finalized and its exit status inspected. No terminal
                    # processing record is written before that happens.
                    preview_ok = False if write_movie else None
                    preview_error = None

                    try:
                        for b_i, batch in enumerate(tqdm(frame_batches, desc=f'Adjusting flips: {key}', disable=not verbose)):
                            batch_idx = np.asarray(list(batch))
                            frame_batch = frames[batch_idx]

                            # apply flip classifier on each batch to find which frames to flip.
                            # the whole batch (including any overlap) is classified so the
                            # median smoothing has full context at the chunk boundaries
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', UserWarning)
                                flips = get_flips(frame_batch.copy(), flip_file=clf_path, smoothing=smoothing)

                            # only commit frames not already handled by an earlier batch
                            new_local = np.flatnonzero(~committed[batch_idx])
                            # Defer the trailing overlap of every non-final batch to the
                            # NEXT batch, where those frames sit near the start and gain
                            # the future context the overlap exists to provide. Committing
                            # them here would decide them from the batch that has the least
                            # of that context.
                            if chunk_overlap > 0 and b_i < n_batches - 1:
                                keep = batch_idx[new_local] < batch_idx[-chunk_overlap]
                                new_local = new_local[keep]
                            if len(new_local) == 0:
                                continue

                            flip_local = new_local[flips[new_local]]
                            if len(flip_local) > 0:
                                # map batch-relative positions to absolute frame indices.
                                # the angle update previously used the batch-relative
                                # indices directly against the whole-session angle array,
                                # so every batch after the first rotated the angle of the
                                # wrong frames.
                                flip_abs = batch_idx[flip_local]

                                rotated = np.rot90(frames[flip_abs], k=2, axes=(1, 2))
                                frames[flip_abs] = rotated
                                frame_batch[flip_local] = rotated

                                angles = f['scalars/angle'][flip_abs]
                                f['scalars/angle'][flip_abs] = angles + np.pi
                                n_flipped += len(flip_abs)

                            committed[batch_idx[new_local]] = True

                            if make_movie:
                                try:
                                    # Writing frame batch to mp4 file
                                    video_pipe = write_frames_preview(output_movie,
                                                                      frame_batch,
                                                                      pipe=video_pipe,
                                                                      close_pipe=False,
                                                                      depth_min=0,
                                                                      depth_max=100,
                                                                      fps=fps,
                                                                      progress_bar=verbose)
                                except AttributeError as e:
                                    # the preview movie is optional; do not abort frame
                                    # correction (and do not mark the file complete) just
                                    # because the movie could not be written. Stop trying
                                    # to write it and keep correcting frames.
                                    warnings.warn(f'Could not generate flipped movie for {key}:{path}. '
                                                  'Continuing to correct frames without a preview movie.')
                                    print(e)
                                    make_movie = False
                                    preview_ok = False
                                    preview_error = '{}: {}'.format(type(e).__name__, e)

                        if video_pipe is not None:
                            try:
                                _, stderr = video_pipe.communicate()
                                returncode = video_pipe.returncode
                                if returncode == 0 and preview_error is None:
                                    preview_ok = True
                                elif returncode != 0:
                                    if isinstance(stderr, bytes):
                                        stderr = stderr.decode(errors='replace')
                                    detail = str(stderr).strip() if stderr else ''
                                    preview_error = (
                                        'preview subprocess exited with status {}{}'
                                        .format(
                                            returncode,
                                            ': ' + detail if detail else '',
                                        )
                                    )
                                    warnings.warn(
                                        'Could not finalize flipped movie for '
                                        '{}:{}. Continuing with corrected '
                                        'frames; the preview failure is '
                                        'recorded.'.format(key, path)
                                    )
                            except Exception as e:
                                preview_ok = False
                                preview_error = '{}: {}'.format(
                                    type(e).__name__, e
                                )
                                warnings.warn(
                                    'Could not finalize flipped movie for '
                                    '{}:{}. Continuing with corrected frames; '
                                    'the preview failure is recorded.'.format(
                                        key, path
                                    )
                                )
                            finally:
                                video_pipe = None
                    except BaseException as e:
                        # Record the failure in the file itself before it propagates.
                        # The frames written so far are already corrected and the rest
                        # are not, so the session must never be picked up again as if
                        # it were untouched. KeyboardInterrupt is caught too -- a
                        # user pressing ctrl-c leaves exactly the same partial state.
                        write_flip_record(f, make_record(
                            'failed',
                            frames_corrected=int(committed.sum()),
                            frames_flipped=int(n_flipped),
                            preview_requested=bool(write_movie),
                            preview_written=preview_ok,
                            preview_error=preview_error,
                            failed_at=datetime.datetime.now().isoformat(),
                            error='{}: {}'.format(type(e).__name__, e)))
                        raise

                    # only mark the session complete if every frame was corrected;
                    # a partially-processed file must not be flagged done and then
                    # permanently skipped on the next run
                    if committed.all():
                        write_flip_record(f, make_record(
                            'complete',
                            frames_corrected=int(committed.sum()),
                            frames_flipped=int(n_flipped),
                            preview_requested=bool(write_movie),
                            preview_written=preview_ok,
                            preview_error=preview_error,
                            finished=datetime.datetime.now().isoformat()))
                    else:
                        # every frame should have been claimed by some batch; if not,
                        # the batching is wrong and the session is partially corrected
                        write_flip_record(f, make_record(
                            'failed',
                            frames_corrected=int(committed.sum()),
                            frames_flipped=int(n_flipped),
                            preview_requested=bool(write_movie),
                            preview_written=preview_ok,
                            preview_error=preview_error,
                            failed_at=datetime.datetime.now().isoformat(),
                            error='{} of {} frames were never classified'.format(
                                int((~committed).sum()), len(frames))))
                        warnings.warn(
                            f'{key}: {int((~committed).sum())} of {len(frames)} frames '
                            'were never classified, so the session is only partially '
                            'corrected. It has been marked failed and will not be '
                            'reprocessed; re-extract it before retrying.')

    @property
    def training_data_path(self):
        return str(self.data_path.parent / 'flip-training-frame-ranges.p')
