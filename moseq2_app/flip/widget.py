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

#: h5 attribute used before the structured record existed. Still honoured when
#: reading so files processed by the older code are not reprocessed.
LEGACY_FLIP_ATTR = 'flip_classifier_applied'


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
    if FLIP_RECORD_PATH in f:
        raw = f[FLIP_RECORD_PATH][()]
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            record = json.loads(raw)
        except ValueError:
            # a truncated record means the process died mid-write; treat the
            # file as being in an unknown state rather than as untouched
            return {'state': 'failed', 'error': 'processing record is corrupt'}
        record.setdefault('state', 'failed')
        return record

    if f.attrs.get(LEGACY_FLIP_ATTR, False):
        return {'state': 'complete', 'legacy': True,
                'note': 'processed before structured records existed; the '
                        'classifier and settings used are not recorded'}

    return {'state': 'not_started'}


def write_flip_record(f, record):
    """
    Write (or replace) the flip-classifier processing record in an open h5 file.

    The record is flushed immediately. This matters: the whole point of the
    ``in_progress`` state is that it survives the process being killed halfway
    through rewriting the frames.

    Args:
    f (h5py.File): open extraction h5, opened for writing.
    record (dict): JSON-serialisable processing record.
    """
    if FLIP_RECORD_PATH in f:
        del f[FLIP_RECORD_PATH]
    f.create_dataset(FLIP_RECORD_PATH, data=json.dumps(record))
    # keep the legacy attribute in sync so older moseq2-app installs still see
    # a completed file as completed
    f.attrs[LEGACY_FLIP_ATTR] = record.get('state') == 'complete'
    f.flush()


def _describe_flip_state(key, record):
    """Explain why a session is being skipped, in terms the user can act on."""
    state = record.get('state')
    if state == 'complete':
        clf = record.get('classifier_sha256')
        detail = f' (classifier {clf[:12]})' if clf else ''
        return (f'Flip classifier already applied to {key}{detail}; skipping. '
                'Rotating frames is its own inverse, so re-running would undo '
                'the existing correction.')
    if state == 'in_progress':
        return (f'{key} has a flip-classifier run marked in_progress: a previous '
                'run was interrupted partway through rewriting the frames, so '
                'part of the session is corrected and part is not. This cannot '
                'be resumed safely -- re-extract the session and apply the '
                'classifier to the fresh output.')
    return (f'{key} has a failed flip-classifier run recorded '
            f'({record.get("error", "no detail recorded")}). Its frames may be '
            'partially corrected. Re-extract the session before retrying.')


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

        Only a session with no record at all (or one written by an older version
        that used the ``flip_classifier_applied`` attribute) is processed; the
        rest are skipped with an explanation.

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
        Exception: anything raised while correcting frames is re-raised after a
            ``failed`` record has been written to the session's h5.
        """
        video_pipe = None
        clf_sha256 = _sha256(clf_path)
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
                    record = read_flip_record(f)
                    if record.get('state') != 'not_started':
                        warnings.warn(_describe_flip_state(key, record))
                        continue

                    frames = f[frame_path]

                    started = datetime.datetime.now().isoformat()

                    def make_record(state, **extra):
                        """Build a processing record with the settings of this run."""
                        record = {
                            'state': state,
                            'classifier_path': str(clf_path),
                            'classifier_sha256': clf_sha256,
                            'chunk_size': int(chunk_size),
                            'chunk_overlap': int(chunk_overlap),
                            'smoothing': int(smoothing),
                            'frame_path': frame_path,
                            'n_frames': int(len(frames)),
                            'started': started,
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
                            finished=datetime.datetime.now().isoformat()))
                    else:
                        # every frame should have been claimed by some batch; if not,
                        # the batching is wrong and the session is partially corrected
                        write_flip_record(f, make_record(
                            'failed',
                            frames_corrected=int(committed.sum()),
                            frames_flipped=int(n_flipped),
                            failed_at=datetime.datetime.now().isoformat(),
                            error='{} of {} frames were never classified'.format(
                                int((~committed).sum()), len(frames))))
                        warnings.warn(
                            f'{key}: {int((~committed).sum())} of {len(frames)} frames '
                            'were never classified, so the session is only partially '
                            'corrected. It has been marked failed and will not be '
                            'reprocessed; re-extract it before retrying.')

                    # Check if video is done writing. If not, wait.
                    if video_pipe is not None:
                        video_pipe.communicate()
                        video_pipe = None

    @property
    def training_data_path(self):
        return str(self.data_path.parent / 'flip-training-frame-ranges.p')