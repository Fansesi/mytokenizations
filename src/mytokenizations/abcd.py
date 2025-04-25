"""Class for MIDI to string representation (abcd representation) conversion.

Uses ``TokSequence.bytes`` to store the abcd representation. This is because 
I find it unnecessary to subclass TokSequence. If I need to do it in the future,
that is an option though.

Created ``ids`` is similar to RelTok, 2D tokenization with three tokens, in 
order: "Pitch", "SameOrNot", "Duration".

NOTE: I have not removed tokenization steps related to Tempo, Velocity and Time Signature.
Therefore, if you would like to implement another tokenizer that uses those,
you may do so by looking into this source code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List
from copy import deepcopy
import numpy as np
from collections import Counter
import re

from symusic import (
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
    Track,
)

from miditok.classes import Event, TokenizerConfig, TokSequence
from miditok.constants import (
    ADD_TRAILING_BARS,
    DEFAULT_VELOCITY,
    MIDI_INSTRUMENTS,
    TIME_SIGNATURE,
    TEMPO,
    USE_BAR_END_TOKENS,
)
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import (
    compute_ticks_per_bar,
    compute_ticks_per_beat,
    get_bars_ticks,
    get_beats_ticks,
    get_score_ticks_per_beat,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def to_interval(array: np.ndarray):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    assert len(array.shape) == 1, "Array should be 1D."
    return (np.append(array, 0) - np.concatenate((np.zeros((1)), array)))[1:-1].astype(
        np.int16
    )


class ABCDNotation:
    """Conversion for interval to abcd and vice versa."""

    def __init__(self, music: List[List[int]] | str | np.ndarray):
        """
        Args:
            music (List[List[int]] | str | np.ndarray):

              interval repr.: List[[interval, duration], ...]
              NOTE: the first interval is discarded.

              string repr.: e.g. 'aaccd'.
                a: up a half step.
                b: down a half step.
                c: duration.
                d: separator for the note.
        """
        if isinstance(music, np.ndarray):
            music = music.tolist()

        self.music = music

    @property
    def intdur(self) -> List[List[int]]:
        """Convert abcd to interval_duration.

        Returns:
            List[List[int]]: interval_duration representation.
        """
        if isinstance(self.music, list):
            return self.music

        return self._string_to_intdur(self.music)

    @property
    def string(self) -> str:
        """Convert interval_duration in the music to abcd.

        Returns:
            str: abcd representation.
        """
        if isinstance(self.music, str):
            return self.music

        text = ""
        for data in self.music:
            text += self._intdur_to_string(data)

        return text

    @string.setter
    def string(self, value: str):
        if not isinstance(value, str):
            ValueError(f"Given value ({value}) is not suitable for setting string.")
        self.music = value

    @intdur.setter
    def intdur(self, value: List[List[int]]):
        if not isinstance(value, list):
            ValueError(f"Given value ({value}) is not suitable for setting intdur.")
        self.music = value

    def _string_to_intdur(self, string: str) -> List[List[int]]:
        final = []

        # a d cc d bb d ccc d d c d - no, you can't use split("d") here.
        # what happens if I use d as the only note separator. Not twice but
        # only once. aacccd bbbcd cd - this also works.

        for note in string.split("d"):
            if note == "":
                continue
            num = 0
            counter = Counter(note)
            if counter["a"] > 0:
                num = counter["a"]  # / 2
            elif counter["b"] > 0:
                num = -counter["b"]  # / 2

            final.append([num, counter["c"]])

        return final

    def _intdur_to_string(self, interval_duration: List[float]) -> str:
        assert (
            len(interval_duration) == 2
        ), "interval_duration pair is not correctly set."
        # NOTE: I just can't remember why I multiplied this by two.
        # num = int(interval_duration[0] * 2)
        num = int(interval_duration[0])
        text = ""
        if num > 0:
            text += "a" * num
        if num < 0:
            text += "b" * abs(num)

        # NOTE: no need for the first d actually.
        # text += "d"
        text += "c" * int(interval_duration[1])
        text += "d"

        return text

    def pitches_durations(self, start_note: int) -> List[List[int]]:
        """Convert intervals into notes.

        Args:
            start_note (int): starting note.

        Returns:
            List[int]: [[pitch_num, duration_num]]
        """

        if isinstance(self.music, str):
            self.music = self.intdur

        notes = [[start_note, self.music[0][1]]]
        for i in range(len(self.music)):
            next_note = notes[-1][0] + int(self.music[i][0])
            notes.append([np.clip(next_note, 0, 127), self.music[i][1]])

        return notes

    def midi(self, start_note: int) -> PrettyMIDI:
        pd = np.array(self.pitches_durations(46))
        pitches, durations = pd[:, 0], pd[:, 1]
        durations *= 256  # duration num to tick conversion
        durations = durations.astype(np.int32)
        return create_score(pitches[1:], durations[1:], 120)

    @staticmethod
    def from_2D(data: List[List[int]] | np.ndarray) -> ABCDNotation:
        """Creates a ABCDNotation object from the tokenized data.

        Args:
            data (List[List[int]]): unprocessed tokenization by Miditok data.

        Returns:
            ABCDNotation
        """
        # Too much fancy indexing hurts the soul.
        data = np.array(data)
        intervals = IntervalCalcs.to_interval(data[:, 0])
        durations = map_to_range(data[:, 2], 4, 131, 1, 16)

        # The first interval means nothing, it is only to be able to stack
        # with durations. ABCDNotation handles this problem by ignoring the
        # first interval in the conversion process.
        intervals = np.concatenate([[0], intervals])
        return ABCDNotation(np.stack([intervals, durations], axis=1).tolist())

    @staticmethod
    def clean_abcd(s: str) -> Tuple[str, int]:
        """Cleans a given abcd music as string.

        * If there are more than one d in a block make them single.
        (Because it means that it is an empty note and we don't want that.)
        * Add d after every block of c so that the note is finished.
        * Change a/b throughout the block if the block starts with a/b.
        (or maybe take majority?)
        * Remove the rest if string doesn't end with c or d.

        Args:
            s (str): abcd music, possibly from a generation.

        Returns:
            Tuple[str, float]: a heavily modified version of the abcd string that is ready for
            outputting as MIDI and number of edits.
        """
        separated = re.findall(r"[ab]+|c+|d", s)

        for block in separated:
            if block[0] == "d" and len(block) > 1:
                block = "d"

        for i in range(len(separated) - 1):
            if separated[i][0] == "c" and separated[i + 1] != "d":
                # because we are going to merge everything together,
                # there is no need for creating an extra element in the list.
                separated[i] += "d"

            if "a" in separated[i] and "b" in separated[i]:
                # take the first one
                # separated[i] = separated[i][0] * len(separated[i])

                # or take the majority
                counter = Counter(separated[i])
                counts = counter.most_common(2)
                separated[i] = counts[0][0] * len(separated[i])

        separated = "".join(separated)

        if separated[-1] == "c":
            separated += "d"

        if separated[-1] != "d":
            last_d_idx = list(reversed(separated)).index("d")
            separated = separated[:-last_d_idx]

        return separated

    @staticmethod
    def error_num(s: str) -> Tuple[str, int]:
        """Same as clean_abcd but does not modify string, only count edits
        if we were to clean this string.

        * If there are more than one d in a block make them single.
        (Because it means that it is an empty note and we don't want that.)
        * Add d after every block of c so that the note is finished.
        * Change a/b throughout the block if the block starts with a/b.
        (or maybe take majority?)
        * Remove the rest if string doesn't end with c or d.

        Args:
            s (str): abcd music, possibly from a generation.

        Returns:
           float: number of edits.
        """
        separated = re.findall(r"[ab]+|c+|d", s)
        edit_count = 0

        for block in separated:
            if block[0] == "d" and len(block) > 1:
                block = "d"
                edit_count += len(block) - 1

        for i in range(len(separated) - 1):
            if separated[i][0] == "c" and separated[i + 1] != "d":
                # because we are going to merge everything together,
                # there is no need for creating an extra element in the list.
                separated[i] += "d"
                edit_count += 1

            if "a" in separated[i] and "b" in separated[i]:
                counter = Counter(separated[i])
                counts = counter.most_common(2)
                # separated[i] = counts[0][0] * len(separated[i])

                edit_count += counts[1][1]

        separated = "".join(separated)

        if separated[-1] == "c":
            separated += "d"
            edit_count += 1

        if separated[-1] != "d":
            last_d_idx = list(reversed(separated)).index("d")
            # separated = separated[:-last_d_idx]
            edit_count += last_d_idx

        return edit_count


class ABCDTokenizer(MusicTokenizer):
    """abcd representation where,
        a: up a half step.
        b: down a half step.
        c: duration.
        d: separator for the note.

    Uses intervalic relationship. Only works for monophonic, single-track music.
    Uses miditok to utilize MIDI to ids conversion, then self.abcd_to_score and
    self.score_to_abcd to store abcd representation as bytes in a TokSequence.

    Examples:
    * `aadccd`: means go up 0.5+0.5 steps and play the note which has length 2 units.
        * `bdcccd`: means go down 0.5 steps and play the note which has length 3 units.
        * `cd`: play the current note 1 unit.

    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: str | Path = None,
    ):
        super().__init__(tokenizer_config, params)

    def _tweak_config_before_creating_voc(self):
        self.config.use_chords = False
        self.config.use_rests = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        self.config.use_pitch_intervals = False
        # self.config.delete_equal_successive_tempo_changes = True
        self.config.program_changes = False

        self._disable_attribute_controls()

        self.config.one_token_stream_for_programs = True
        self.config.one_token_stream = True

        self.config.use_time_signatures = False
        self.config.use_tempos = False

        # This implementation is for single track instruments
        self.config.use_programs = False

        token_types = ["Pitch", "SameOrNot", "Duration"]
        self.vocab_types_idx = {
            type_: idx for idx, type_ in enumerate(token_types)
        }  # used to find the event location and augmentation.

    def abcd_dataset(self, files_paths: str | Path | Sequence[str | Path]):
        """Convert a path full of midis to abc representations.

        Args:
            files_paths (str | Path | Sequence[str  |  Path]): file paths.
        """
        raise NotImplementedError("ALE related functionality will create a dataset.")

    def score_to_abcd(self, score: Score) -> TokSequence:
        """Convert score into abcd representation.
        Uses ``self._score_to_tokens``. Operates on ``ids`` created from
        ``self._score_to_tokens``. Additionally uses ABCDNotation to do
        string conversion.

        Args:
            score (Score): symusic.Score object.

        Returns:
            TokSequence: TokSequence that holds the abcd representation as bytes.
        """
        # tokseq = self._score_to_tokens(score)
        tokseq = self.encode(score)
        ids = np.array(tokseq.ids)

        # Because we are working on monophonic pieces, we ignore SameOrNot tokens:
        intervals = to_interval(ids[:, 0])
        # The first interval means nothing, it is only to be able to stack
        # with durations. ABCDNotation handles this problem by ignoring the
        # first interval in the conversion process.
        intervals = np.concatenate([[0], intervals])

        # TODO: Consider mapping the durations into smaller range.
        # See ABCDNotation.from_2D() and map_to_range.

        # NOTE: We are not going to use first four tokens in the vocabulary:
        # (`PAD_None': 0, 'BOS_None': 1, 'EOS_None': 2, 'MASK_None: 3`)
        durations = ids[:, 2] - 4

        abcd = ABCDNotation(np.stack([intervals, durations], axis=1))
        tokseq.bytes = abcd.string
        return tokseq

    def abcd_to_score(self, tokens: TokSequence | str, start_note: int = 46) -> Score:
        if isinstance(tokens, TokSequence):
            if tokens.bytes == "":
                raise ValueError("Tokens do not have abcd representation.")
            tokens = tokens.bytes
        if not isinstance(tokens, str):
            raise ValueError(f"Incompatible tokens type: {type(tokens)}.")

        abcd = ABCDNotation(tokens)
        abcd.string = self._preprocess_abcd(abcd.string)
        pd = np.array(abcd.pitches_durations(start_note))
        pitches = pd[:, [0]].flatten()
        durations = pd[:, [1]]
        _fours = np.zeros((pd.shape[0], 1), dtype=pd.dtype) + 4

        # Need to clamp because of vocabulary overflow
        clipped_pitches = np.clip(
            pitches, self.config.pitch_range[0], self.config.pitch_range[1] - 1
        )
        if not np.array_equal(pitches, clipped_pitches):
            print("[WARNING] Pitches have been clamped to fit into vocabulary.")
            pitches = clipped_pitches

        # Conversion from MIDI Pitch to ids.
        pitches = np.array(list(self.vocab[0][f"Pitch_{i}"] for i in pitches))
        pitches = np.expand_dims(pitches, 1)

        # psd: pitch, sameornot, duration
        # Adding 4 to durations as well. See self.score_to_abcd
        psd = np.hstack((pitches, _fours, durations + 4))
        tokseq = TokSequence(ids=psd.tolist())

        self.complete_sequence(tokseq)
        return self._tokens_to_score(tokseq)

    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:

        # Unsqueeze tokens in case of one_token_stream
        if self.config.one_token_stream_for_programs:  # ie single token seq
            tokens = [tokens]
        for i in range(len(tokens)):
            tokens[i] = tokens[i].tokens

        score = Score(self.time_division)

        # RESULTS
        tracks: Dict[int, Track] = {}
        tempo_changes, time_signature_changes = [Tempo(-1, self.default_tempo)], []
        # tempo_changes[0].tempo = -1

        def is_track_empty(track: Track) -> bool:
            return (
                len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0
            )

        # operates only on 4/4
        time_signature_changes = [TimeSignature(0, 4, 4)]

        current_program = 24
        current_tick = 0

        dynamic_duration_list = []
        for si, seq in enumerate(tokens):
            # Decode tokens
            for k, time_step in enumerate(seq):
                nb_tok_to_check = 3

                if any(
                    tok.split("_")[1] == "None" for tok in time_step[:nb_tok_to_check]
                ):
                    continue  # Either padding, mask: error of prediction or end of sequence anyway

                # If it's a rest, then no need to create a Note.
                # And also current tick should be updated based on the duration
                # of the rest and previous note.
                if time_step[0].split("_")[1] == "REST" and k != 0:
                    # adding the previous note duration.
                    current_tick += self._time_token_to_ticks(
                        seq[k - 1][2].split("_")[1], time_division
                    )

                    # adding the rest duration
                    current_tick += self._time_token_to_ticks(
                        time_step[2].split("_")[1], time_division
                    )
                    continue

                # Note attributes
                pitch = int(time_step[0].split("_")[1])
                # vel = int(time_step[1].split("_")[1])
                duration = self._time_token_to_ticks(
                    time_step[2].split("_")[1], self.time_division
                )
                if k == 0:
                    dynamic_duration_list = [duration]  # initializing.
                    current_tick = duration

                # if 0, cursor should be moved.
                # if 1, cursor should be at the same place.
                same_or_not = int(time_step[1].split("_")[1])

                if (
                    same_or_not == 0
                    and k != 0
                    and seq[k - 1][0].split("_")[1] != "REST"
                    and seq[k - 1][0].split("_")[1] != "None"
                ):
                    # If we should move the cursor, we move it by adding
                    # the previous min(notes duration). It shouldn't be the first note.
                    # and it shouldn't be a rest note because otherwise we would double
                    # the duration of rest.

                    current_tick += min(dynamic_duration_list)
                    dynamic_duration_list = []
                    # current_tick += self._token_duration_to_ticks(
                    #    seq[k - 1][2].split("_")[1], time_division
                    # )

                dynamic_duration_list.append(duration)

                if current_program not in tracks.keys():
                    tracks[current_program] = Track(
                        program=current_program,
                        is_drum=False,
                        name=MIDI_INSTRUMENTS[current_program]["name"],
                    )

                tracks[current_program].notes.append(
                    Note(current_tick, duration, pitch, 100)
                )

                # Tempo, adds a Tempo if necessary
                # tempo = time_step[self.vocab_types_idx["Tempo"]].split("_")[1]

                # if si == 0 and self.config.use_tempos and tempo != "None":
                #     tempo = float(tempo)
                #     if tempo != tempo_changes[-1].tempo:
                #         tempo_changes.append(Tempo(current_tick, tempo))

                # time_sig = time_step[self.vocab_types_idx["TimeSig"]].split("_")[1]

                # Time Signature, adds a TimeSignatureChange if necessary
                # if self.config.use_time_signatures and time_sig != "None":
                #     num, den = self._parse_token_time_signature(time_sig)
                #     if (
                #         num != time_signature_changes[-1].numerator
                #         or den != time_signature_changes[-1].denominator
                #     ):  # checking whether the new ts is the same as the last.
                #         # tick from bar of ts change
                #         final_time_sig = TimeSignature(current_tick, num, den)
                #         if si == 0:
                #             time_signature_changes.append(final_time_sig)

                #         # ticks_per_bar = self._compute_ticks_per_bar(
                #         #    time_sig, time_division
                #         # )

        # del tempo_changes[0]
        # if len(tempo_changes) == 0 or (
        #     tempo_changes[0].time != 0
        #     and round(tempo_changes[0].tempo, 2) != self.default_tempo
        # ):
        #     tempo_changes.insert(0, Tempo(0, self.default_tempo))
        # elif round(tempo_changes[0].tempo, 2) == self.default_tempo:
        #     tempo_changes[0].time = 0

        # if self.config.one_token_stream_for_programs:
        #     score.tracks = list(tracks.values())
        # else:
        score.tracks = [tracks[current_program]]

        score.tempos = tempo_changes
        score.time_signatures = time_signature_changes
        return score

    def _score_to_tokens(
        self,
        score: Score,
        attribute_controls_indexes: (
            Mapping[int, Mapping[int, Sequence[int] | bool]] | None
        ) = None,
    ) -> TokSequence | list[TokSequence]:

        # I'd like to work on single instrument right now.
        # and because that, I'll take the self._one_token_stream
        assert len(score.tracks) == 1

        # Create events list
        all_events: List[Event] = []

        # Global events (Tempo, TimeSignature)
        # We don't use these because ABCDTokenizer is quite simplistic.
        # all_events += self._create_global_events(score)

        # Compute ticks_per_beat sections depending on the time signatures
        # This has to be computed several times, in preprocess after resampling & here.
        if (
            not self._note_on_off
            or (self.config.use_sustain_pedals and self.config.sustain_pedal_duration)
            or self.config.use_chords
            or self.config.use_pitch_intervals
        ):
            if self.config.use_time_signatures and len(score.time_signatures) > 0:
                ticks_per_beat = get_score_ticks_per_beat(score)
            else:
                ticks_per_beat = np.array([[score.end(), self.time_division]])
        else:
            ticks_per_beat = None

        ticks_bars = get_bars_ticks(score, only_notes_onsets=True)
        ticks_beats = get_beats_ticks(score, only_notes_onsets=True)

        # Adds track tokens in other words,
        # Pitch, Velocity, Duration, NoteOn, NoteOff and optionally Chord
        track_events = self._create_track_events(
            score.tracks[0],
            ticks_per_beat,
            score.ticks_per_quarter,
            ticks_bars,
            ticks_beats,
        )

        # track_events are ordered as follows: Pitch, Duration.
        # Based on these I'm going to add Rest Events in between notes.
        track_events = self._create_inbetween_rests(track_events)
        all_events += track_events

        all_events.sort(key=lambda x: (x.time, self._order(x)))

        # Add time events, below, we are not using += because this function
        # inserts (not inplace) time events
        all_events = self._add_time_events(all_events, self.time_division)

        tok_sequence = TokSequence(events=all_events)
        self.complete_sequence(tok_sequence)  # creates the missing representations

        return tok_sequence

    @staticmethod
    def _order(x: Event) -> int:
        """Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """

        if x.type_ == "Tempo":
            return 0
        elif x.type_ == "TimeSig":
            return 1
        elif x.type_ in ["Pitch", "Velocity", "Duration"]:
            # These three shouldn't change their position
            return 2
        elif x.type_ == "Rest":
            # But if there exists a rest it should be
            # after a (Pitch, Velocity, Duration) trio
            return 3
        else:
            return 10

    def _add_time_events(
        self, events: List[Event], time_division: int
    ) -> List[List[Event]]:
        """Below is taken and revised from RelTok.

        Takes a sequence of note events, and insert (not inplace) time tokens
        (TimeShift, Rest) to complete the sequence.
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: SameOrNot
            2: Duration

        :param events: note events to complete.
        :return: the same events, with time events inserted."""

        # initializing some variables for later usage and storage
        all_events = []
        previous_tick = 0
        current_time_sig = TIME_SIGNATURE
        current_tempo = TEMPO

        for e, event in enumerate(events):
            if event.time != previous_tick:
                previous_tick = event.time
                same_or_not_value = 0  # if it's the same as before 1, otherwise 0
            else:
                same_or_not_value = 1  # if it's the same as before 1, otherwise 0

            # Starting note
            if e == 0:
                same_or_not_value = 0

            # if event.type_ == "TimeSig":
            #     current_time_sig = list(map(int, event.value.split("/")))

            # elif event.type_ == "Tempo":
            #     current_tempo = event.value

            elif event.type_ == "Rest" and e + 2 < len(events):
                # Catching the placeholder rest Events.

                # TODO: experiment with rest=True and rest=False param.
                if event.time - event.desc == 0:
                    raise Exception("This notes duration value is 0!")

                # WARNING: This might cause trouble
                # because of the static time_division.
                dur_value, dur_ticks = self._time_ticks_to_tokens(
                    abs(event.time - event.desc),
                    time_division,
                    rest=False,
                )

                new_event = [
                    Event(type_="Pitch", value="REST", time=event.time),
                    # Event(type_="Velocity", value=0, time=event.time),
                    Event(type_="SameOrNot", value=0, time=event.time),
                    Event(
                        type_="Duration",
                        value=".".join(map(str, dur_value[0])),
                        time=event.time,
                    ),
                ]

                # if self.config.use_tempos:
                #     new_event.append(Event(type_="Tempo", value=current_tempo))
                # if self.config.use_time_signatures:
                #     new_event.append(
                #         Event(
                #             type_="TimeSig",
                #             value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                #         )
                #     )
                all_events.append(new_event)

            elif event.type_ == "Pitch" and e + 2 < len(events):
                # lg.debug(f"Velocity for this note is: {events[e + 1].value}")
                new_event = [
                    Event(type_="Pitch", value=event.value, time=event.time),
                    Event(type_="SameOrNot", value=same_or_not_value, time=event.time),
                    # Event(type_="Velocity", value=events[e + 1].value, time=event.time),
                    Event(type_="Duration", value=events[e + 2].value, time=event.time),
                    # We don't use Position and Bar values!
                    # Event(type="Position", value=current_pos, time=event.time),
                    # Event(type="Bar", value=current_bar, time=event.time),
                ]
                # if self.config.use_tempos:
                #     new_event.append(Event(type_="Tempo", value=current_tempo))
                # if self.config.use_time_signatures:
                #     new_event.append(
                #         Event(
                #             type_="TimeSig",
                #             value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                #         )
                #     )
                all_events.append(new_event)

        return all_events

    def _create_base_vocabulary(self):
        # Based on https://github.com/Natooz/MidiTok/blob/main/miditok/tokenizations/octuple.py#L353
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = [[] for _ in range(3)]

        # PITCH
        # Rests are a special kind of pitch value. It can be generated automatically,
        # whenever there is a difference between a NoteOff and a NoteOn there should be a rest there.
        # Music is cumulative!
        vocab[0] += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]
        vocab[0] += [f"Pitch_REST"]

        # VELOCITY
        # vocab[1] += [f"Velocity_{i}" for i in self.velocities]

        # SAME_AS_BEFORE_OR_NOT
        vocab[1] += [f"SameOrNot_{i}" for i in range(2)]  # basic yes or no

        # DURATION
        vocab[2] += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        ## POSITION
        # max_nb_beats = max(
        #    map(lambda ts: ceil(4 * ts[0] / ts[1]), self.time_signatures)
        # )
        # nb_positions = max(self.config.beat_res.values()) * max_nb_beats
        # vocab[3] += [f"Position_{i}" for i in range(nb_positions)]
        #
        ## BAR (positional encoding)
        # vocab[4] += [
        #    f"Bar_{i}"
        #    for i in range(self.config.additional_params["max_bar_embedding"])
        # ]

        # PROGRAM
        # if self.config.use_programs:
        #     vocab.append([f"Program_{i}" for i in self.config.programs])

        # TEMPO
        # if self.config.use_tempos:
        #     vocab.append([f"Tempo_{i}" for i in self.tempos])

        # TIME_SIGNATURE
        # if self.config.use_time_signatures:
        #     vocab.append([f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures])

        return vocab

    def tokens_errors(self, tokens: TokSequence | str) -> float | List[float]:
        """Returns the error ratio (lower is better).

        Operates on abcd representation. In this documentation, a group is
        defined as (interval, play, separator) == (a/b, c, d).

        * Checks if there is at least one c in every group.
        * Checks if there exists both a and b in group's interval section.
        * The sorting is messed up.

        NOTE: It is possible to have an error rate larger than 1.0.
        """
        if isinstance(tokens, TokSequence):
            if tokens.bytes == "":
                raise ValueError("Tokens are not converted into abcd representation.")
            tokens = tokens.bytes
        if not isinstance(tokens, str):
            raise ValueError(f"Incompatible tokens type: {type(tokens)}.")

        return ABCDNotation.error_num(tokens) / len(ABCDNotation.clean_abcd(tokens))

    def _preprocess_abcd(self, tokens: TokSequence | str):
        if isinstance(tokens, TokSequence):
            if tokens.bytes == "":
                raise ValueError("Tokens are not converted into abcd representation.")
            tokens = tokens.bytes
        if not isinstance(tokens, str):
            raise ValueError(f"Incompatible tokens type: {type(tokens)}.")

        return ABCDNotation.clean_abcd(tokens)

    def _create_token_types_graph(self):
        return {}

    def _create_inbetween_rests(
        self, track_events: List[Event], threshold: Optional[int] = None
    ):
        f"""Creates and inserts Rest events into correct position based on given track events.

        :param track_events: pitch, velocity, duration events.
        :param threshold: threshold to create rest events. Default to the minimumu duration
        that tokenizer handles.

        :return: the same events, with rest events inserted."""

        # time_note: Dict[Tuple[int, int], List[Event, Event, Event]] = {}
        # {(time_start, time_end) : [PitchEvent, VelocityEvent, DurationEvent]}

        def create_rest_event(start, end):
            """Start and end are in ticks. Although there are no rest type token
            in the vocabulary, this method returns the type as Rest. The
            value will be defined later on in a DurationEvent, so this Event is just
            a data placeholder."""
            return Event(
                type="Rest", value="not_defined_right_now", time=start, desc=end
            )

        if threshold == None:
            # if there are no special tokens ('PAD_None': 0, 'BOS_None': 1, 'EOS_None': 2, 'MASK_None': 3)
            # this will throw error or set the threshold too high.
            first_duration_token = 0
            for dur_tok in self.vocab[2]:
                if dur_tok.split("_")[1] != "None":
                    first_duration_token = dur_tok
            threshold = self._time_token_to_ticks(
                first_duration_token.split("_")[1], self.time_division
            )
            # lg.debug(threshold)

        # I can just look at 3k k=0,...
        start_end_times: List[Tuple[int, int]] = []
        for i in range(0, len(track_events), 3):
            data = (track_events[i].time, track_events[i].desc)
            if data not in start_end_times:
                start_end_times.append(data)

        # Creating the *contrary* list which makes up the full time
        # with the start_end_times list.
        contrary_start_end_times: List[Tuple[int, int]] = []
        for i in range(len(start_end_times) - 1):
            contrary_data = (start_end_times[i][1], start_end_times[i + 1][0])
            if contrary_data[0] != contrary_data[1]:
                contrary_start_end_times.append(contrary_data)

        for element in contrary_start_end_times:
            # adding a threshold because sometimes unneccessary rest events
            # are created and when we try to convert them to MIDI, we upsample
            # their duration values resulting in unneccessary rests between notes.
            if abs(element[0] - element[1]) > threshold:
                track_events.append(create_rest_event(element[0], element[1]))

        return track_events
