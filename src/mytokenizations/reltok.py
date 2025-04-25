from __future__ import annotations

from typing import TYPE_CHECKING

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



class RelTok(MusicTokenizer):
    """Doesn't use duration and bar tokens instead of them use same_as_before_or_not tokens."""

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        params: str | Path = None,
    ):
        super().__init__(tokenizer_config, params)

        # Adding Velocity_0 to velocity vocab.
        # This will be used in rest tokens (which are pitch type) velocity data.
        self.add_to_vocab(
            "Velocity_0", vocab_idx=1
        )  # 1 is the index of velocity in the dict.

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
        # self.config.one_token_stream = True

        # time signature is necessary to parse the notes into bars
        self.config.use_time_signatures = True
        self.config.use_tempos = True

        # This implementation is for single track instruments
        self.config.use_programs = False
        # self.config.program_changes = False

        token_types = ["Pitch", "Velocity", "Duration", "SameOrNot", "Tempo", "TimeSig"]
        self.vocab_types_idx = {
            type_: idx for idx, type_ in enumerate(token_types)
        }  # used to find the event location and augmentation.

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
        # midi = MidiFile(ticks_per_beat=time_division)
        score = Score(self.time_division)

        # RESULTS
        tracks: Dict[int, Track] = {}
        tempo_changes, time_signature_changes = [Tempo(-1, self.default_tempo)], []
        tempo_changes[0].tempo = -1

        def is_track_empty(track: Track) -> bool:
            return (
                len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0
            )

        _first_time_sig = tokens[0][0][self.vocab_types_idx["TimeSig"]].split("_")[1]
        if _first_time_sig == "None":
            # if there is no time_sig then it's set to 4/4 as default
            time_signature_changes = [TimeSignature(0, 4, 4)]
        else:
            num, den = self._parse_token_time_signature(_first_time_sig)
            time_signature_changes = [TimeSignature(num, den, 0)]

        current_program = 24
        current_tick = 0

        dynamic_duration_list = []
        for si, seq in enumerate(tokens):
            # Decode tokens
            for k, time_step in enumerate(seq):
                # lg.debug(f"current_tick: {current_tick}")

                nb_tok_to_check = 6
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
                vel = int(time_step[1].split("_")[1])
                duration = self._time_token_to_ticks(
                    time_step[2].split("_")[1], self.time_division
                )
                if k == 0:
                    dynamic_duration_list = [duration]  # initializing.
                    current_tick = duration

                # if 0, cursor should be moved.
                # if 1, cursor should be at the same place.
                same_or_not = int(time_step[3].split("_")[1])

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

                # Append the created note
                if current_program not in tracks.keys():
                    tracks[current_program] = Track(
                        program=current_program,
                        is_drum=False,
                        name=MIDI_INSTRUMENTS[current_program]["name"],
                    )

                tracks[current_program].notes.append(
                    Note(current_tick, duration, pitch, vel)
                )

                # Tempo, adds a Tempo if necessary
                tempo = time_step[self.vocab_types_idx["Tempo"]].split("_")[1]

                if si == 0 and self.config.use_tempos and tempo != "None":
                    tempo = float(tempo)
                    if tempo != tempo_changes[-1].tempo:
                        tempo_changes.append(Tempo(current_tick, tempo))

                time_sig = time_step[self.vocab_types_idx["TimeSig"]].split("_")[1]

                # Time Signature, adds a TimeSignatureChange if necessary
                if self.config.use_time_signatures and time_sig != "None":
                    num, den = self._parse_token_time_signature(time_sig)
                    if (
                        num != time_signature_changes[-1].numerator
                        or den != time_signature_changes[-1].denominator
                    ):  # checking whether the new ts is the same as the last.
                        # tick from bar of ts change
                        final_time_sig = TimeSignature(current_tick, num, den)
                        if si == 0:
                            time_signature_changes.append(final_time_sig)

                        # ticks_per_bar = self._compute_ticks_per_bar(
                        #    time_sig, time_division
                        # )

        del tempo_changes[0]
        if len(tempo_changes) == 0 or (
            tempo_changes[0].time != 0
            and round(tempo_changes[0].tempo, 2) != self.default_tempo
        ):
            tempo_changes.insert(0, Tempo(0, self.default_tempo))
        elif round(tempo_changes[0].tempo, 2) == self.default_tempo:
            tempo_changes[0].time = 0

        if self.config.one_token_stream_for_programs:
            score.tracks = list(tracks.values())
        score.tempos = tempo_changes
        score.time_signatures = time_signature_changes

        # NOTE: No idea why I have written this in the previous versions.
        # time_signature_changes[0].time = 0

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
        all_events = []

        # Global events (Tempo, TimeSignature)
        all_events += self._create_global_events(score)
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

        # track_events are ordered as follows: Pitch, Velocity, Duration.
        # Based on these I'm going to add Rest Events in between notes.
        track_events = self._create_inbetween_rests(track_events)
        all_events += track_events

        all_events.sort(key=lambda x: (x.time, self._order(x)))

        # Add time events. Below, we are not using += because this function
        # inserts (not inplace) time events
        all_events = self._add_time_events(all_events, self.time_division)

        tok_sequence = TokSequence(events=all_events)
        self.complete_sequence(tok_sequence)  # creates the missing representations

        return tok_sequence
        # return super()._score_to_tokens(score)

    def _add_time_events(
        self, events: List[Event], time_division: int
    ) -> List[List[Event]]:
        """Below is taken and revised from Octuple.

        Takes a sequence of note events (containing optionally Chord, Tempo and TimeSignature tokens),
        and insert (not inplace) time tokens (TimeShift, Rest) to complete the sequence.
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: SameOrNot
            (4: Tempo)
            (5: TimeSignature)

        Not: Although 4 and 5 seems to be optional, time signature is necessary to parse the notes
        into bars. So it has been set to True in _tweak_config_before_creating_voc method.

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

            if e == 0:
                same_or_not_value = 0

            if event.type_ == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))

            elif event.type_ == "Tempo":
                current_tempo = event.value

            elif event.type_ == "Rest" and e + 2 < len(events):
                # Catching the placeholder rest Events.

                # TODO: experiment with rest=True and rest=False param.
                if event.time - event.desc == 0:
                    raise Exception("This notes duration value is 0!")

                # dur_value, dur_ticks = self._ticks_to_duration_tokens(
                #     abs(event.time - event.desc), rest=False
                # )

                # WARNING: This might cause trouble
                # because of the static time_division.
                dur_value, dur_ticks = self._time_ticks_to_tokens(
                    abs(event.time - event.desc),
                    time_division,
                    rest=False,
                )

                new_event = [
                    Event(type_="Pitch", value="REST", time=event.time),
                    Event(type_="Velocity", value=0, time=event.time),
                    Event(
                        type_="Duration",
                        value=".".join(map(str, dur_value[0])),
                        time=event.time,
                    ),
                    Event(type_="SameOrNot", value=0, time=event.time),
                ]

                if self.config.use_tempos:
                    new_event.append(Event(type_="Tempo", value=current_tempo))
                if self.config.use_time_signatures:
                    new_event.append(
                        Event(
                            type_="TimeSig",
                            value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                        )
                    )
                all_events.append(new_event)

            elif event.type_ == "Pitch" and e + 2 < len(events):
                # lg.debug(f"Velocity for this note is: {events[e + 1].value}")
                new_event = [
                    Event(type_="Pitch", value=event.value, time=event.time),
                    Event(type_="Velocity", value=events[e + 1].value, time=event.time),
                    Event(type_="Duration", value=events[e + 2].value, time=event.time),
                    Event(type_="SameOrNot", value=same_or_not_value, time=event.time),
                    # We don't use Position and Bar values!
                    # Event(type="Position", value=current_pos, time=event.time),
                    # Event(type="Bar", value=current_bar, time=event.time),
                ]
                if self.config.use_tempos:
                    new_event.append(Event(type_="Tempo", value=current_tempo))
                if self.config.use_time_signatures:
                    new_event.append(
                        Event(
                            type_="TimeSig",
                            value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                        )
                    )
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
        vocab = [[] for _ in range(4)]

        # PITCH
        # Rests are a special kind of pitch value. It can be generated automatically,
        # whenever there is a difference between a NoteOff and a NoteOn there should be a rest there.
        # Music is cumulative!
        vocab[0] += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]
        vocab[0] += [f"Pitch_REST"]

        # VELOCITY
        vocab[1] += [f"Velocity_{i}" for i in self.velocities]

        # DURATION
        vocab[2] += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # SAME_AS_BEFORE_OR_NOT
        vocab[3] += [f"SameOrNot_{i}" for i in range(2)]  # basic yes or no

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
        if self.config.use_programs:
            vocab.append([f"Program_{i}" for i in self.config.programs])

        # TEMPO
        if self.config.use_tempos:
            vocab.append([f"Tempo_{i}" for i in self.tempos])

        # TIME_SIGNATURE
        if self.config.use_time_signatures:
            vocab.append([f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures])

        return vocab

    def tokens_errors(
        self, tokens: TokSequence | List[int | List[int]]
    ) -> float | List[float]:
        """Checks if a sequence of tokens is made of good token types
        successions and returns the error ratio (lower is better).
        """
        raise NotImplementedError("Not needed.")

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
