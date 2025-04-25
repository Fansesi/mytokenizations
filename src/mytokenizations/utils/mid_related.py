from typing import Union, List, Dict, Optional, Any
import pretty_midi
import symusic as sy
from pathlib import Path


def create_pretty_midi(
    pitches: List[int] | int,
    durations: List[float] | float,
    tempo=120,
    time_signature=(4, 4),
):
    """Create a PrettyMIDI object with the given notes and durations, and save it to a file.

    Args:
        notes (List[int] | int): MIDI note numbers (e.g., 60 for C4, 62 for D4, etc.).
        durations (List[float] | float): Duration of each note in ticks.
        tempo (int, optional): Tempo in beats per minute. Defaults to 120.
        time_signature (tuple, optional): Time signature as (numerator, denominator).
        Defaults to (4, 4).

    Returns:
        pretty_midi.PrettyMIDI: The created PrettyMIDI object
    """
    if isinstance(pitches, list) and isinstance(durations, list):
        assert len(pitches) == len(
            durations
        ), "Lengths of the notes and duration values do not match."

    if isinstance(pitches, int):
        pitches = [pitches] * len(durations)

    if isinstance(durations, (int, float)):
        durations = [durations] * len(pitches)

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Create an Instrument object for the notes (e.g., a piano)
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    instrument = pretty_midi.Instrument(program=piano_program)

    # Add notes to the instrument
    time = 0.0
    for pitch, duration in zip(pitches, durations):
        secs = midi.tick_to_time(duration)
        note = pretty_midi.Note(
            velocity=100,  # Velocity of the note (e.g., volume)
            pitch=pitch,  # MIDI note number
            start=time,  # Start time of the note
            end=time + secs,  # End time of the note
        )
        instrument.notes.append(note)
        time += secs  # Increment time by the duration of each note

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)

    # Set the time signature (numerator and denominator)
    midi.time_signature_changes.append(
        pretty_midi.TimeSignature(
            numerator=time_signature[0], denominator=time_signature[1], time=0
        )
    )

    # Write the MIDI to a file
    # midi.write(str(output_path))

    return midi


def create_score(
    pitches: List[int] | int,
    durations: List[int] | int,
    tempo=120,
    time_signature=(4, 4),
):
    """Create a symusic.Score object with the given notes and durations.
    Args:
        notes (List[int] | int): MIDI note numbers (e.g., 60 for C4, 62 for D4, etc.).
        durations (List[float] | float): Duration of each note in *ticks*.
        tempo (int, optional): Tempo in beats per minute. Defaults to 120.
        time_signature (tuple, optional): Time signature as (numerator, denominator).
        Defaults to (4, 4).

    Returns:
        symusic.Score: The created Score object
    """
    if isinstance(pitches, list) and isinstance(durations, list):
        assert len(pitches) == len(
            durations
        ), "Lengths of the notes and duration values do not match."

    if isinstance(pitches, int):
        pitches = [pitches] * len(durations)

    if isinstance(durations, int):
        durations = [durations] * len(pitches)

    score = sy.Score()
    track = sy.Track(program=12)
    score.tempos.append(sy.Tempo(0, tempo))
    score.time_signatures.append(
        sy.TimeSignature(0, time_signature[0], time_signature[1])
    )

    current_time = 0
    for pitch, duration in zip(pitches, durations):
        note = sy.Note(
            velocity=100,
            pitch=pitch,
            time=current_time,
            duration=duration,
        )
        score.tracks[0].notes.append(note)
        current_time += duration

    score.tracks.append(track)

    return score


def group_by_bars(midi: str | Path | pretty_midi.PrettyMIDI) -> List[List[int]]:
    """Group notes that are in the same bar, considering time signature and tempo changes.

    Args:
        midi (Union[str, Path, pretty_midi.PrettyMIDI]): MIDI file path or a PrettyMIDI object.

    Returns:
        List[List[int]]: Grouped MIDI note values by bar.
    """
    # Load the MIDI file if it is a path or string
    if isinstance(midi, (str, Path)):
        try:
            midi = pretty_midi.PrettyMIDI(str(midi))
        except:
            lg.error(f"Error on file: {str(midi)}")
            return None
    # Sort time signature and tempo changes by time
    time_signatures = sorted(midi.time_signature_changes, key=lambda ts: ts.time)
    tempo_changes, tempi = midi.get_tempo_changes()

    # Helper function to find the active time signature at a given time
    def get_active_time_signature(current_time: float) -> pretty_midi.TimeSignature:
        for i in range(len(time_signatures)):
            if (
                i == len(time_signatures) - 1
                or current_time < time_signatures[i + 1].time
            ):
                return time_signatures[i]

    # Helper function to find the active tempo at a given time
    def get_active_tempo(current_time: float) -> float:
        for i in range(len(tempo_changes)):
            if i == len(tempo_changes) - 1 or current_time < tempo_changes[i + 1]:
                return tempi[i]

    # Group notes by bar
    grouped_notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks
        for note in instrument.notes:
            # Determine the time signature and tempo at the note's start time
            active_time_signature = get_active_time_signature(note.start)
            active_tempo = get_active_tempo(note.start)

            # Calculate the bar length in seconds
            beats_per_bar = active_time_signature.numerator
            beat_length = 60 / active_tempo  # Length of one beat in seconds
            bar_length = beats_per_bar * beat_length

            # Calculate the bar index for the note
            bar_index = int(note.start // bar_length)

            # Ensure grouped_notes is large enough to hold this bar index
            while len(grouped_notes) <= bar_index:
                grouped_notes.append([])  # Add an empty list for the new bar

            # Add the note pitch to the corresponding bar group
            grouped_notes[bar_index].append(note.pitch)

    return grouped_notes


def create_dummy_pm(tempo=120, key="C", scale_type="major"):
    """Create a simple dummy song with a basic melody.

    Args:
        tempo (int): The tempo of the song in beats per minute (BPM).
        key (str): The root note of the scale (e.g., 'C', 'D', 'E').
        scale_type (str): The type of scale ('major' or 'minor').

    Returns:
        PrettyMIDI
    """
    # Create a PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Define the scale
    if scale_type.lower() == "major":
        intervals = [0, 2, 4, 5, 7, 9, 11]
    elif scale_type.lower() == "minor":
        intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        raise ValueError("Scale type must be either 'major' or 'minor'.")

    # Get the MIDI note number for the root key
    root_note = pretty_midi.note_name_to_number(f"{key}4")

    # Define the scale notes
    scale_notes = [root_note + interval for interval in intervals]

    # Add an instrument (e.g., Piano)
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)
    ts = pretty_midi.TimeSignature(4, 4, 0)
    midi_data.time_signature_changes = [ts]

    # Define a simple melody pattern
    melody_pattern = [0, 2, 4, 5, 7, 9, 11, 12]  # Half steps

    # Add notes to the instrument
    duration = 0.5  # Duration of each note in seconds
    current_time = 0

    for i in range(2):  # Create an 2-bar melody
        for step in melody_pattern:
            pitch = scale_notes[step % len(scale_notes)]
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=current_time,
                end=current_time + duration,
            )
            piano.notes.append(note)
            current_time += duration

    # Add the instrument to the MIDI data
    midi_data.instruments.append(piano)

    return midi_data


def create_dummy_score(tempo=120, scale_type="major"):
    """Create a simple dummy song with a basic melody.

    Args:
        tempo (int): The tempo of the song in beats per minute (BPM).
        key (str): The root note of the scale (e.g., 'C', 'D', 'E').
        scale_type (str): The type of scale ('major' or 'minor').

    Returns:
        symusic.Score
    """
    score = sy.Score()
    score.tempos.append(sy.Tempo(0, 120))
    score.tracks.append(sy.Track(program=12))
    score.time_signatures.append(sy.TimeSignature(0, 4, 4))

    # Define the scale
    if scale_type.lower() == "major":
        intervals = [0, 2, 4, 5, 7, 9, 11]
    elif scale_type.lower() == "minor":
        intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        raise ValueError("Scale type must be either 'major' or 'minor'.")

    root_note = 48

    # Define the scale notes
    scale_notes = [root_note + interval for interval in intervals]

    # Define a simple melody pattern
    melody_pattern = [0, 2, 4, 5, 7, 9, 11, 12]  # Half steps

    # Add notes to the instrument
    duration = 480
    current_time = 0

    for i in range(2):  # Create an 2-bar melody
        for step in melody_pattern:
            pitch = scale_notes[step % len(scale_notes)]
            note = sy.Note(
                velocity=100,
                pitch=pitch,
                time=current_time,
                duration=duration,
            )
            score.tracks[0].notes.append(note)
            current_time += duration

    return score
