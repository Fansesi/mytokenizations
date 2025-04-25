def tokenize_dataset(mid_path: Path, out_dir: Path):
    """To tokenize the melody_only dataset."""
    from miditok import TokenizerConfig
    from src import RelTok

    my_tok_config = TokenizerConfig(
        pitch_range=[40, 85],
        nb_velocities=16,
        use_tempos=False,
        nb_tempos=16,
        beat_res={(0, 4): 16, (4, 12): 8},
    )

    my_tokenizer = RelTok(my_tok_config)
    my_tokenizer.tokenize_midi_dataset(midi_paths=mid_path, out_dir=out_dir)


def detokenize_dataset(dataset_path: Path, out_dir: Path):
    """Detokenize the already tokenized dataset in order to obtain perfectly quantized midis."""
    # from constants import DATASET_PATH
    from miditok import TokenizerConfig, MIDITokenizer, TokSequence
    from src import RelTok

    my_tok_config = TokenizerConfig(
        pitch_range=[40, 85],
        nb_velocities=16,
        use_tempos=False,
        nb_tempos=16,
        beat_res={(0, 4): 16, (4, 12): 8},
    )

    my_tokenizer: MIDITokenizer = RelTok(my_tok_config)

    tokenized_midi_paths = dataset_path.glob("*.json")

    # see MIDITokenizer.__call__()
    # save() function is from MidiFile
    for path in tqdm(tokenized_midi_paths):
        toked = TokSequence(ids=my_tokenizer.load_tokens(path)["ids"])
        my_tokenizer.tokens_to_midi(toked).dump(
            filename=str(out_dir)
            + "detoked_"
            + str(path.stem)
            + ".mid"
        )
