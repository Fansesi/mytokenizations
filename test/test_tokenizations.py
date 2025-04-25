import unittest
import numpy as np
from miditok import TokenizerConfig, TokSequence
import symusic as sy
import matplotlib.pyplot as plt
from pathlib import Path

from src import RelTok, ABCDTokenizer, ABCDNotation
from src.utils import create_dummy_score, out_txt, in_txt


class TestRelTok(unittest.TestCase):
    def setUp(self):
        self.conf = TokenizerConfig(
            pitch_range=[40, 85],
            num_velocities=16,
            use_tempos=True,
            num_tempos=16,
            # beat_res={(0, 4): 16, (4, 12): 8},
        )

        self.reltok = RelTok(self.conf)

    def test_full_conversion_dummy(self):
        score = create_dummy_score()
        score.dump_midi("output/test/reltok_dummy_test.mid")
        tokseq = self.reltok(score)
        self.reltok(tokseq).dump_midi(
            "output/test/reltok_dummy_test_recreated.mid"
        )

    def test_full_conversion_real(self):
        score = sy.Score("data/ussr27.mid")
        score.dump_midi("output/test/ussr27.mid")
        tokseq = self.reltok(score)
        self.reltok(tokseq).dump_midi(
            "output/test/reltok_ussr27_recreated.mid"
        )

    def _test_octuple(self):
        """Testign Octuple to see if the duration problem (ticks being so small)
        occurs there too.
        """
        from miditok import Octuple

        score = sy.Score("data/ussr27.mid")

        if not Path("output/test/ussr27.mid").exists():
            score.dump_midi("output/test/ussr27.mid")

        octuple = Octuple(self.conf)
        breakpoint()
        tokseq = octuple(score)

        octuple(tokseq).dump_midi(
            "output/test/octuple_ussr27_recreated.mid"
        )


class TestABCDTokenizer(unittest.TestCase):
    def setUp(self):
        self.conf = TokenizerConfig()
        self.abcd = ABCDTokenizer(self.conf)

    def test_tokens_to_score_simple_example(self):
        tokseq = TokSequence(
            tokens=[
                ["Pitch_45", "SameOrNot_0", "Duration_0.8.16"],
                ["Pitch_46", "SameOrNot_0", "Duration_0.8.16"],
            ]
        )
        score = self.abcd._tokens_to_score(tokseq)
        # print(score)
        score.dump_midi("output/test/abcd_simple_example.mid")
        # pianoroll = score.tracks[0].pianoroll()
        # plt.imshow(pianoroll[0] + pianoroll[1], aspect="auto")

    def test_full_conversion_real(self):
        score = sy.Score("data/ussr27.mid")

        if not Path("output/test/ussr27.mid").exists():
            score.dump_midi("output/test/ussr27.mid")

        tokseq = self.abcd.score_to_abcd(score)

        self.abcd.abcd_to_score(tokseq).dump_midi(
            "output/test/abcd_ussr27_recreated.mid"
        )

    def test_to_abcd(self):
        score = sy.Score("data/ussr27.mid")
        abcd_repr = self.abcd.score_to_abcd(score)
        out_txt(abcd_repr.bytes, "output/test/abcd_ussr27.txt")

    def test_from_abcd(self):
        text = (
            "abaadbabddbaabcbbcdcbdddcbcbdacbdacaaddbaacdbbcbcabcbcdbcdadbbc"
            + "cdababcaddaacbbbbdbbcbccbaacddcababadddaaadcaaaadddadaacccddadbdccbdacc"
            + "ccabacaccbaccccdccadabdbbdbdbbdcbbbabdaadccdccacccabcbabcdccdaacdaaadcd"
            + "abdbadaaaacdabbacdacaccaabddcababcadacccacddccbcddcbacacbbdcdcacdcabcaa"
            + "baccacbaacadaabaaccabcccddacdcbcabcdcbbacabacbbcddcddddbacbbbbcdacaacac"
            + "cccacbbbccbdacadadcacbacaaccaadaaddbccacddabacccacaacccabdabbbcacddcdbc"
            + "acbdaadcdaccaaccddaadaabaacbdbcbbbdaaadccacaccadadcbbaadaacbdcacbbaaaba"
            + "aaadaabccccbbaddcbdbcaadddabbbdadbbbaababbaaabbabdabacccdccdbdbcdacbcbd"
            + "babacbdbaadddcacaaabdbadcccbdbdcacaabcddacbcbdaabcdcbcdccdbddcacdcdccba"
            + "adbbdadbddcdccaadacaaccbddaaacccdababcdcdacaabaaaaabbaacbbadbabdbdbadda"
            + "bbbddaadbcbcddaadccababaacacabbdaaabbccabbacdcaaaacccbabccbababacdddaab"
            + "addaaacccaccbbbdbbcdbaccbddccbcacdcabdacbacadbcdbcaabbaaccacbaabcbdbcbb"
            + "bdcaababaaacddbbabaaaccdcbacbdbbabacbbcddacbbdacccaadbbdaddaaddccacbdcd"
            + "aabddacabccadbdbaacccddcabcdcdabbdcddbadbababcddaddabaaddbadcdbbabccbbd"
            + "cbdaadbbcccaba"
        )

        score = self.abcd.abcd_to_score(text)
        print(f"[INFO] Error rate is {self.abcd.tokens_errors(text)}")
        score.dump_midi("output/test/ALEgeneration_from_abcd.mid")

    def test_convert_generation(self):
        """Test for converting a BAD generation from."""

        text = "cdccddaddaddcbbbcaaaababcdbcaaaacaaddbabdcadbdcbdadcbcabddaccbacbbbadddbddcabcdaabcdbacccaccdccabbba"

        score = self.abcd.abcd_to_score(text)
        print(f"[INFO] Error rate is {self.abcd.tokens_errors(text)}")
        # score.dump_midi("output/test/convert_generation.mid")


class TestABCDNotation(unittest.TestCase):
    def setUp(self):
        self.notation = ABCDNotation(None)

    def test_clean_abcd(self):
        """Test for ABCDNotation.clean_abcd() function."""
        text1 = "cabccabccccababcababccabacbacbabcaccbccabcccaccb"
        corrected_text1 = "cdaaccdaaccccdaaaacdaaaaccdaaacdbbcdbbbcdaccdbccdaacccdaccd"
        self.assertEqual(ABCDNotation.clean_abcd(text1), corrected_text1)

        text2 = "caaabccdabccdccdababcabbbabccdc"
        corrected_text2 = "cdaaaaccdaaccdccdaaaacdbbbbbbccdcd"
        self.assertEqual(ABCDNotation.clean_abcd(text2), corrected_text2)
