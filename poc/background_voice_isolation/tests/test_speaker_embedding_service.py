"""Basic tests for speaker embedding helper behavior."""

from __future__ import annotations

import unittest

from src.audio_io import load_audio
from src.speaker_embedding_service import (
    SpeakerEmbeddingService,
    cosine_similarity_score,
    same_speaker_decision,
)


class SpeakerEmbeddingServiceTests(unittest.TestCase):
    def test_import_service(self) -> None:
        service = SpeakerEmbeddingService()
        self.assertEqual(service.model_source, "speechbrain/spkrec-ecapa-voxceleb")

    def test_cosine_similarity_same_vector_is_near_one(self) -> None:
        score = cosine_similarity_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        self.assertAlmostEqual(score, 1.0, places=6)

    def test_cosine_similarity_different_vector_is_lower(self) -> None:
        score = cosine_similarity_score([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

        self.assertLess(score, 0.1)

    def test_missing_file_handling(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_audio("sample_audio/missing_reference.wav")

    def test_threshold_decision_logic(self) -> None:
        self.assertTrue(same_speaker_decision(0.8, threshold=0.75))
        self.assertFalse(same_speaker_decision(0.7, threshold=0.75))


if __name__ == "__main__":
    unittest.main()
