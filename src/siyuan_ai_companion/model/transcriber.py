"""
A functional class to transcribe audio files
"""

import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from siyuan_ai_companion.consts import TRANSCRIBE_KEEP_MODEL, HUGGINGFACE_HUB_TOKEN
from .siyuan_api import SiyuanApi


class Transcriber:
    """
    A functional class to transcribe audio files
    """
    _whisper_model: WhisperModel | None = None
    _pipeline = None

    @property
    def whisper_model(self) -> WhisperModel:
        """
        Get the whisper model.
        :return: WhisperModel object.
        """
        if self._whisper_model is None:
            wm = WhisperModel(
                'medium',
                compute_type='int8',
            )

            if TRANSCRIBE_KEEP_MODEL:
                Transcriber._whisper_model = wm
            else:
                return wm

        return Transcriber._whisper_model

    @property
    def pipeline(self) -> Pipeline:
        """
        Get the diarisation pipeline.
        :return: A Pipeline object.
        """
        if self._pipeline is None:
            pipeline = Pipeline.from_pretrained(
                'pyannote/speaker-diarization',
                use_auth_token=HUGGINGFACE_HUB_TOKEN,
            )

            if TRANSCRIBE_KEEP_MODEL:
                Transcriber._pipeline = pipeline
            else:
                return pipeline

        return Transcriber._pipeline

    async def _transcribe_and_diarise_file(self, audio_path: str) -> list[dict]:
        """
        Transcribe and diarise an audio file.
        :param audio_path: The absolute path to the audio file.
        :return: A list of dictionaries containing the start time, end time,
                 speaker label, and text.
        """
        # Transcribe
        model = self.whisper_model
        segments_iter, _ = model.transcribe(
            audio_path,
            language='en',
        )
        segments = list(segments_iter)

        # Diarise
        pipeline = self.pipeline
        diarisation = pipeline(audio_path)

        # Merge efficiently
        output = []
        seg_idx = 0
        num_segments = len(segments)

        for turn, _, speaker in diarisation.itertracks(yield_label=True):
            # Advance to the first segment that might overlap
            while seg_idx < num_segments and segments[seg_idx].end <= turn.start:
                seg_idx += 1

            i = seg_idx
            while i < num_segments and segments[i].start < turn.end:
                seg = segments[i]
                if seg.end > turn.start:
                    output.append({
                        'start': seg.start,
                        'end': seg.end,
                        'speaker': speaker,
                        'text': seg.text,
                    })
                i += 1

        return output

    async def process_asset(self,
                            asset_path: str,
                            ):
        async with SiyuanApi() as siyuan:
            async with siyuan.download_asset(asset_path) as audio_file:
                audio_path = os.path.abspath(audio_file.name)

                result = self._transcribe_and_diarise_file(
                    audio_path=audio_path,
                )


