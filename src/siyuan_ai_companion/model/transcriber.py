"""
A functional class to transcribe audio files
"""

import os
import asyncio
from concurrent.futures import ProcessPoolExecutor
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from pyannote.audio import Pipeline

from siyuan_ai_companion.consts import HUGGINGFACE_HUB_TOKEN, WHISPER_WORKERS
from .siyuan_api import SiyuanApi


def _transcribe(audio_path) -> list[Segment]:
    transcriber = Transcriber()
    model = transcriber.whisper_model

    segments_iter, _ = model.transcribe(
        audio_path,
        language='en',
    )

    segments = list(segments_iter)

    return segments


def _diarise(audio_path):
    transcriber = Transcriber()
    pipeline = transcriber.pipeline

    diarisation = pipeline(audio_path)

    return diarisation


class Transcriber:
    """
    A functional class to transcribe audio files
    """

    @property
    def whisper_model(self) -> WhisperModel:
        """
        Get the whisper model.
        :return: WhisperModel object.
        """
        wm = WhisperModel(
            'medium',
            compute_type='int8_float32',
            num_workers=WHISPER_WORKERS,
        )
        return wm

    @property
    def pipeline(self) -> Pipeline:
        """
        Get the diarisation pipeline.
        :return: A Pipeline object.
        """
        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization',
            use_auth_token=HUGGINGFACE_HUB_TOKEN,
        )
        return pipeline

    async def _transcribe_and_diarise_file(self, audio_path: str) -> list[dict]:
        """
        Transcribe and diarise an audio file.
        :param audio_path: The absolute path to the audio file.
        :return: A list of dictionaries containing the start time, end time,
                 speaker label, and text.
        """
        loop = asyncio.get_event_loop()
        executor = ProcessPoolExecutor(max_workers=2)

        tasks = [
            loop.run_in_executor(executor, _transcribe, audio_path),
            loop.run_in_executor(executor, _diarise, audio_path),
        ]

        segments, diarisation = await asyncio.gather(*tasks)

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
