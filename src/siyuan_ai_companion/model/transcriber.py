"""
A functional class to transcribe audio files
"""

import os
import datetime
import asyncio
from concurrent.futures import ProcessPoolExecutor
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from pyannote.audio import Pipeline

from siyuan_ai_companion.consts import HUGGINGFACE_HUB_TOKEN, WHISPER_WORKERS, \
    SIYUAN_TRANSCRIBE_NOTEBOOK
from .siyuan_api import SiyuanApi


def _transcribe(audio_path) -> list[Segment]:
    """
    Transcribe an audio file using the Whisper model.
    :param audio_path: The absolute path to the audio file.
    :return: A list of Segment objects, containing the start time,
             end time, and text.
    """
    transcriber = Transcriber()
    model = transcriber.whisper_model

    segments_iter, _ = model.transcribe(
        audio_path,
        language='en',
    )

    # Segments are lazy-processed, so load everything into a list
    segments = list(segments_iter)

    return segments


def _diarise(audio_path):
    """
    Diarise an audio file using the pyannote speaker diarization pipeline.
    :param audio_path: The absolute path to the audio file.
    :return: A PipelineOutput object containing the result
    """
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

    @staticmethod
    async def _transcribe_and_diarise_file(audio_path: str) -> list[dict]:
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

    @staticmethod
    def _cleanup_segments(segments: list[dict]) -> list[dict]:
        """
        Clean up the segments by removing duplicates and ensuring
        that no speakers are assigned to the same segment.
        :param segments: A list of dictionaries containing the start time, end time,
                         speaker label, and text.
        :return: Same format as input, but with duplicates removed and speakers
                 cleaned up.
        """
        cleaned_segments = []
        last_speaker = None

        # Clean up duplicated speaker assignments
        for seg in segments:
            # Find all segments with same start, end, and text
            duplicates = [
                r for r in segments
                if r['start'] == seg['start']
                   and r['end'] == seg['end']
                   and r['text'] == seg['text']
            ]

            if len(duplicates) > 1:
                # More than one speaker assigned to utterance, choose the one
                # immediately before the current one
                prev_speaker = last_speaker
                result_to_use = None

                if prev_speaker and any(d['speaker'] == prev_speaker for d in duplicates):
                    result_to_use = next(d for d in duplicates if d['speaker'] == prev_speaker)
                else:
                    # No immediate previous speaker, use the first one
                    result_to_use = duplicates[0]

                if not any(d == result_to_use for d in cleaned_segments):
                    cleaned_segments.append(result_to_use)
                    last_speaker = result_to_use['speaker']
            else:
                # Only one speaker assigned to utterance
                cleaned_segments.append(seg)
                last_speaker = seg['speaker']

        return cleaned_segments

    @staticmethod
    def _merge_segments(segments: list[dict]) -> list[str]:
        """
        Merge segments with the same speaker into a single string.
        :param segments: A list of dictionaries containing the start time, end time,
                         speaker label, and text.
        :return: A list of strings, each containing the speaker label and the
                 corresponding text.
        """
        merged_segments = []
        buffer = {'speaker': None, 'text': ''}

        for seg in segments:
            speaker_label = f'**{seg["speaker"].replace("_", " ")}**'

            if buffer['speaker'] == speaker_label:
                buffer['text'] += ' ' + seg['text'].strip()
            else:
                if buffer['speaker']:
                    merged_segments.append(f'{buffer["speaker"]}: {buffer["text"].strip()}')
                buffer = {'speaker': speaker_label, 'text': seg['text'].strip()}

        if buffer['speaker']:
            merged_segments.append(f'{buffer["speaker"]}: {buffer["text"].strip()}')

        return merged_segments

    async def process_asset(self,
                            asset_path: str,
                            title: str = '',
                            t_notebook: str = None,
                            t_base_path: str = None,
                            ):
        """
        Process a given asset from SiYuan note server
        :param asset_path: The path to asset file, relative from `/data/assets/`
        :param title: The title of the asset, will be used as the transcription title
        :param t_notebook: The notebook to save the transcription to. None for environment
                           variable controlled value
        :param t_base_path: The base path to save the transcription to. None for root of
                            the notebook
        """
        async with SiyuanApi() as siyuan:
            async with siyuan.download_asset(asset_path) as audio_file:
                audio_path = os.path.abspath(audio_file.name)

                results = await self._transcribe_and_diarise_file(
                    audio_path=audio_path,
                )

            results.sort(key=lambda x: x['start'])
            cleaned_result = self._cleanup_segments(results)
            merged_result = self._merge_segments(cleaned_result)
            formatted_output = '\n\n'.join(merged_result)

            notebook_id = t_notebook or SIYUAN_TRANSCRIBE_NOTEBOOK
            if notebook_id is None:
                raise RuntimeError(
                    'Notebook ID for transcription is not set.'
                )

            base_path = t_base_path or '/'
            if not base_path.endswith('/'):
                base_path += '/'

            title = title or ('Transcription ' +
                              datetime.datetime.now().strftime('%Y%M%D%H%M%S'))

            await siyuan.create_note(
                notebook_id=notebook_id,
                path=f'{base_path}{title}',
                markdown_content=formatted_output,
            )
