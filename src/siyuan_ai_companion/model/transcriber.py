"""
A functional class to transcribe audio files
"""

import os
import datetime
import asyncio
from typing import BinaryIO
from concurrent.futures import ProcessPoolExecutor
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from pyannote.audio import Pipeline

from siyuan_ai_companion.consts import APP_CONFIG, LOGGER
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

    LOGGER.info('Transcribing %s', audio_path)

    segments_iter, _ = model.transcribe(
        audio_path,
        language='en',
    )

    # Segments are lazy-processed, so load everything into a list
    segments = list(segments_iter)

    LOGGER.info('Transcription finished for %s', audio_path)

    return segments


def _diarise(audio_path):
    """
    Diarise an audio file using the pyannote speaker diarization pipeline.

    :param audio_path: The absolute path to the audio file.
    :return: A PipelineOutput object containing the result
    """
    transcriber = Transcriber()
    pipeline = transcriber.pipeline

    LOGGER.info('Diarising %s', audio_path)

    diarisation = pipeline(audio_path)

    LOGGER.info('Diarisation finished for %s', audio_path)

    return diarisation


class Transcriber:
    """
    A functional class to transcribe audio files
    """

    @property
    def whisper_model(self) -> WhisperModel:
        """
        Get the whisper model.

        This getter always reloads the model to remain compatible
        with multiprocessing functions.
        :return: WhisperModel object.
        """
        wm = WhisperModel(
            'medium',
            compute_type='int8_float32',
            num_workers=APP_CONFIG.whisper_workers,
        )
        return wm

    @property
    def pipeline(self) -> Pipeline:
        """
        Get the diarisation pipeline.

        This getter always reloads the model to remain compatible
        with multiprocessing functions.
        :return: A Pipeline object.
        """
        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization',
            use_auth_token=APP_CONFIG.huggingface_hub_token,
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
            if await siyuan.is_processing(asset_path):
                LOGGER.warning('Asset %s is already being processed', asset_path)
                return

            async with siyuan.download_asset(asset_path) as audio_file:
                await siyuan.add_to_processing(asset_path)

                audio_path = os.path.abspath(audio_file.name)

                results = await self._transcribe_and_diarise_file(
                    audio_path=audio_path,
                )

            results.sort(key=lambda x: x['start'])
            formatted_output = '\n\n'.join(
                self._merge_segments(
                    self._cleanup_segments(results)
                )
            )

            notebook_id = t_notebook or APP_CONFIG.siyuan_transcribe_notebook
            audio_block_id = await siyuan.get_audio_block(
                audio_name=asset_path.split('/')[-1],
            )
            if notebook_id is None:
                # No notebook specified, insert in the original note
                formatted_output = '**Transcription**\n\n' + formatted_output

                new_block_id = await siyuan.insert_block(
                    markdown_content=formatted_output,
                    previous_id=audio_block_id,
                )

                await siyuan.set_block_attribute(
                    block_id=new_block_id,
                    attributes={
                        'alias': f'transcription-{audio_block_id}',
                    },
                )

                return

            base_path = t_base_path or '/'
            if not base_path.endswith('/'):
                base_path += '/'

            title = title or ('Transcription ' +
                              datetime.datetime.now().strftime('%Y%M%D%H%M%S'))

            note_id = await siyuan.create_note(
                notebook_id=notebook_id,
                path=f'{base_path}{title}',
                markdown_content=formatted_output,
            )

            await siyuan.set_block_attribute(
                block_id=note_id,
                attributes={
                    'alias': f'transcription-{audio_block_id}',
                },
            )

            await siyuan.remove_from_processing(asset_path)

    async def process_buffer(self,
                             audio_buffer: BinaryIO,
                             ):
        """
        Process an audio buffer and generate the response
        :param audio_buffer: A binary buffer containing the audio data.
            Could be a file content or any compatible buffer
        :return: A generator that yields the transcribed text.
        """
        segments, _ = self.whisper_model.transcribe(
            audio=audio_buffer,
            language='en',
        )

        for segment in segments:
            yield segment.text
            await asyncio.sleep(0)
