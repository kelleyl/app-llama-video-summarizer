import argparse
import logging
import re
from typing import Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh, text_document_helper
from lapps.discriminators import Uri

class LlamaVideoSummarizer(ClamsApp):

    def __init__(self):
        super().__init__()
        self.model_name = "lmsys/vicuna-7b-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        self.logger.debug("Running app")
        # stride = parameters.get('stride', 2)  # Default stride is 60 seconds
        stride = 20
        self.video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)

        video_duration = 60 * 20 # to do implement this
        start_time = 0

        while start_time < video_duration:
            print (f"starttime: {start_time}\n")
            end_time = min(start_time + stride, video_duration)
            print (f"endtime: {end_time}\n")
            context = self.get_context(mmif, start_time, end_time)
            summary = self.generate_summary(context)

            text_document = new_view.new_textdocument(summary)
            time_frame = new_view.new_annotation(AnnotationTypes.TimeFrame)
            time_frame.add_property("start", start_time)
            time_frame.add_property("end", end_time)
            alignment = new_view.new_annotation(AnnotationTypes.Alignment)
            alignment.add_property("source", time_frame.long_id)
            alignment.add_property("target", text_document.long_id)

            start_time = end_time

        return mmif

    def get_context(self, mmif: Mmif, start_time: float, end_time: float) -> str:
        caption_text = ""
        # get timeframe from mmif for app app-llava-captioner
        for _view in mmif.get_views_contain(AnnotationTypes.TimeFrame):
            if _view.metadata.app == "http://apps.clams.ai/transnet-wrapper/unresolvable":
                # convert start and end seconds to frame number
                start_frame = vdh.second_to_framenum(self.video_doc, start_time)
                end_frame = vdh.second_to_framenum(self.video_doc, end_time)
                # get all annotations between start and end time
                for _timeframe in _view.get_annotations(AnnotationTypes.TimeFrame):
                    if _timeframe.properties['start'] >= start_frame and _timeframe.properties['end'] <= end_frame:
                        print (f"shot timeframe: {_timeframe}")
                        for aligned in _timeframe.get_all_aligned():
                            if aligned.at_type == DocumentTypes.TextDocument:
                                #strip caption text of anything between [INST] and [/INST] remove all text between [INST] and [/INST]
                                # caption_text += re.sub(r"\[INST\].*?\[/INST\]", "", aligned.text_value, flags=re.DOTALL) + "\n"
                                caption_text += aligned.text_value + "\n"
                transcript_text = text_document_helper.slice_text(
                    mmif, 
                    start=start_frame,
                    end=end_frame,
                    unit="frame"
                )
                return f"Frame Captions: {caption_text.strip()}\nTranscript: {transcript_text.strip()}"

    def generate_summary(self, context: str) -> str:
        prompt = f"You will be provided with image captions and speech data from a broadcast news video. Summarize the following video segment:\n\n{context}\n\nSummary:"
        print (f"prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print (f"summary: {summary}")
        return summary.split("Summary:")[-1].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parser.add_argument("--stride", type=int, default=60, help="Time stride in seconds for summarization")

    parsed_args = parser.parse_args()

    app = LlamaVideoSummarizer()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()