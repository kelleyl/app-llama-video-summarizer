import argparse
import logging
import re
from typing import Union, Dict
from pathlib import Path

import torch
import yaml  # Import PyYAML for YAML parsing
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh, text_document_helper
from lapps.discriminators import Uri

from postprocessors import apply_postprocessing


class LlamaVideoSummarizer(ClamsApp):

    def __init__(self):
        super().__init__()
        
        self.model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            quantization_config=quantization_config
        )

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        config_file = parameters.get('config')
        config_dir = Path(__file__).parent
        config_file = config_dir / config_file
        self.config = self.load_config(config_file)
        print(f"[DEBUG] Loaded config: {self.config}")
        
        # Set config-dependent variables
        default_prompt = self.config.get("default_prompt", {})
        self.system_message = default_prompt.get("system", "You are a helpful assistant.")
        self.user_prompt = default_prompt.get("user", "")
        print(f"[DEBUG] System message: {self.system_message}")
        print(f"[DEBUG] User prompt: {self.user_prompt[:100]}...")
        
        # Load other config settings
        self.input_context = self.config['context_config'].get('input_context', 'timeframe')
        self.app_mappings = self.config['context_config'].get('apps', {})
        print(f"[DEBUG] Input context: {self.input_context}")
        print(f"[DEBUG] App mappings: {self.app_mappings}")
        
        # Initialize missing attributes
        self.label_mapping = self.config['context_config'].get('timeframe', {}).get('label_mapping', {})
        self.prompts = self.config.get('custom_prompts', {})
        self.postprocessor = self.config.get('postprocessor', None)
        print(f"[DEBUG] Label mapping: {self.label_mapping}")
        print(f"[DEBUG] Custom prompts: {list(self.prompts.keys())}")
        print(f"[DEBUG] Postprocessor: {self.postprocessor}")
        
        self.logger.debug("Running app")
        self.video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.TimeFrame, timeUnit="frame")
        cap = vdh.capture(video_document=self.video_doc)

        video_duration = self.video_doc.get_property('duration') / 1000

        summary_segments = []

        if self.input_context == "timeframe":
            summary_segments = self.generate_summaries_timeframe(mmif, video_duration, parameters)
        elif self.input_context == "fixed_window":
            summary_segments = self.generate_summaries_fixed_window(video_duration, parameters)
        else:
            raise ValueError(f"Unknown input_context: {self.input_context}")

        print(f"Processing {len(summary_segments)} segments...")

        for segment in summary_segments:
            # Pass the label directly from the segment
            context = self.build_context(
                mmif,
                segment['start_time'],
                segment['end_time'],
                segment.get('label')  # Assuming 'label' exists for timeframe contexts
            )
            summary = self.generate_summary(context)
            print(f"Llama output for segment {segment['start_time']}-{segment['end_time']}: {summary}")
            
            # Apply postprocessing if configured
            processed_summary = apply_postprocessing(summary, self.postprocessor, self.logger)
            print(f"[DEBUG] Text document content after postprocessing: {processed_summary}")
            
            text_document = new_view.new_textdocument(processed_summary)
            time_frame = new_view.new_annotation(AnnotationTypes.TimeFrame)
            time_frame.add_property("start", segment['start_time'])
            time_frame.add_property("end", segment['end_time'])
            alignment = new_view.new_annotation(AnnotationTypes.Alignment)
            alignment.add_property("source", time_frame.long_id)
            alignment.add_property("target", text_document.long_id)

        return mmif

    def generate_summaries_timeframe(self, mmif: Mmif, video_duration: float, parameters: dict):
        summary_segments = []
        timeframe_config = self.config['context_config']['timeframe']

        # Iterate over views containing TimeFrame annotations
        views = mmif.get_views_contain(AnnotationTypes.TimeFrame)
        print(f"[DEBUG] Found {len(views)} views containing TimeFrame annotations")
        
        for i, _view in enumerate(views):
            print(f"[DEBUG] View {i} app: {_view.metadata.app}")
            if timeframe_config['app_uri'] in _view.metadata.app:
                print(f"[DEBUG] View {i} matches app_uri: {timeframe_config['app_uri']}")
                timeframes = list(_view.get_annotations(AnnotationTypes.TimeFrame))
                print(f"[DEBUG] Found {len(timeframes)} timeframes in view {i}")
                for timeframe in timeframes:
                    # Get start/end times from target TimePoints 
                    targets = timeframe.properties.get('targets', [])
                    if targets:
                        # Get all timePoints from targets and find min/max
                        timepoints = []
                        for target_id in targets:
                            # Parse the view and timepoint ID from target_id (format: v_X:tp_Y)
                            if ':' in target_id:
                                view_id, tp_id = target_id.split(':', 1)
                                # Find the view containing TimePoints
                                for target_view in mmif.views:
                                    if target_view.id == view_id:
                                        # Look for the timepoint annotation in the target view
                                        for tp_annotation in target_view.get_annotations(AnnotationTypes.TimePoint):
                                            if tp_annotation.id == tp_id:
                                                timepoint = tp_annotation.properties.get('timePoint')
                                                if timepoint is not None:
                                                    timepoints.append(timepoint)
                                                break
                                        break
                        
                        if timepoints:
                            start_time = min(timepoints)
                            end_time = max(timepoints)
                            label = timeframe.properties.get("label")
                            summary_segments.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'label': label
                            })
        return summary_segments

    def generate_summaries_fixed_window(self, video_duration: float, parameters: dict):
        summary_segments = []
        fixed_window = self.config['context_config']['fixed_window']
        window_duration = fixed_window.get('window_duration', 30)
        stride = fixed_window.get('stride', 15)

        start_time = 0
        while start_time < video_duration:
            end_time = min(start_time + window_duration, video_duration)
            summary_segments.append({
                'start_time': start_time,
                'end_time': end_time
            })
            start_time += stride

        return summary_segments

    def build_context(self, mmif: Mmif, start_time: float, end_time: float, label: Union[str, None]) -> str:
        """
        Build context by substituting placeholders in the prompt template
        with content extracted from MMIF views based on the app mappings and label.
        """
        context_dict = {}
        selected_prompt = self.user_prompt  # Default prompt

        for app_uri, placeholder in self.app_mappings.items():
            content = self.extract_content(mmif, app_uri, start_time, end_time)

            context_dict[placeholder] = content

        if label:
            # Select the appropriate prompt based on the label
            label_id = self.label_mapping.get(label)
            if label_id and label_id in self.prompts:
                custom_prompt = self.prompts[label_id]
                # Extract user prompt from dictionary structure if needed
                if isinstance(custom_prompt, dict) and 'user' in custom_prompt:
                    selected_prompt = custom_prompt['user']
                else:
                    selected_prompt = custom_prompt
            else:
                self.logger.warning(f"No prompt found for label: {label}. Using default prompt.")
        else:
            self.logger.warning("No label provided. Using default prompt.")

        # Replace placeholders in the selected prompt format
        selected_prompt = selected_prompt.format(**context_dict)
        return selected_prompt

    def extract_content(self, mmif: Mmif, app_uri: str, start_time: float, end_time: float) -> str:
        """
        Extract content from MMIF views corresponding to the given app URI and time range.
        This looks for TextDocuments from the specified app that are aligned to TimeFrames
        within the given time range.
        """
        contents = []
        
        # Find the view from the specified app (e.g., LLaVA captioner)
        target_view = None
        for view in mmif.views:
            if app_uri in view.metadata.app:
                target_view = view
                break
        
        if not target_view:
            return ""
        
        # Get all TextDocuments from this view
        text_documents = list(target_view.get_annotations(DocumentTypes.TextDocument))
        
        # For each TextDocument, check if it's aligned to a TimePoint in our time range
        for text_doc in text_documents:
            # Get alignments to find what TimePoint this text is aligned to
            for alignment in target_view.get_annotations(AnnotationTypes.Alignment):
                if alignment.properties.get('target') == text_doc.long_id:
                    # This alignment points to our text document
                    source_id = alignment.properties.get('source')
                    if source_id and ':' in source_id:
                        # Parse the timepoint reference (format: v_X:tp_Y)
                        view_id, tp_id = source_id.split(':', 1)
                        
                        # Find the timepoint in the referenced view
                        for tp_view in mmif.views:
                            if tp_view.id == view_id:
                                for tp_annotation in tp_view.get_annotations(AnnotationTypes.TimePoint):
                                    if tp_annotation.id == tp_id:
                                        timepoint = tp_annotation.properties.get('timePoint')
                                        if timepoint is not None:
                                            # Check if this timepoint falls within our desired range
                                            if start_time <= timepoint <= end_time:
                                                if hasattr(text_doc, 'text_value') and text_doc.text_value:
                                                    # Clean the text
                                                    cleaned_text = re.sub(r"\[INST\].*?\[/INST\]", "", text_doc.text_value, flags=re.DOTALL).strip()
                                                    if cleaned_text:
                                                        contents.append(cleaned_text + "\n")
                                        break
                                break
        
        return "".join(contents)

    def generate_summary(self, context: str) -> str:
        # Create messages list for chat template
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": context}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        # print ("prompt: ", prompt)
        # self.logger.debug("model input: " + prompt)
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(prompt, max_new_tokens=100)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.debug("model output: " + summary)
        return summary


def get_app():
    return LlamaVideoSummarizer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    app = LlamaVideoSummarizer()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
