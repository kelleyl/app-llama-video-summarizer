"""
Postprocessing functions for Llama model outputs.

This module provides optional postprocessing functions that can be applied 
to the raw Llama output before creating text documents in the MMIF.
"""

import json
import re
import logging
from typing import Optional, Callable, Dict


class PostProcessor:
    """Base class for postprocessing functions."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def process(self, text: str) -> str:
        """
        Process the input text and return the modified version.
        
        Args:
            text: Raw text output from Llama model
            
        Returns:
            Processed text
        """
        raise NotImplementedError("Subclasses must implement process method")


class JSONExtractorPostProcessor(PostProcessor):
    """Extracts the first JSON substring from the Llama output."""
    
    def process(self, text: str) -> str:
        """
        Extract the first valid JSON substring from the input text.
        
        Args:
            text: Raw text output from Llama model
            
        Returns:
            First valid JSON string found, or original text if no JSON found
        """
        self.logger.debug(f"JSONExtractor input: {text[:200]}...")
        
        # Look for JSON-like patterns (starting with { or [)
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested object pattern
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Simple nested array pattern
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Validate it's actually valid JSON
                    json.loads(match)
                    self.logger.debug(f"JSONExtractor extracted: {match}")
                    return match.strip()
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON with more complex nesting
        try:
            # Look for balanced braces/brackets
            for start_char, end_char in [('{}'), ('[]')]:
                start_idx = text.find(start_char[0])
                if start_idx != -1:
                    count = 0
                    for i, char in enumerate(text[start_idx:], start_idx):
                        if char == start_char[0]:
                            count += 1
                        elif char == end_char:
                            count -= 1
                            if count == 0:
                                candidate = text[start_idx:i+1]
                                try:
                                    json.loads(candidate)
                                    self.logger.debug(f"JSONExtractor extracted (balanced): {candidate}")
                                    return candidate.strip()
                                except json.JSONDecodeError:
                                    break
        except Exception as e:
            self.logger.warning(f"Error in JSON extraction: {e}")
        
        self.logger.warning("No valid JSON found in text, returning original")
        return text


# Registry of available postprocessors
POSTPROCESSORS: Dict[str, Callable[..., PostProcessor]] = {
    'json_extractor': JSONExtractorPostProcessor,
}


def get_postprocessor(name: str, logger: Optional[logging.Logger] = None) -> Optional[PostProcessor]:
    """
    Get a postprocessor by name.
    
    Args:
        name: Name of the postprocessor
        logger: Optional logger instance
        
    Returns:
        PostProcessor instance or None if not found
    """
    if name in POSTPROCESSORS:
        return POSTPROCESSORS[name](logger=logger)
    return None


def apply_postprocessing(text: str, postprocessor_name: Optional[str], 
                        logger: Optional[logging.Logger] = None) -> str:
    """
    Apply postprocessing to text if a postprocessor is specified.
    
    Args:
        text: Input text to process
        postprocessor_name: Name of postprocessor to apply, or None
        logger: Optional logger instance
        
    Returns:
        Processed text
    """
    if not postprocessor_name:
        return text
    
    processor = get_postprocessor(postprocessor_name, logger)
    if processor:
        return processor.process(text)
    else:
        if logger:
            logger.warning(f"Unknown postprocessor: {postprocessor_name}")
        return text