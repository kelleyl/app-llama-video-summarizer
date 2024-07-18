"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from lapps.discriminators import Uri


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification.
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="Llama Video Summarizer",
        description="Summarizes video content using Llama model based on captions and transcript",
        app_license="",  # short name for a software license like MIT, Apache2, GPL, etc.
        identifier="llama-video-summarizer",  # should be a single string without whitespaces. If you don't intent to publish this app to the CLAMS app-directory, please use a full IRI format.
        url="https://github.com/clamsproject/app-llama-video-summarizer",
        # (if you are on the CLAMS team, this MUST be "https://github.com/clamsproject/app-llama-video-summarizer"
        # (see ``.github/README.md`` file in this directory for the reason)
        analyzer_version='version_X', # use this IF THIS APP IS A WRAPPER of an existing computational analysis algorithm
        # (it is very important to pinpoint the primary analyzer version for reproducibility)
        # (for example, when the app's implementation uses ``torch``, it doesn't make the app a "torch-wrapper")
        # (but, when the app doesn't implementaion any additional algorithms/model/architecture, but simply use API's of existing, for exmaple, OCR software, it is a wrapper)
        # if the analyzer is a python app, and it's specified in the requirements.txt
        # this trick can also be useful (replace ANALYZER_NAME with the pypi dist name)
        # analyzer_version=[l.strip().rsplit('==')[-1] for l in open('requirements.txt').readlines() if re.match(r'^ANALYZER_NAME==', l)][0],
        analyzer_license="",  # short name for a software license
    )
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_input(AnnotationTypes.TimeFrame)
    metadata.add_input(Uri.TOKEN)
    metadata.add_input(DocumentTypes.TextDocument)

    metadata.add_output(DocumentTypes.TextDocument)

    metadata.add_parameter(
        name="stride",
        type="integer",
        default=60,
        description="Time stride in seconds for summarization"
    )

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
