# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
from dataclasses import asdict
from typing import Dict, List, Union

from api.api import CompletionRequest, OpenAiApiGenerator
from api.models import get_model_info_list, retrieve_model_info

from build.builder import BuilderArgs, TokenizerArgs
from flask import Flask, request, Response
from flask_cors import CORS  # Importing CORS
from generate import GeneratorArgs

OPENAI_API_VERSION = "v1"

def create_app(args):
    """
    Creates a flask app that can be used to serve the model as a chat API.
    """
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

    gen: OpenAiApiGenerator = initialize_generator(args)

    def _del_none(d: Union[Dict, List]) -> Union[Dict, List]:
        """Recursively delete None values from a dictionary."""
        if isinstance(d, dict):
            return {k: _del_none(v) for k, v in d.items() if v}
        elif isinstance(d, list):
            return [_del_none(v) for v in d if v]
        return d

    @app.errorhandler(Exception)
    def handle_error(e):
        print(f"An error occurred: {str(e)}")
        return str(e), 500

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    @app.route("/chat", methods=["OPTIONS"])
    @app.route("/{OPENAI_API_VERSION}/chat", methods=["OPTIONS"])
    def options_handler(path):
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        return response

    @app.route("/chat", methods=["POST"])
    @app.route(f"/{OPENAI_API_VERSION}/chat", methods=["POST"])
    def chat_endpoint():
        """
        Endpoint for the Chat API. This endpoint is used to generate a response to a user prompt.
        This endpoint emulates the behavior of the OpenAI Chat API. (https://platform.openai.com/docs/api-reference/chat)
        """
        print(" === Completion Request ===")

        # Parse the request into a CompletionRequest object
        data = request.get_json()
        req = CompletionRequest(**data)

        match = re.search(r'"(.*?)"', req.messages[1]['content'])
        if match:
          extracted_value = match.group(1)
          print(f" content: {extracted_value}")

        if data.get("stream") == "true":

            def chunk_processor(chunked_completion_generator):
                """Inline function for postprocessing CompletionResponseChunk objects.

                Here, we just jsonify the chunk and yield it as a string.
                """
                for chunk in chunked_completion_generator:
                    if (next_tok := chunk.choices[0].delta.content) is None:
                        next_tok = ""
                    print(next_tok, end="")
                    yield json.dumps(_del_none(asdict(chunk)))

            return Response(
                chunk_processor(gen.chunked_completion(req)),
                mimetype="text/event-stream",
            )
        else:
            response = gen.sync_completion(req)

            return json.dumps(_del_none(asdict(response)))

    @app.route(f"/models", methods=["GET"])
    @app.route(f"/{OPENAI_API_VERSION}/models", methods=["GET"])
    def models_endpoint():
        return json.dumps(asdict(get_model_info_list(args)))

    @app.route(f"/models/<model_id>", methods=["GET"])
    @app.route(f"/{OPENAI_API_VERSION}/models/<model_id>", methods=["GET"])
    def models_retrieve_endpoint(model_id):
        if response := retrieve_model_info(args, model_id):
            return json.dumps(asdict(response))
        else:
            return "Model not found", 404

    return app

def initialize_generator(args) -> OpenAiApiGenerator:
    builder_args = BuilderArgs.from_args(args)
    speculative_builder_args = BuilderArgs.from_speculative_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)
    generator_args.chat_mode = False

    return OpenAiApiGenerator(
        builder_args=builder_args,
        speculative_builder_args=speculative_builder_args,
        tokenizer_args=tokenizer_args,
        generator_args=generator_args,
        profile=args.profile,
        quantize=args.quantize,
        draft_quantize=args.draft_quantize,
    )

def main(args):
    app = create_app(args)
    app.run(host="0.0.0.0", port=5001)
