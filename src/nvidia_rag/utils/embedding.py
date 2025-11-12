# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The wrapper for interacting with embedding models.
1. get_embedding_model: Get the embedding model. Uses the NVIDIA AI Endpoints or HuggingFace.
"""

import logging
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from nvidia_rag.utils.common import ConfigProxy, sanitize_nim_url

logger = logging.getLogger(__name__)
CONFIG = ConfigProxy()


@lru_cache
def get_embedding_model(
    model: str, url: str, truncate: str | None = "END"
) -> Embeddings:
    """Create the embedding model."""

    # Sanitize the URL
    url = sanitize_nim_url(url, model, "embedding")

    logger.info(
        "Using %s as model engine and %s and model for embeddings",
        CONFIG.embeddings.model_engine,
        model,
    )

    if CONFIG.embeddings.model_engine == "nvidia-ai-endpoints":
        if url:
            logger.info("Using embedding model %s hosted at %s", model, url)
            if truncate is not None:
                return NVIDIAEmbeddings(
                    base_url=url,
                    model=model,
                    truncate=truncate,
                    dimensions=CONFIG.embeddings.dimensions,
                )
            else:
                return NVIDIAEmbeddings(
                    base_url=url,
                    model=model,
                    dimensions=CONFIG.embeddings.dimensions,
                )

        logger.info("Using embedding model %s hosted at api catalog", model)
        if truncate is not None:
            return NVIDIAEmbeddings(
                model=model,
                truncate=truncate,
                dimensions=CONFIG.embeddings.dimensions,
            )
        else:
            return NVIDIAEmbeddings(
                model=model,
                dimensions=CONFIG.embeddings.dimensions,
            )

    raise RuntimeError(
        "Unable to find any supported embedding model. Supported engine is huggingface and nvidia-ai-endpoints."
    )
