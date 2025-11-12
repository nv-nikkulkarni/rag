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

"""
Collection management test module
"""

import json
import logging
import time
from typing import Any

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case

logger = logging.getLogger(__name__)


class CollectionManagementModule(BaseTestModule):
    """Collection management test module"""

    CUSTOM_METADATA_COLLECTION = "test_custom_metadata_collection"
    TEST_FILE_TYPES_COLLECTION = "test_file_types"
    CSV_DELETION_BATCH_COLLECTION = "test_csv_deletion_batch"

    @property
    def expected_collections(self):
        """Get all expected collections for this module"""
        return list(self.collections.values()) + [
            self.CUSTOM_METADATA_COLLECTION,
            self.TEST_FILE_TYPES_COLLECTION,
            self.CSV_DELETION_BATCH_COLLECTION,
        ]

    @test_case(2, "Create Collections")
    async def _test_create_collections(self) -> bool:
        """Test creating collections"""
        logger.info("\n=== Test 2: Create Collections ===")
        collection_start = time.time()

        # Basic metadata schema for standard collections
        basic_metadata_schema = [
            {
                "name": "timestamp",
                "type": "datetime",
                "description": "Timestamp of when the document was created",
            },
            {
                "name": "meta_field_1",
                "type": "string",
                "description": "Description for the document",
            },
        ]

        # Custom metadata schema for custom metadata tests
        custom_metadata_schema = [
            {
                "name": "title",
                "type": "string",
                "required": True,
                "max_length": 200,
                "description": "Document title",
            },
            {
                "name": "category",
                "type": "string",
                "required": False,
                "description": "Document category",
            },
            {
                "name": "rating",
                "type": "float",
                "required": False,
                "description": "Document rating",
            },
            {
                "name": "is_public",
                "type": "boolean",
                "required": False,
                "description": "Whether document is public",
            },
            {
                "name": "tags",
                "type": "array",
                "array_type": "string",
                "max_length": 50,
                "required": False,
                "description": "Document tags",
            },
            {
                "name": "created_date",
                "type": "datetime",
                "required": True,
                "description": "Document creation date",
            },
            {
                "name": "updated_date",
                "type": "datetime",
                "required": False,
                "description": "Document update date",
            },
        ]

        collection1_success = await self._create_collection(
            self.collections["with_metadata"], basic_metadata_schema
        )
        collection2_success = await self._create_collection(
            self.collections["without_metadata"]
        )
        collection3_success = await self._create_collection(
            self.CUSTOM_METADATA_COLLECTION, custom_metadata_schema
        )
        collection4_success = await self._create_collection(
            self.TEST_FILE_TYPES_COLLECTION
        )
        collection5_success = await self._create_collection(
            self.CSV_DELETION_BATCH_COLLECTION
        )
        collection_time = time.time() - collection_start

        if (
            collection1_success
            and collection2_success
            and collection3_success
            and collection4_success
            and collection5_success
        ):
            self.add_test_result(
                self._test_create_collections.test_number,
                self._test_create_collections.test_name,
                f"Create five test collections - one with basic metadata schema, one without metadata, one with custom metadata schema, one for file type testing, and one for CSV deletion batch processing. Collections: {', '.join(self.expected_collections)}. Basic metadata schema includes fields: timestamp (datetime), meta_field_1 (string). Custom metadata schema includes fields: title, category, rating, is_public, tags, created_date, updated_date.",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension", "metadata_schema"],
                collection_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_create_collections.test_number,
                self._test_create_collections.test_name,
                f"Create five test collections - one with basic metadata schema, one without metadata, one with custom metadata schema, one for file type testing, and one for CSV deletion batch processing. Collections: {', '.join(self.expected_collections)}. Basic metadata schema includes fields: timestamp (datetime), meta_field_1 (string). Custom metadata schema includes fields: title, category, rating, is_public, tags, created_date, updated_date.",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension", "metadata_schema"],
                collection_time,
                TestStatus.FAILURE,
                "Failed to create one or more collections",
            )
            return False

    @test_case(3, "Verify Collections")
    async def _test_verify_collections(self) -> bool:
        """Test verifying collections"""
        logger.info("\n=== Test 3: Verify Collections ===")
        verify_start = time.time()
        verify_success = await self._verify_collections()
        verify_time = time.time() - verify_start

        if verify_success:
            self.add_test_result(
                self._test_verify_collections.test_number,
                self._test_verify_collections.test_name,
                f"Verify collections are created and metadata schema is properly configured. Collections: {', '.join(self.expected_collections)}. Validates metadata schema fields (timestamp: datetime, meta_field_1: string) with type and description verification.",
                ["GET /v1/collections"],
                [
                    "total_collections",
                    "collections[].collection_name",
                    "collections[].metadata_schema",
                ],
                verify_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_verify_collections.test_number,
                self._test_verify_collections.test_name,
                f"Verify collections are created and metadata schema is properly configured. Collections: {', '.join(self.expected_collections)}. Validates metadata schema fields (timestamp: datetime, meta_field_1: string) with type and description verification.",
                ["GET /v1/collections"],
                [
                    "total_collections",
                    "collections[].collection_name",
                    "collections[].metadata_schema",
                ],
                verify_time,
                TestStatus.FAILURE,
                "Collection verification failed",
            )
            return False

    async def _create_collection(
        self, collection_name: str, metadata_schema: list[dict[str, Any]] = None
    ) -> bool:
        """Create a collection with optional metadata schema"""
        try:
            payload = {
                "collection_name": collection_name,
                "embedding_dimension": 2048,
            }

            if metadata_schema:
                payload["metadata_schema"] = metadata_schema

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ingestor_server_url}/v1/collection", json=payload
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Collection '{collection_name}' created successfully:"
                        )
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to create collection '{collection_name}': {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error creating collection '{collection_name}': {e}")
            return False

    async def _create_collections(self, collection_names: list[str]) -> bool:
        """API to create multiple collections"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ingestor_server_url}/v1/collections", json=collection_names
                ) as response:
                    result = await response.json()
                    if (
                        response.status == 200
                        and result.get("successful") == collection_names
                    ):
                        logger.info(
                            f"✅ Collections '{collection_names}' created successfully:"
                        )
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to create collections '{collection_names}': {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error creating collections '{collection_names}': {e}")
            return False

    async def _verify_collections(self) -> bool:
        """Verify collections are created and metadata schema is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ingestor_server_url}/v1/collections"
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        collections = result.get("collections", [])
                        logger.info("✅ Collections retrieved successfully:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        # Check if our test collections exist
                        collection_names = [
                            col.get("collection_name") for col in collections
                        ]
                        expected_collections = self.expected_collections

                        missing_collections = [
                            name
                            for name in expected_collections
                            if name not in collection_names
                        ]
                        if missing_collections:
                            logger.error(
                                f"❌ Missing collections: {missing_collections}"
                            )
                            return False

                        # Verify metadata schema for collection with metadata
                        metadata_collection = self.collections["with_metadata"]
                        for collection in collections:
                            if collection.get("collection_name") == metadata_collection:
                                schema = collection.get("metadata_schema", [])
                                if not self._validate_metadata_schema(
                                    schema, metadata_collection
                                ):
                                    return False
                                break

                        # Verify metadata schema for custom metadata collection
                        custom_metadata_collection = self.CUSTOM_METADATA_COLLECTION
                        for collection in collections:
                            if (
                                collection.get("collection_name")
                                == custom_metadata_collection
                            ):
                                schema = collection.get("metadata_schema", [])
                                if not self._validate_custom_metadata_schema(
                                    schema, custom_metadata_collection
                                ):
                                    return False
                                break

                        logger.info(
                            f"✅ Collections verified successfully: {collection_names}"
                        )
                        return True
                    else:
                        logger.error(f"❌ Failed to get collections: {response.status}")
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error verifying collections: {e}")
            return False

    def _validate_metadata_schema(
        self, actual_schema: list[dict[str, Any]], collection_name: str
    ) -> bool:
        """Validate that the actual metadata schema matches the expected schema"""
        expected_fields = {
            "timestamp": {
                "type": "datetime",
                "description": "Timestamp of when the document was created",
            },
            "meta_field_1": {
                "type": "string",
                "description": "Description for the document",
            },
        }

        actual_fields = {field["name"]: field for field in actual_schema}

        for field_name, expected_config in expected_fields.items():
            if field_name not in actual_fields:
                logger.error(
                    f"❌ Missing field '{field_name}' in collection '{collection_name}'"
                )
                return False

            actual_field = actual_fields[field_name]
            if actual_field.get("type") != expected_config["type"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong type. Expected: {expected_config['type']}, Got: {actual_field.get('type')}"
                )
                return False

            if actual_field.get("description") != expected_config["description"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong description. Expected: {expected_config['description']}, Got: {actual_field.get('description')}"
                )
                return False

        logger.info(
            f"✅ Metadata schema validation passed for collection '{collection_name}'"
        )
        return True

    def _validate_custom_metadata_schema(
        self, actual_schema: list[dict[str, Any]], collection_name: str
    ) -> bool:
        """Validate that the actual custom metadata schema matches the expected schema"""
        expected_fields = {
            "title": {
                "type": "string",
                "required": True,
                "max_length": 200,
                "description": "Document title",
            },
            "category": {
                "type": "string",
                "required": False,
                "description": "Document category",
            },
            "rating": {
                "type": "float",
                "required": False,
                "description": "Document rating",
            },
            "is_public": {
                "type": "boolean",
                "required": False,
                "description": "Whether document is public",
            },
            "tags": {
                "type": "array",
                "array_type": "string",
                "max_length": 50,
                "required": False,
                "description": "Document tags",
            },
            "created_date": {
                "type": "datetime",
                "required": True,
                "description": "Document creation date",
            },
            "updated_date": {
                "type": "datetime",
                "required": False,
                "description": "Document update date",
            },
        }

        actual_fields = {field["name"]: field for field in actual_schema}

        for field_name, expected_config in expected_fields.items():
            if field_name not in actual_fields:
                logger.error(
                    f"❌ Missing field '{field_name}' in collection '{collection_name}'"
                )
                return False

            actual_field = actual_fields[field_name]
            if actual_field.get("type") != expected_config["type"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong type. Expected: {expected_config['type']}, Got: {actual_field.get('type')}"
                )
                return False

            if actual_field.get("required") != expected_config["required"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong required flag. Expected: {expected_config['required']}, Got: {actual_field.get('required')}"
                )
                return False

            if (
                "max_length" in expected_config
                and actual_field.get("max_length") != expected_config["max_length"]
            ):
                logger.error(
                    f"❌ Field '{field_name}' has wrong max_length. Expected: {expected_config['max_length']}, Got: {actual_field.get('max_length')}"
                )
                return False

            if (
                "array_type" in expected_config
                and actual_field.get("array_type") != expected_config["array_type"]
            ):
                logger.error(
                    f"❌ Field '{field_name}' has wrong array_type. Expected: {expected_config['array_type']}, Got: {actual_field.get('array_type')}"
                )
                return False

            if actual_field.get("description") != expected_config["description"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong description. Expected: {expected_config['description']}, Got: {actual_field.get('description')}"
                )
                return False

        logger.info(
            f"✅ Custom metadata schema validation passed for collection '{collection_name}'"
        )
        return True
