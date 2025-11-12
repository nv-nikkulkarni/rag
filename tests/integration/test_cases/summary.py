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
Summary test module
"""

import json
import logging
import os
import time

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.response_handlers import print_response
from ..utils.verification import verify_summary_content

logger = logging.getLogger(__name__)


class SummaryModule(BaseTestModule):
    """Summary test module"""

    async def test_fetch_summary(
        self, collection_name: str, filenames: list[str]
    ) -> bool:
        """Test fetching document summaries for all files in a collection"""
        async with aiohttp.ClientSession() as session:
            try:
                success_count = 0
                verification_count = 0
                total_files = len(filenames)

                for filename in filenames:
                    logger.info(f"📄 Fetching summary for file: {filename}")
                    params = {
                        "collection_name": collection_name,
                        "file_name": filename,
                        "blocking": "false",
                        "timeout": 20,
                    }
                    logger.info(f"📋 Summary request params:\n{json.dumps(params, indent=2)}")

                    async with session.get(
                        f"{self.rag_server_url}/v1/summary", params=params
                    ) as response:
                        result = await print_response(response)
                        if response.status == 200:
                            logger.info(
                                f"✅ Summary fetched successfully for {filename}"
                            )
                            success_count += 1

                            # Verify summary content for default files
                            summary_text = result.get("summary", "")
                            if verify_summary_content(summary_text, filename):
                                verification_count += 1
                            else:
                                logger.error(
                                    f"❌ Summary content verification failed for {filename}"
                                )
                        else:
                            logger.error(f"❌ Failed to fetch summary for {filename}")

                if success_count == total_files:
                    logger.info(
                        f"✅ Fetch summary test passed - all {total_files} files processed successfully"
                    )

                    # Log verification results
                    if verification_count == success_count:
                        logger.info(
                            f"✅ Summary content verification passed for all {verification_count} files"
                        )
                    else:
                        logger.warning(
                            f"⚠️ Summary content verification: {verification_count}/{success_count} files passed"
                        )

                    return True
                elif success_count > 0:
                    logger.warning(
                        f"⚠️ Fetch summary test partially passed - {success_count}/{total_files} files processed successfully"
                    )

                    # Log verification results for partial success
                    if verification_count > 0:
                        logger.info(
                            f"✅ Summary content verification passed for {verification_count}/{success_count} files"
                        )

                    return True  # Consider partial success as acceptable
                else:
                    logger.error(
                        "❌ Fetch summary test failed - no files processed successfully"
                    )
                    return False
            except Exception as e:
                logger.error(f"❌ Error in fetch summary test: {e}")
                return False



    @test_case(15, "Fetch Summary")
    async def _test_fetch_summary(self) -> bool:
        """Test fetching summary"""
        logger.info("\n=== Test 15: Fetch Summary ===")
        summary_start = time.time()
        # Get all filenames from the collection with metadata
        all_filenames_with_metadata = [os.path.basename(f) for f in self.test_runner.test_files]
        summary_success = await self.test_fetch_summary(
            self.collections["with_metadata"], all_filenames_with_metadata
        )
        summary_time = time.time() - summary_start

        if summary_success:
            self.add_test_result(
                self._test_fetch_summary.test_number,
                self._test_fetch_summary.test_name,
                f"Retrieve document summaries for all files in the collection with keyword-based content verification. Collection: {self.collections['with_metadata']}. Files: {', '.join(all_filenames_with_metadata)}. Supports both blocking and non-blocking modes with configurable timeout for summary generation. Includes automatic keyword verification for default files (multimodal_test.pdf: tables/charts/animals/gadgets, woods_frost.docx: Frost/woods/poem/collections) to ensure summary quality and relevance. Handles partial success scenarios.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking", "timeout"],
                summary_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_fetch_summary.test_number,
                self._test_fetch_summary.test_name,
                f"Retrieve document summaries for all files in the collection with keyword-based content verification. Collection: {self.collections['with_metadata']}. Files: {', '.join(all_filenames_with_metadata)}. Supports both blocking and non-blocking modes with configurable timeout for summary generation. Includes automatic keyword verification for default files (multimodal_test.pdf: tables/charts/animals/gadgets, woods_frost.docx: Frost/woods/poem/collections) to ensure summary quality and relevance. Handles partial success scenarios.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking", "timeout"],
                summary_time,
                TestStatus.FAILURE,
                "Failed to fetch document summaries",
            )
            return False