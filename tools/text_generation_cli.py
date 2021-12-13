# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import sys
import requests

# class PutRequest(urllib.request):
#     '''class to handling putting with urllib2'''

#     def get_method(self, *args, **kwargs):
#         return 'PUT'

if __name__ == "__main__":
    url = "http://0.0.0.0:5000/api"
    while True:
        sentence = input("Enter prompt: ")
        response = requests.put(url, data=json.dumps({"prompts": sentence, "tokens_to_generate":100}), headers={'Content-Type': 'application/json; charset=UTF-8'})
        resp_sentences = response.json()
        print("Megatron Response: ")
        print(resp_sentences["text"][0])
