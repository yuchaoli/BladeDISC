# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/bin/bash
# install dependencies
export HOME=$(pwd)
git lfs install --force
python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
pip3 install librosa torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cu113

script_dir=$(cd $(dirname "$0"); pwd)
pushd $script_dir # pytorch_blade/benchmark/TorchBench
# setup for torchbenchmark
git clone -q https://github.com/pytorch/benchmark.git --recursive torchbenchmark
cd torchbenchmark && python3 install.py 

# setup for torchdynamo
cd $script_dir && git clone -q https://github.com/pytorch/torchdynamo.git dynamo && pip3 install dynamo/

# dynamo frontend and disc backend
python3 blade_bench.py --backend blade_disc_compiler -d cuda --isolate --float32 --skip-accuracy-check 2>&1 | tee speedup_blade.log

popd
