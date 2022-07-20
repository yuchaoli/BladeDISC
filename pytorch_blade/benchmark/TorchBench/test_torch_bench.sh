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
OLD_HOME=$HOME
echo $OLD_HOME

python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
pip3 install -U pip
pip3 install -q librosa torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cu113

script_dir=$(cd $(dirname "$0"); pwd)
# setup for torchbenchmark
benchmark_repo_dir=$OLD_HOME/.cache/torchbenchmark
echo $benchmark_repo_dir
if [ ! -d $benchmark_repo_dir ]; then
    git clone -q https://github.com/pytorch/benchmark.git --recursive $benchmark_repo_dir
fi
cd $benchmark_repo_dir && export HOME=$(pwd) && git lfs install --force && git pull && git submodule update --init --recursive --depth 1 && python3 install.py

pushd $script_dir # pytorch_blade/benchmark/TorchBench
# setup for torchdynamo
echo $(pwd)
git clone -q https://github.com/pytorch/torchdynamo.git dynamo && pip3 install -q dynamo/

# dynamo frontend and disc backend
python3 blade_bench.py --backend blade_disc_compiler -d cuda --isolate --float32 --skip-accuracy-check 2>&1 | tee speedup_blade.log
cat speedup_blade_disc_compiler.csv
popd
