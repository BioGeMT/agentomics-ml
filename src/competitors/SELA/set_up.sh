conda create -p /tmp/sela_env python=3.9 -y
source /opt/conda/etc/profile.d/conda.sh
# conda activate "$AGENT_ENV"
conda activate /tmp/sela_env
# pip install --upgrade metagpt #use --no-cache-dir in case of problems
git pull https://github.com/davidcechak/MetaGPT.git
cd MetaGPT
pip install -e .
cd metagpt/ext/sela
pip install -r requirements.txt

pip install agentops==0.4.9 #Fix metagpt env error

pip install wandb
pip install python-dotenv
pip install pyyaml
pip install hrid==0.2.4
pip install pandas
pip install scikit-learn
# This is needed for SELA
pip install openml

# mkdir "$AGENT_DIR"/DI
# cd "$AGENT_DIR"/DI
# metagpt --init-config
mkdir /tmp/sela
cd /tmp/sela
metagpt --init-config

cp /repository/src/competitors/SELA/run.py /tmp/sela/run.py
cp /repository/src/competitors/SELA/set_config.py /tmp/sela/set_config.py

# Give the run permission to write to the DI directory and env
chmod -R 777 /tmp/sela
chmod -R 777 /tmp/sela_env