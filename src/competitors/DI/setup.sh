conda create -p /tmp/di_env python=3.9 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate /tmp/di_env
pip install --no-cache-dir --upgrade metagpt

pip install wandb
pip install python-dotenv
pip install pyyaml
pip install hrid==0.2.4
pip install pandas
pip install scikit-learn

mkdir /tmp/DI
cd /tmp/DI
metagpt --init-config
