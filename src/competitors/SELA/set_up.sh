git submodule add https://github.com/davidcechak/MetaGPT.git
# instead of agentomics yaml we PROBABLY only need the one-shot yaml 
# conda env create -n agentomics_sela_tmp -f environment.yaml
conda env create -n agentomics_sela_tmp -f ../1-shot_llm/environment.yaml
conda activate agentomics_sela_tmp
cd src/competitors/SELA/MetaGPT
pip install -e .
cd metagpt/ext/sela
pip install -r requirements.txt