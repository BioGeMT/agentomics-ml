# Agentomics-ML

Agentomics-ML is an autonomous agentic system for development of machine learning models for omics data.

Given a classification dataset, the system automatically generates:
- A trained model file, loaded by the inference script.
- An inference script, allowing immediate predictions on new data.

Agents run inside an isolated docker environment for security.

![example](https://d3mprxcutrc9wk.cloudfront.net/w9a8z0%2Fpreview%2F65550682%2Fmain_large.gif?response-content-disposition=inline%3Bfilename%3D%22main_large.gif%22%3B&response-content-type=image%2Fgif&Expires=1742626823&Signature=VK7h--GDalPwuX7dihN9TAIkwBtWNcqZTH-CDeUkZqz2InKohC3ak-fsRoSWbzmgWFe4ZtTtMXP3PkjXadDp3xbLS-KbameoLlogyQnSNgelFWIF9NLJUpG32M7hUJb~MfI4-8zA8syMraOBbfkndMn7UUwjRDM8wnnjgVOSJQtHKP3Vxs38lvwSWHnPUMTe1dV8H2Jf8zuHih4XNUqhr95WzyQ1x8n71e~1R3e5T0rfZT3iDDIAuMf5CAI5LBONfGOsqIw21jK1A7JRJT5vY2ttuZ8625b4Fd3LEQgGN7jx2z0Xi5NfoyhOyh7oFlX9NcirNoRWKY1DC9Lh3Aa4ug__&Key-Pair-Id=APKAJT5WQLLEOADKLHBQ)

## Environment setup (Docker + conda environment)
- Make sure you have Docker installed
```
docker --version
```
  
- Install plugin for rescricting the volume size
```
docker plugin install ashald/docker-volume-loopback
```

- Create volume with maximum storage size
```
docker volume create -d ashald/docker-volume-loopback:latest -o size=50G agents_volume
```

- Build docker image
```
docker build -t agents_img .
```

- Run the container
```
docker run -d \
    --name agents_cont \
    -v $(pwd):/repository:ro \
    -v agents_volume:/workspace/runs \
    agents_img
```

- Attach console to the container
```
docker exec -it agents_cont bash
python tests/tests.py -v
```
- Activate the default conda environment
```
source activate agentomics-env
```

Create a `.env` file containing your API keys. Example content: `OPENAI_API_KEY=my-api-key-1234`

Run `python src/playground.py` to check everything works

Go to `https://wandb.ai/ceitec-ai/BioAgents` to see agent logs.

## Guidelines and tips

Whatever packages you add, add them to the environment.yaml file

If they are conda packages, you can just re-run
`conda env export --from-history > environment.yaml`

If they are pip packages, you have to manually add them.

You can update your existing environment by running this command
`conda env update --file environment.yaml --prune`