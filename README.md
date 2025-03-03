# MultiAgent-ML

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
source activate multiagent-ml-env
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

## Jupyter server
If you want to use jupyter notebook, run the container with the following command.

```
docker run -d \
    --name agents_cont \
    -v $(pwd):/repository:ro \
    -v agents_volume:/workspace \
    -p 8888:8888 \
    agents_img
```

Then run this from within the container.

`jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`

When running this command, you will see urls containing a token.

The notebook will then be accessible through `localhost:8888/tree`. The password to connect to it is one of the available tokens.
