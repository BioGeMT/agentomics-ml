# MultiAgent-ML

## Environment setup (Docker + conda environment)
Make sure you have Docker installed

```
docker build -t agents_img .
```

```
docker run -d \
    --name agents_cont \
    -v $(pwd):/repository:ro \
    -p 6006:6006 \
    agents_img
```


```
docker exec -it agents_cont bash
python tests/tests.py -v
source activate multiagent-ml-env
```

Create a `.env` file containing your API keys. Example content: `OPENAI_API_KEY=my-api-key-1234`

Run `python src/playground.py` to check everything works

Go to `http://0.0.0.0:6006/projects` to see agent logs (smolagents only).

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
    -p 6006:6006 \
    -p 8888:8888 \
    agents_img
```

Then run this from within the container.

`jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`

When running this command, you will see urls containing a token.

The notebook will then be accessible through `localhost:8888/tree`. The password to connect to it is one of the available tokens.