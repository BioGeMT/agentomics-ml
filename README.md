# MultiAgent-ML

## Environment setup (Docker + conda environment)

```
docker build -t agents_img .
```


```
docker run -d \
    --name agents_cont \
    -v $(pwd):/repository:ro \
    agents_img
```

```
docker exec -it agents_cont bash
python tests/tests.py
source activate multiagent-ml-env
```

Create a `.env` file containing your API keys

## Guidelines and tips

Whatever packages you add, add them to the environment.yaml file

If they are conda packages, you can just re-run
`conda env export --from-history > environment.yaml`

If they are pip packages, you have to manually add them.

You can update your existing environment by running this command
`conda env update --file environment.yaml --prune`

