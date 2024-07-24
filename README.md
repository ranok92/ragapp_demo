Repository for demo of a EnergyGPT: an LLM powered assistant for the enegry sector.

Dependencies:
- Ollama (https://ollama.com/download/windows)
- Nvidia cuda (https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

The code relies on external datasource and databases. Follow the following steps to get started.
There are 2 additional datafolders that is needed:
```
ragapp_demo:
  |- data: raw datafiles to support our LLM perform retrieval augmented generation (RAG)
  |- vectorstores: Vectorstores built from the data for RAG applications.
```

Steps to get started:
1. Paste the datafolders in the root folder.
2. Create conda env using the environment.yaml file.
```
conda env create --name <env_name> --file environment.yaml --python=3.10.14
```
3. Run the setup.py file
```
python install -e .
```

To start the application:

Host the geojson files.
```
cd ragapp_demo/utils
python host_geojson.py --filename anomalous.geojson
```
In a new terminal:
```
cd ragapp_demo/scripts
streamlit run dashboard.py
```
