# G-PFL-IDS: Graph-Driven Personalized Federated Learning for Unsupervised Intrusion Detection in Non-IID IoT Systems

![Alt text](./G-PFL-IDS-1.png "Optional title text")

A Flower-based implementation of Graph-Driven Personalized Federated Learning for Unsupervised Intrusion Detection in Non-IID
IoT Systems.

## Install dependencies and project

1. Create a virtual environment:

```bash
python -m venv g-pfl-ids
source g-pfl-ids/bin/activate  # On Windows: fl_gcn\Scripts\activate

# then git clone the project:
git clone https://github.com/danieloladele7/g-pfl-ids.git
```

2. **Configuration:** The dependencies are listed in the `pyproject.toml` and you can edit them and install them using `pip install -e .`. Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html):

## Usage

1. Process the IoT-23 data:

```bash
python scripts/run_pipeline.py --input data/raw/IoT23/conn.log.labeled --processed_dir data/processed --graph_dir data/graph_data --n_clients 10 --mode binary
```

2. **Simulation** In the `G-PFL-IDS` directory run the FL training using. For more check [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html):

```bash
flwr run .
```

## Project Structure

```text
fl-gcn-ids/
├── data/
│   ├── raw/
│   │   └── IoT23/
│   │       └── conn.log.labeled
│   ├── processed/
│   └── graph_data/
├── src/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── analyze_non_iid.py
│   │   ├── malicious_evaluator.py
│   │   └── non_iid_metrics.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gcn.py
│   │   └── gae.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   └── trainers.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py
│   │   ├── graph_builder.py
│   │   ├── oneclass_metrics.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── config.py
│   ├── client_app.py
│   ├── server_app.py
│   └── personalize.py
├── pyproject.toml
└── README.md
```

## Run with the Simulation Engine

In the `FL-GCN-IoT-IDS` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

## Citation:

If you use `g-pfl-ids` for your research, please consider citing our paper:

```bibtex
@article{Oladele2025_g-pfl-ids,
    title = {G-PFL-IDS: Graph-Driven Personalized Federated Learning for Unsupervised Intrusion Detection in Non-IID IoT Systems},
    author = {Daniel A. Oladele, Ige Ayokunle, Agbo-Ajala Olatunbosun, Ekundayo Olufisayo, Gaanesh Sree, Sibiya Malusi, Mnkandla
Ernest},
    year = {2025},
    journal = {Smart Cities},
    volume = {},
    number = {},
    pages = {},
    doi = {},
    url = {https://github.com/danieloladele7/g-pfl-ids.git},
}
```

## Resources

- For more example on using the Flower's Deployment Engine:
  - [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)
  - [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html)
  - [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html).
- Docker implementation guide: [Flower with Docker](https://flower.ai/docs/framework/docker/index.html).
- Guide to optimizing simulations: [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html).
- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
