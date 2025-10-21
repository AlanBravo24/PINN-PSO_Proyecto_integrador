## Setup virtual environment

Create environment:

```bash
python3 -m venv --system-site-packages ./.venv
source ./.venv/bin/activate
```

Install requirements:

```
pip install -r requirements
```

Install the project module (on the root folder):

```bash
pip install -e ./src
```

## Citation

```
 @article{davi2022pso,
  title={PSO-PINN: Physics-Informed Neural Networks Trained with Particle Swarm Optimization},
  author={Davi, Caio and Braga-Neto, Ulisses},
  journal={arXiv preprint arXiv:2202.01943},
  year={2022}
}
```
Para reproducir los experimentos y generar las figuras basta con correr el archivo de python correspondiente a cada experimento.