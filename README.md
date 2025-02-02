# hedonic
Community Detection using Hedonic Game Theory

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the repository

```sh
git clone https://github.com/lucaslopes/hedonic.git
```

### 2. Navigate into the project directory

```sh
cd hedonic
```

### 3. Create a virtual environment


```sh
python -m venv venv
pip install --upgrade pip
```


### 4. Activate the virtual environment

#### On macOS and Linux:

```sh
. venv/bin/activate
```

#### On Windows:

```sh
.\venv\Scripts\activate
```

### 5. Install the Hedonic package dependencies in editable mode

```sh
pip install -e .
```

Using pip install `-e .` installs the project in "editable" mode, which means any changes made to the project files will be immediately reflected without needing to reinstall the package. This is particularly useful during development, as it allows you to see the effects of your changes right away.

## Usage

The `hedonic.Game` class extends from `igraph.Graph`, inheriting all its methods and properties. This means you can use it in a similar way to how you use `igraph.Graph`, with additional functionality specific to hedonic games.

Here's an example of how to use it:

```python
# Import the package
import hedonic as hd

# Create a famous graph, like the Zachary's Karate Club graph
G = hd.Game.Famous('Zachary')
# G = hd.Game(graph)  # or if you already have an igraph.Graph object

# You can now detect communities using the `community_hedonic()` method:
partition = G.community_hedonic()

# You can also compare the detected communities with the ground truth communities:
ground_truth = G.vs['community']
accuracy = hd.compare_communities(partition, ground_truth, method='adjusted_rand')
print(f'Accuracy: {accuracy}')
```

In this example, `G` is an instance of the `Game` class created from a famous graph, similar to how you would use `igraph.Graph.Famous`. The `Game` class inherits from `igraph.Graph`, so you can use all the standard `igraph.Graph` methods on `G` while also having access to additional methods specific to hedonic games, such as `community_hedonic()`.

## Contributing

If you would like to contribute, please fork the repository and use a feature branch. Pull requests are welcome.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
```