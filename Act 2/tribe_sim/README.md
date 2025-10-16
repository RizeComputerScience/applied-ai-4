# Hunter-Gatherer Tribe Simulation

A genetic algorithm teaching tool that simulates hunter-gatherer tribes evolving survival strategies while balancing food gathering and predator avoidance.

## Installation

1. Install Python 3.8 or higher
2. Install pygame:
   ```bash
   pip install pygame
   ```
   Or install from requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Assets

The simulation uses PNG images for visual sprites. Place the following files in the `assets/` directory:
- `apple.png` - Food items
- `friendly.png` - Evolving tribe members (your tribe)
- `ninja.png` - Ninja tribe members (3 lives each)
- `runner.png` - Runner tribe members (super fast, no caution)
- `farmer.png` - Farmer tribe members (double food value)
- `jaguar.png` - Predators

If any image files are missing, the simulation will fall back to basic geometric shapes.

## Running the Simulation

```bash
python simulation.py
```

## How It Works

### Entities
- **Evolving Tribe** (friendly.png): Your named tribe with evolving genes for speed, caution, search patterns, and efficiency (10 members)
- **Ninja Tribe** (ninja.png): 3 lives each, teleports to random location when hit, base speed, normal caution (5 members)
- **Runner Tribe** (runner.png): Super fast (3.5 speed), no predator avoidance, dies in one hit (5 members)  
- **Farmer Tribe** (farmer.png): Base speed, high caution, gets double food value (5 members)
- **Predators** (jaguar.png): Hunt all tribes within their detection radius (5 predators)
- **Food** (apple.png): Large icons, spawn in clusters, respawn after being consumed, compete for by all tribes

### Genetic Algorithm
- **Population**: 10 evolving tribe members per generation
- **Selection**: Top 30% survive based on fitness (survival time × food collected)
- **Reproduction**: Remaining 70% are offspring from survivors
- **Mutation**: 10% chance per gene with ±20% variation
- **Competition**: Competing tribes respawn fresh each generation (no evolution)

### Controls
- **SPACE**: Pause/Resume simulation
- **N**: Advance to next generation manually
- **R**: Reset simulation
- **ESC**: Quit
- **Mouse**: Click "Next Generation" and "Reset" buttons

### Observable Evolution
Watch your evolving tribe compete against specialized competitors:
- **Early generations**: Your tribe struggles against specialized competitors
- **Mid generations**: Evolution begins optimizing survival strategies  
- **Late generations**: Your tribe develops competitive advantages through genetic optimization
- **Visual feedback**: Only your evolving tribe shows fitness colors (red→yellow→green)
- **Competition dynamics**: Each tribe has unique strengths, creating complex survival challenges

## Educational Value

This simulation demonstrates key concepts in evolutionary biology and genetic algorithms:
- Natural selection pressure
- Trade-offs in survival strategies
- Mutation and genetic diversity
- Fitness landscapes
- Population dynamics

Perfect for teaching evolution, AI, and optimization concepts!