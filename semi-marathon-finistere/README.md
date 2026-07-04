# Race Planner: Unified Event Planner Entry

The active Streamlit entry in this directory now launches the repository's
unified event planner. The original Semi-Marathon du Finistère logic and the
notebook remain in the repository for compatibility, reference, and regression
testing.

## Development Environment (Repository Standard)

Use the repository root for all development commands.

```bash
uv sync
uv run streamlit run semi-marathon-finistere/app.py
```

Deployment dependencies for Streamlit Community Cloud are generated via:

```bash
scripts/export_streamlit_requirements.sh
```

Do not hand-edit `semi-marathon-finistere/requirements.txt`.

---

## English

### Overview

This repository now serves a curated event-first planning flow through the
Streamlit app in this directory. It still includes the original Semi-Marathon du
Finistère material, but the active app experience is no longer split between
legacy and beta planner modes.

Two versions are available:

- **Streamlit Web App** (Recommended) - Fast, modern web interface
- **Jupyter Notebook** - Original version for Voila

### Features

- **Bilingual interface** (English/French toggle in the sidebar)
- Curated event-first selection across road and trail events
- Road events: race intent, split bias, aid stop timing
- Trail/ultra events: climb-to-hike threshold, descent caution, fade profiles, effort policy
- Athlete profile with LT1/LT2 baselines, road capability, trail adjustments, and universal factors
- Weather-driven heat penalty with diurnal temperature modeling
- Fueling and nutrition plan with per-block calorie, carb, and fluid targets
- Per-kilometer and aggregated split pacing with elevation gain/loss
- JSON plan and athlete profile export/import

### Unified Planner Highlights

- Curated event-first selection
- Race models:
  - `half_marathon`
  - `road_marathon`
  - `fire_road_ultra` (Z1/Z2/hike anchors)
  - `technical_trail_ultra` (flat/hike + descent caution)
- Input modes:
  - `finish_time`
  - `effort_anchor`
- JSON plan export/import
- Repository-backed curated GPX files for supported events

If a saved plan references a curated GPX file that is not available locally,
restore the file in the repository and retry.

### Quick Start

#### Option 1: Streamlit Web App (Recommended)

1. **Install Python 3.9+** if you haven't already

2. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/race-planners.git
   cd race-planners/semi-marathon-finistere
   ```

3. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```bash
   uv sync
   ```

5. **Launch the Streamlit app:**

   ```bash
   uv run streamlit run semi-marathon-finistere/app.py
   ```

6. Your browser will open automatically at `http://localhost:8501`

#### Option 2: Run with Voila (Jupyter Notebook)

For the original notebook interface:

1. **Install Voila dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch with Voila:**
   ```bash
   voila race_planner_semi_marathon_finistere.ipynb
   ```

#### Option 3: Run in Jupyter Notebook

If you want to see and modify the code:

```bash
jupyter notebook race_planner_semi_marathon_finistere.ipynb
```

Then run all cells (Cell → Run All) to activate the interactive widgets.

### Course Information

| Attribute      | Value                   |
| -------------- | ----------------------- |
| Distance       | 21.06 km                |
| Elevation Gain | ~153m                   |
| Rest Stops     | 5.3 km, 9.1 km, 14.5 km |

### How to Use

1. **Select a curated event**
2. **Choose input mode:** finish time or effort anchor
3. **Enter your target inputs** for the selected event template
4. **Adjust pacing bias or rest assumptions** when relevant
5. **Click "Calculate Plan"** to generate pacing, aid-station, and section outputs

---

## Français

### Aperçu

Cette application Streamlit sert maintenant de point d'entrée au planificateur
unifié d'événements du dépôt. Le contenu historique du Semi-Marathon du
Finistère reste présent pour compatibilité et référence, mais l'expérience
active n'est plus séparée entre un mode legacy et un mode beta.

Deux versions sont disponibles:

- **Application Web Streamlit** (Recommandé) - Interface web moderne et rapide
- **Notebook Jupyter** - Version originale pour Voila

### Fonctionnalités

- Interface bilingue (Anglais/Français)
- Saisie du temps cible OU de l'allure cible
- Ajustement de la gestion d'effort pour les splits négatifs/positifs
- Timing et stratégie aux ravitaillements
- Conseils d'allure spécifiques pour chaque section du parcours
- Carte poche et brassard imprimables
- Visualisation du profil altimétrique
- Détail de l'allure par kilomètre

### Démarrage Rapide

#### Option 1: Application Web Streamlit (Recommandé)

1. **Installez Python 3.9+** si ce n'est pas déjà fait

2. **Clonez le dépôt:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/race-planners.git
   cd race-planners/semi-marathon-finistere
   ```

3. **Créez un environnement virtuel (optionnel mais recommandé):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

4. **Installez les dépendances:**

   ```bash
   uv sync
   ```

5. **Lancez l'application Streamlit:**

   ```bash
   uv run streamlit run semi-marathon-finistere/app.py
   ```

6. Votre navigateur s'ouvrira automatiquement à `http://localhost:8501`

#### Option 2: Exécuter avec Voila (Notebook Jupyter)

Pour l'interface originale du notebook:

1. **Installez les dépendances Voila:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Lancez avec Voila:**
   ```bash
   voila race_planner_semi_marathon_finistere.ipynb
   ```

#### Option 3: Exécuter dans Jupyter Notebook

Si vous voulez voir et modifier le code:

```bash
jupyter notebook race_planner_semi_marathon_finistere.ipynb
```

Puis exécutez toutes les cellules (Cell → Run All) pour activer les widgets interactifs.

### Informations sur le Parcours

| Attribut         | Valeur                  |
| ---------------- | ----------------------- |
| Distance         | 21,06 km                |
| Dénivelé positif | ~153m                   |
| Ravitaillements  | 5,3 km, 9,1 km, 14,5 km |

### Comment Utiliser

1. **Sélectionnez un événement pris en charge**
2. **Choisissez le mode de saisie:** temps cible ou ancrage d'effort
3. **Entrez vos paramètres cibles** pour le modèle associé à l'événement
4. **Ajustez si besoin** le biais d'allure ou la durée des arrêts aux ravitos
5. **Cliquez sur "Calculate Plan"** pour générer les sorties d'allure, de
   ravitaillement et de sections

---

## Technical Details / Détails Techniques

### Dependencies / Dépendances

**Streamlit App:**

```
streamlit>=1.28.0
numpy
matplotlib
pandas
```

**Jupyter Notebook (Voila):**

```
numpy
matplotlib
ipywidgets
voila
```

### Files / Fichiers

| File                                         | Description                                                    |
| -------------------------------------------- | -------------------------------------------------------------- |
| `app.py`                                     | Unified planner entry / Point d'entrée du planificateur unifié |
| `race_planner_semi_marathon_finistere.ipynb` | Original Jupyter notebook / Notebook Jupyter original          |
| `WR-GPX-Semi-marathon-du-Finistere.gpx`      | Course GPX data / Données GPX du parcours                      |
| `requirements-streamlit.txt`                 | Streamlit dependencies / Dépendances Streamlit                 |
| `requirements.txt`                           | Voila dependencies / Dépendances Voila                         |
| `.streamlit/config.toml`                     | Streamlit configuration / Configuration Streamlit              |
| `README.md`                                  | This file / Ce fichier                                         |

---

## Deployment / Déploiement

### Streamlit Cloud (Free / Gratuit)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select the `semi-marathon-finistere` folder and `app.py` as the entry point
5. Deploy!

Your app will be available at: `https://your-app-name.streamlit.app`

### Hugging Face Spaces (Free / Gratuit)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select "Streamlit" as the SDK
3. Upload all files from the `semi-marathon-finistere` folder
4. The app will auto-deploy

### Local Docker

```bash
# Build the image
docker build -t race-planner .

# Run the container
docker run -p 8501:8501 race-planner
```

Example `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-streamlit.txt .
RUN pip install -r requirements-streamlit.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## License / Licence

MIT License

## Contributing / Contribution

Contributions welcome! Feel free to:

- Add support for other races
- Improve the pacing algorithms
- Translate to additional languages
- Report bugs or suggest features

---

_Happy running! / Bonne course!_
