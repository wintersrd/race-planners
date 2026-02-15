# Race Planner: Semi-Marathon du Finistère

A bilingual (English/French) interactive race pacing tool for the Semi-Marathon du Finistère.

---

## English

### Overview

This tool helps runners plan their race strategy for the Semi-Marathon du Finistère (21.1 km with ~153m elevation gain). It uses **Grade-Adjusted Pace (GAP)** to account for elevation changes, providing realistic pace targets for each kilometer.

Two versions are available:
- **Streamlit Web App** (Recommended) - Fast, modern web interface
- **Jupyter Notebook** - Original version for Voila

### Features

- Bilingual interface (English/French toggle)
- Target finish time OR target pace input
- Power fade adjustment for negative/positive splits
- Rest stop timing and strategy
- Course-specific pacing tips for each section
- Printable pocket card and wristband reference
- Elevation profile visualization
- Per-kilometer pace breakdown

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

4. **Install Streamlit dependencies:**
   ```bash
   pip install -r requirements-streamlit.txt
   ```

5. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
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

| Attribute | Value |
|-----------|-------|
| Distance | 21.06 km |
| Elevation Gain | ~153m |
| Rest Stops | 5.3 km, 9.1 km, 14.5 km |

### How to Use

1. **Select your language** (English or French) using the dropdown
2. **Choose input mode:** Target finish time OR target average pace
3. **Enter your goal** (e.g., `01:45:00` for finish time or `05:00` for pace per km)
4. **Adjust power fade:**
   - Negative values = negative split (slower start, faster finish)
   - Positive values = positive split (faster start, slower finish)
   - Zero = even pacing
5. **Set rest duration** (seconds per water station)
6. **Click "Calculate Pacing"** to see your personalized plan
7. **Print or screenshot** the Quick Reference section for race day

---

## Français

### Aperçu

Cet outil aide les coureurs à planifier leur stratégie pour le Semi-Marathon du Finistère (21,1 km avec ~153m de dénivelé positif). Il utilise l'**Allure Ajustée au Dénivelé (GAP)** pour tenir compte des variations d'élévation, fournissant des objectifs d'allure réalistes pour chaque kilomètre.

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

4. **Installez les dépendances Streamlit:**
   ```bash
   pip install -r requirements-streamlit.txt
   ```

5. **Lancez l'application Streamlit:**
   ```bash
   streamlit run app.py
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

| Attribut | Valeur |
|----------|--------|
| Distance | 21,06 km |
| Dénivelé positif | ~153m |
| Ravitaillements | 5,3 km, 9,1 km, 14,5 km |

### Comment Utiliser

1. **Sélectionnez votre langue** (Anglais ou Français) avec le menu déroulant
2. **Choisissez le mode de saisie:** Temps cible OU allure moyenne cible
3. **Entrez votre objectif** (ex: `01:45:00` pour le temps ou `05:00` pour l'allure par km)
4. **Ajustez la gestion d'effort:**
   - Valeurs négatives = split négatif (départ lent, fin rapide)
   - Valeurs positives = split positif (départ rapide, fin lente)
   - Zéro = allure constante
5. **Définissez la durée des ravitos** (secondes par station)
6. **Cliquez "Calculer"** pour voir votre plan personnalisé
7. **Imprimez ou capturez** la section Référence Rapide pour le jour de course

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

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application / Application web Streamlit |
| `race_planner_semi_marathon_finistere.ipynb` | Original Jupyter notebook / Notebook Jupyter original |
| `WR-GPX-Semi-marathon-du-Finistere.gpx` | Course GPX data / Données GPX du parcours |
| `requirements-streamlit.txt` | Streamlit dependencies / Dépendances Streamlit |
| `requirements.txt` | Voila dependencies / Dépendances Voila |
| `.streamlit/config.toml` | Streamlit configuration / Configuration Streamlit |
| `README.md` | This file / Ce fichier |

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

*Happy running! / Bonne course!*
