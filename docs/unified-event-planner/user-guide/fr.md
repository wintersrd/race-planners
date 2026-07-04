# Planificateur d'Événements Unifié — Guide Utilisateur

## Démarrage

Le Planificateur d'Événements Unifié est une application Streamlit qui aide les coureurs à planifier l'allure, la nutrition et le chronométrage pour les événements sur route et trail.

### Lancement de l'application

```bash
uv run streamlit run semi-marathon-finistere/app.py
```

L'application s'ouvre dans votre navigateur à l'adresse `http://localhost:8501`.

## Mode d'Emploi

### 1. Sélectionner un Événement

Choisissez un événement pris en charge dans le menu déroulant de la barre latérale. Le planificateur charge automatiquement le parcours, le modèle d'allure et les paramètres de terrain corrects. Les événements disponibles incluent :

- **Semi-Marathon du Finistere** — demi-marathon sur route
- **Marathon des Etoiles de la Baie** — marathon sur route
- **Grand Raid du Finistere 56/92/166** — ultras trail
- **Trail de l'Odet Ultra** — ultra trail

### 2. Choisir le Mode de Ciblage

- **Temps Cible** : Entrez un temps d'arrivée objectif. Le planificateur dérive toute l'allure de cette cible.
- **Ancrage d'Effort** : Entrez vos allures d'entraînement connues. Le planificateur construit un plan à partir de vos références.

Les deux modes fonctionnent pour tous les événements. Les événements sur route utilisent le Temps Cible par défaut. Les événements trail utilisent l'Ancrage d'Effort par défaut.

### 3. Ajuster les Contrôles de Stratégie

**Événements sur route** :

- **Intention de Course** : À quel point vous voulez courir agressivement (À Fond → Durable)
- **Biais de Split** : Économiser au début (négatif) ou charger au départ (positif)
- **Temps d'Arrêt Ravito** : Secondes par ravitaillement pour l'eau ou marche brève

**Événements trail/ultra** :

- **Seuil Marche/Course** : Pourcentage de pente où marcher devient plus efficace
- **Prudence en Descente** : À quel point vous descendez prudemment le terrain technique
- **Profil de Dégradation** : Importance de la dégradation d'allure pendant l'événement (Stable → Risque d'Effondrement)
- **Temps d'Arrêt Ravito** : Minutes par ravitaillement

### 4. Définir les Conditions Météorologiques

Entrez la température maximale prévue. Le planificateur modélise une courbe de température diurne à partir de l'heure de départ et applique une pénalité d'allure quand il fait chaud.

### 5. Configurer le Profil de l'Athlète (Optionnel)

La section Profil de l'Athlète vous permet de saisir :

- **FC LT1/LT2 et allures** : Vos fréquences cardiaques aux seuils lactiques et allures sur route
- **Capacité sur route** : Vos meilleurs temps optimaux sur demi-marathon ou marathon
- **Valeurs prédites** : Estimations externes de COROS, Strava ou Intervals.icu
- **Ajustements trail** : À quel point le trail plat et technique est plus lent pour vous vs route
- **Poids corporel** : Utilisé pour les calculs caloriques et nutritionnels
- **Durabilité/Tolérance Chaleur/Dénivelé** : Facteurs universels affectant la performance

Le profil est sauvegardé dans les exports JSON du plan et utilisé pour initialiser des valeurs par défaut plus intelligentes.

### 6. Contrôles Trail Avancés (Événements trail uniquement)

- **Politique d'Effort** : Attitude d'allure Conservatrice, Équilibrée ou Agressive
- **Garde-fou FC** : Plafond de fréquence cardiaque dérivé optionnel basé sur votre profil LT1/LT2

### 7. Calculer et Lire les Résultats

Cliquez sur **Calculer le Plan** pour générer :

- **Résumé** : Distance, temps total, temps en mouvement, temps de ravito, allure moyenne
- **Profil du Parcours** : Profil altimétrique et histogramme d'allure par kilomètre
- **Ravitaillements** : Heures d'arrivée/départ à chaque station (heure réelle et temps écoulé)
- **Sections** : Segments d'allure adaptés au terrain avec dénivelé positif/négatif
- **Nutrition** : Cibles caloriques, glucidiques et hydriques par bloc avec recommandations de gels
- **Splits** : Allure par blocs de distance (1/2/5/10 km selon la longueur de l'événement)

### 8. Sauvegarder et Recharger des Plans

- Cliquez sur **Télécharger le Plan JSON** pour sauvegarder votre plan
- Utilisez **Charger un Plan Sauvegardé** dans la barre latérale pour recharger un plan
- Le profil athlète est inclus dans les exports JSON du plan

## Bascule de Langue

Utilisez le bouton radio **Langue** en haut de la barre latérale pour basculer entre l'anglais et le français. Toutes les étiquettes, textes d'aide et messages se traduisent instantanément.
