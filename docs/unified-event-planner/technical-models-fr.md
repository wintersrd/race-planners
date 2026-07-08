# Modélisateur de Prédiction de Course — Référence Technique des Modèles

Ce document explique les principaux modèles mathématiques derrière le Modélisateur de Prédiction de Course. Il s'adresse aux coureurs curieux qui veulent comprendre les hypothèses, ainsi qu'aux développeurs qui souhaitent suivre la logique du code.

## Références et inspirations

L'outil n'est pas un moteur physiologique de laboratoire, mais il s'appuie sur des idées bien connues :

- les travaux de Minetti sur le coût énergétique de la course et de la marche en pente
- l'heuristique classique autour de `~1 kcal / kg / km`
- les modèles de pacing basés sur une interpolation entre des repères proches de LT1 et LT2
- les recommandations courantes sur les apports glucidiques en endurance, typiquement autour de `30-90 g / h`, limitées par la tolérance digestive et la durée de l'effort

Lorsque l'application utilise des heuristiques plutôt que des équations directement tirées de la littérature, ces heuristiques sont expliquées clairement ci-dessous.

---

## Modèle de Capacité Route

**Module** : `race_planners/road_capability.py`

Le modèle estime votre meilleur temps probable sur une distance donnée à partir de vos allures LT1 et LT2, puis applique un ajustement lié au parcours et à la météo.

### Table de fraction d'effort

Le modèle utilise une table qui relie la durée attendue de course à une fraction de l'écart entre LT1 et LT2.

En résumé :

1. on part d'une durée supposée
2. on récupère une fraction d'effort adaptée à cette durée
3. on interpole entre allure LT1 et allure LT2
4. on recalcule une nouvelle durée à partir de l'allure obtenue
5. on recommence jusqu'à stabilisation

Cette boucle converge vite parce que la fraction d'effort varie lentement avec la durée.

### Ajustement parcours et météo

Le temps optimal ajusté combine deux multiplicateurs :

- un multiplicateur de parcours, dérivé du coût GAP du profil altimétrique
- un multiplicateur météo, dérivé de la pénalité de chaleur moyenne sur la durée attendue

Ces deux coûts sont ensuite modulés par la tolérance au dénivelé et à la chaleur de l'athlète.

---

## Modèle Météo / Chaleur

**Module** : `race_planners/weather.py`

La pénalité de chaleur utilise une courbe simple :

- `<=10°C` : 0%
- `15°C` : 1.5%
- `20°C` : 3.5%
- `25°C` : 6.5%
- `28°C` : 8.5%
- `30°C` : 10%
- `35°C` : 15%
- au-dessus de `35°C` : +0.5% par degré supplémentaire

Entre ces points, l'application interpole linéairement.

### Courbe diurne

Au lieu d'appliquer une pénalité fixe sur toute la course, l'application estime la température effective à chaque moment :

- minimum thermique vers `04:00`
- amplitude journalière supposée de `12°C`
- heure de départ de l'événement utilisée comme ancre
- les ultras multi-jours réutilisent la courbe sur 24 heures via un modulo

La tolérance à la chaleur de l'athlète applique ensuite une réduction ou une amplification de cette pénalité.

---

## Fatigue, Dégradation et Durabilité

**Module** : `race_planners/fatigue.py`

### Fatigue

Chaque modèle de course applique une pente de fatigue croissante avec la progression dans l'événement.

### Profils de dégradation

Les profils de dégradation décrivent comment l'allure se détériore au fil de la journée via trois phases :

- début
- milieu
- fin

Les préréglages comme **Stable**, **Dégradation Progressive** ou **Risque d'Effondrement** ne sont donc pas de simples étiquettes marketing : ils correspondent à de vraies valeurs interpolées dans le moteur.

### Durabilité

La durabilité n'est pas modélisée comme un simple coefficient constant. Elle augmente avec :

- la progression dans l'épreuve
- la durée écoulée

Forme simplifiée :

- `progress_load = progress_ratio^2.4`
- `duration_load = elapsed_hours / 8.0`
- `breakdown_load = progress_load * duration_load`

Cela explique pourquoi l'effet est presque négligeable sur un semi mais devient énorme sur un ultra long.

---

## Garde-fou FC

**Module** : `race_planners/guardrails.py`

Le garde-fou FC estime la fréquence cardiaque segment par segment, puis ralentit l'allure si l'estimation dépasse un plafond dynamique.

### Estimation segmentaire de la FC

Le modèle utilise les repères LT1 / LT2 :

1. en dessous de LT1, la FC monte progressivement vers LT1
2. entre LT1 et LT2, la FC interpole entre les deux seuils
3. au-dessus de LT2, la FC grimpe plus vite

Comme le trail ralentit l'allure à effort constant, l'application convertit d'abord l'allure trail en équivalent route avant d'estimer la FC.

### Plafond dynamique

Le plafond varie selon la phase de course et la politique d'effort :

- **Conservateur** : plus bas tôt dans l'événement
- **Équilibré** : intermédiaire
- **Agressif** : plus permissif

Le terrain peut aussi abaisser temporairement le plafond sur les segments très raides.

---

## Modèle Nutrition / Hydratation

**Module** : `race_planners/fueling.py`

### Dépense énergétique

L'approximation de base est :

`kcal ≈ poids(kg) × distance(km) × multiplicateur_de_pente`

Le multiplicateur de pente augmente fortement en montée, baisse légèrement en descente modérée, et reste borné pour éviter des valeurs absurdes.

### Cibles glucidiques

Les cibles glucidiques sont liées à la durée :

- effort court : peu ou pas de glucides nécessaires
- effort moyen : augmentation progressive
- effort très long : légère baisse des objectifs théoriques, car la tolérance digestive devient limitante

### Hydratation

Le taux de sudation peut être saisi manuellement, sinon il est estimé à partir de la température.

### Gels et fenêtres de prise

Les recommandations utilisent des tailles de gels réalistes (`30g` et `50g`).

L'application évite aussi de recommander des glucides :

- au tout début de la course (fenêtre de grâce de départ)
- dans les toutes dernières minutes, quand l'effet physiologique serait trop tardif

### Types de ravitaillement

- **Eau** : pas d'apport calorique significatif
- **Eau + snacks** : apport modéré
- **Ravitaillement complet** : possibilité de consommer beaucoup plus, que ce soit la nourriture de la course ou votre propre matériel

---

## GAP et coût de pente

**Modules** : `race_planners/grade.py`, `race_planners/pacing.py`

Le GAP convertit les pentes montantes et descendantes en équivalents d'effort sur plat.

- les montées ralentissent progressivement l'allure
- les descentes modérées peuvent l'accélérer un peu
- les descentes très raides la ralentissent à nouveau à cause du freinage musculaire et de l'impact

Cela permet de comparer l'effort sur un parcours vallonné comme si on le projetait sur une route plate.

---

## Ce qui est mesuré et ce qui est heuristique

Le modèle combine trois catégories de données :

1. **Données mesurées / spécifiques à l'athlète**
   - FC LT1 / LT2
   - allures LT1 / LT2
   - meilleurs temps probables
   - ralentissements trail
   - tolérance digestive / sudation

2. **Données dérivées du parcours**
   - distance GPX
   - dénivelé positif / négatif
   - distribution des pentes
   - positions et types de ravitaillements

3. **Heuristiques / modélisation**
   - profils de dégradation
   - courbe de pénalité de chaleur
   - offsets du garde-fou FC
   - fenêtres de nutrition de début / fin d'épreuve

Cette combinaison est volontaire. L'objectif n'est pas de prétendre à une vérité physiologique parfaite, mais de rendre les hypothèses explicites et utiles pour la planification.
