# Modélisateur de Prédiction de Course — Guide Utilisateur

## À quoi sert cet outil

Le Modélisateur de Prédiction de Course aide à estimer l'allure, le temps final, le passage aux ravitaillements, la nutrition et la structure des splits pour des événements route et trail pris en charge.

Ce n'est pas seulement un calculateur de splits. Il combine :

- les données du parcours
- les données de référence du profil athlète
- les hypothèses météo
- des modèles d'allure spécifiques au type d'événement
- une logique de nutrition et d'hydratation

## Lancement de l'application

```bash
uv run streamlit run semi-marathon-finistere/app.py
```

L'application s'ouvre dans votre navigateur à `http://localhost:8501`.

## Flux recommandé

### 1. Choisir la langue

Utilisez le sélecteur **Langue** en haut de la barre latérale.

### 2. Lire d'abord le guide

Près du haut de la barre latérale, vous trouverez le **Guide Utilisateur**.

Il explique comment utiliser le planificateur. La section séparée **Comment Fonctionnent les Modèles** n'est là que si vous voulez un niveau de détail technique plus poussé.

### 3. Remplir le Profil de l'Athlète avant de planifier sérieusement

La section **Profil de l'Athlète** est l'une des parties les plus importantes de l'outil.

Ajoutez les données de référence que vous connaissez réellement, surtout :

- **FC LT1** et **allure LT1 sur route**
- **FC LT2** et **allure LT2 sur route**
- **temps optimaux probables** sur semi et/ou marathon si vous les connaissez
- **valeurs prédictives** issues de COROS, Strava, Intervals.icu ou d'un entraîneur si vous leur faites confiance
- **ralentissement trail plat** et **ralentissement trail technique** si vous savez comment le trail affecte votre allure
- **poids corporel** pour les calculs nutritionnels
- **durabilité**, **tolérance à la chaleur** et **tolérance au dénivelé** pour refléter votre comportement dans les événements longs ou difficiles

### 4. Appliquer le profil à l'événement sélectionné

Après avoir modifié le profil, cliquez sur **Appliquer les Valeurs par Défaut à cet Événement**.

Cette étape est importante parce que le planificateur ne réécrit pas automatiquement la configuration de l'événement courant chaque fois que vous modifiez le profil. Le bouton demande à l'application de :

- réinitialiser les valeurs par défaut de l'événement courant à partir du profil
- mettre à jour les références route, les ancres trail, le profil de dégradation et la disponibilité du garde-fou FC
- effacer le résultat précédent pour que le prochain calcul reflète bien les nouvelles hypothèses

Si vous sautez cette étape, l'événement courant peut encore utiliser d'anciennes valeurs par défaut.

### 5. Sélectionner un événement

Choisissez un événement pris en charge dans le menu déroulant de la barre latérale. Le planificateur charge automatiquement le bon parcours, le bon terrain et le bon modèle d'allure.

Événements actuellement pris en charge :

- **Semi-Marathon du Finistere** — demi-marathon route
- **Marathon des Etoiles de la Baie** — marathon route
- **Grand Raid du Finistere 56 / 92 / 166** — ultras trail
- **Trail de l'Odet Ultra** — ultra trail

### 6. Choisir le mode de ciblage

- **Temps Cible** : vous donnez un temps d'arrivée objectif et le planificateur dérive l'allure
- **Ancrage d'Effort** : vous donnez des allures connues et le planificateur construit l'événement à partir d'elles

Les événements route se prêtent généralement bien au mode **Temps Cible**. Les événements trail et ultra se prêtent souvent mieux au mode **Ancrage d'Effort**.

## Contrôles selon le type d'événement

### Événements sur route

Les événements route montrent un ensemble de contrôles plus simple :

- **Intention de Course**
- **Biais de Split**
- **Temps d'Arrêt Ravito (sec)**
- **Température Maximale Prévue**

Ces événements utilisent une configuration d'allure plus simple que les événements trail et ultra.

### Événements trail / ultra

Les événements trail et ultra exposent davantage de contrôles liés au terrain :

- **Seuil Marche/Course**
- **Prudence en Descente**
- **Profil de Dégradation**
- **Temps d'Arrêt Ravito (min)** — avec durées par type de station (eau seule, standard, ravitaillement complet)
- **Température Maximale Prévue**
- **Contrôles Trail Avancés** optionnels comme **Politique d'Effort**, **Utiliser le Garde-fou FC Dérivé** et **Poids Moyen Transporté**

## Signification des principaux contrôles

### Intention de Course

Pour la route, ce contrôle décale la cible par rapport au meilleur temps probable ajusté.

- **À Fond** : courir proche du plafond de capacité ajustée
- **Costaud** : gros effort avec un peu de marge
- **Contrôlé** : courir sérieusement mais proprement
- **Durable** : finir clairement en dessous de la zone rouge

### Biais de Split

Contrôle réservé à la route.

- valeurs négatives : économie au début, finish plus fort
- zéro : plus proche d'une allure régulière
- valeurs positives : effort plus chargé au départ

### Profil de Dégradation

Contrôle réservé au trail/ultra.

Il décrit à quel point l'allure devrait se dégrader au fil de la journée.

- **Stable** : faible dégradation
- **Dégradation Tardive** : début correct, effondrement plus tardif
- **Dégradation Progressive** : dégradation continue
- **Risque d'Effondrement** : hypothèse pessimiste de grosse dérive

### Politique d'Effort

Contrôle trail avancé.

Il influence l'attitude générale d'allure et interagit avec la logique de garde-fou FC dérivé.

- **Conservateur**
- **Équilibré**
- **Agressif**

### Garde-fou FC Dérivé

Contrôle trail avancé.

Il n'est utile que si vous avez saisi suffisamment de données dans le profil athlète pour que le planificateur puisse estimer la fréquence cardiaque de façon crédible. L'application vous le signale si les données LT sont manquantes.

### Température Maximale Prévue

Le planificateur utilise la température maximale prévue et l'heure de départ pour estimer une courbe de chaleur diurne sur toute la course.

### Poids Moyen Transporté

Contrôle trail avancé.

Saisissez le poids moyen du sac, de l'eau, de la nourriture et du matériel que vous prévoyez de transporter pendant l'événement. Il ne s'agit pas du poids de départ, mais de votre meilleure estimation de ce que vous portez en moyenne entre les ravitaillements.

Le planificateur utilise cette valeur pour augmenter le coût métabolique de chaque kilomètre. Par exemple, un coureur de 75 kg portant 3 kg ajoute environ 3,2 % au coût énergétique de chaque kilomètre, ce qui se traduit directement par une allure plus lente.

Nécessite le **Poids Corporel** dans le Profil de l'Athlète pour prendre effet.

## Lire les résultats

### Résumé

Affiche :

- distance
- temps total
- temps en mouvement
- temps d'arrêt
- allure moyenne

### Profil du Parcours

Affiche :

- le profil altimétrique avec coloration du terrain
- le profil d'allure avec bandes colorées selon la vitesse relative
- la courbe de temps cumulé

### Ravitaillements

Affiche :

- l'heure d'arrivée et de départ
- le temps écoulé à l'arrivée et au départ
- le split et l'allure jusqu'à chaque ravitaillement
- le type de ravitaillement / l'expérience attendue sur place

### Sections

Affiche des sections adaptées au terrain avec :

- début / fin de section
- dénivelé positif et négatif
- allure et durée de section

### Nutrition

Affiche :

- calories totales
- cible glucidique
- cible hydrique
- recommandations d'emport par bloc
- contribution des ravitaillements
- déficit glucidique cumulé

### Splits

Affiche l'allure par blocs de distance configurables (1 / 2 / 5 / 10 km selon la longueur de l'événement).

### Analyse des Splits

Affiche :

- comparaison première moitié / deuxième moitié
- répartition montée / plat / descente en distance et en temps

## Sauvegarde et rechargement

### Plan JSON

Utilisez **Télécharger le Plan JSON** pour sauvegarder le plan courant. Rechargez-le ensuite depuis **Charger un Plan Sauvegardé**.

### Profil Athlète JSON

Utilisez **Télécharger le Profil JSON** pour sauvegarder votre profil séparément. C'est utile pour réutiliser votre profil sur plusieurs événements ou plusieurs machines.

## Conseils pratiques

- saisissez d'abord le profil athlète si vous avez les données
- cliquez sur **Appliquer les Valeurs par Défaut à cet Événement** après l'avoir modifié
- utilisez les contrôles propres au type d'événement au lieu d'essayer de forcer un modèle route sur un ultra ou l'inverse
- pour les ultras, raisonnez en **terrain + dégradation + chaleur + nutrition**, pas seulement en allure cible
