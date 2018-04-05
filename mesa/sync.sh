#!/bin/bash

rsync -zvr --max-size=100m --exclude "*~" \
perseus:/tigress/adrianp/projects/tidal-circularization/mesa/0.8Msun/LOGS ~/projects/tidal-circularization/mesa/0.8Msun/

rsync -zvr --max-size=100m --exclude "*~" \
perseus:/tigress/adrianp/projects/tidal-circularization/mesa/1.0Msun/LOGS ~/projects/tidal-circularization/mesa/1.0Msun/

rsync -zvr --max-size=100m --exclude "*~" \
perseus:/tigress/adrianp/projects/tidal-circularization/mesa/1.2Msun/LOGS ~/projects/tidal-circularization/mesa/1.2Msun/

rsync -zvr --max-size=100m --exclude "*~" \
perseus:/tigress/adrianp/projects/tidal-circularization/mesa/1.4Msun/LOGS ~/projects/tidal-circularization/mesa/1.4Msun/

rsync -zvr --max-size=100m --exclude "*~" \
perseus:/tigress/adrianp/projects/tidal-circularization/mesa/1.6Msun/LOGS ~/projects/tidal-circularization/mesa/1.6Msun/

rsync -zvr --max-size=100m --exclude "*~" \
perseus:/tigress/adrianp/projects/tidal-circularization/mesa/1.8Msun/LOGS ~/projects/tidal-circularization/mesa/1.8Msun/

rsync -zvr --max-size=100m --exclude "*~" \
perseus:/tigress/adrianp/projects/tidal-circularization/mesa/2.0Msun/LOGS ~/projects/tidal-circularization/mesa/2.0Msun/

rsync -zvr --max-size=100m --exclude "*~" \
perseus:/tigress/adrianp/projects/tidal-circularization/mesa/2.5Msun/LOGS ~/projects/tidal-circularization/mesa/2.5Msun/

rsync -zvr --max-size=100m --exclude "*~" \
perseus:/tigress/adrianp/projects/tidal-circularization/mesa/3.0Msun/LOGS ~/projects/tidal-circularization/mesa/3.0Msun/
