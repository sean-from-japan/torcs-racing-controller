# A derived image: the organiser's competition container, plus everything this
# repository needs to drive a measured lap without reaching the network.
#
# Build it with container/build.sh, which supplies BASE_IMAGE from
# image-lock.json for the host architecture.  The default below is the arm64
# digest; CI checks that it still matches the lock.
#
# What this Dockerfile does NOT claim: the base image's own build definition is
# not public, so this image is *derived from* a pinned base, not reproducible
# from source end to end.  Everything in this file is verifiable; the base is
# identified, not rebuilt.  See container/README.md.
#
# Build locally.  Do not push it: that would redistribute the organiser's image.

ARG BASE_IMAGE=docker.io/johnsloe/torcs-competition@sha256:fed57137c217a524ead2281dd085179066be373c1ffbc8c82880ee85ae07c4b9
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
LABEL org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.title="torcs-racing-controller runner" \
      org.opencontainers.image.description="Corkscrew CMA-ES controller, ready to race"

# gym is the one Python dependency the base image does not ship.  Baking it in
# removes a PyPI fetch from every run; the base has gymnasium, not gym, and the
# bridge imports gym.
USER root
RUN python3 -m pip install --no-cache-dir "gym==0.26.2"

COPY --chown=student:student container/prepare_bridge.py \
                             container/configure_race.py \
                             container/image-lock.json \
                             /opt/torcs-repro/

USER student

# Build the gym_torcs bridge from the pinned upstream commit and keep only the
# result.  After this the image carries the exact bridge the lap was measured
# against, and a run needs no GitHub access.
RUN set -eux; \
    commit="$(python3 -c "import json;print(json.load(open('/opt/torcs-repro/image-lock.json'))['upstream_bridge']['commit'])")"; \
    git clone --quiet https://github.com/ugo-nama-kun/gym_torcs.git /tmp/gym_torcs_src; \
    git -C /tmp/gym_torcs_src checkout --quiet "$commit"; \
    python3 /opt/torcs-repro/prepare_bridge.py \
        --src /tmp/gym_torcs_src \
        --out /home/student/gym_torcs; \
    rm -rf /tmp/gym_torcs_src

# TORCS writes ~/.torcs on its first run.  `-h` is enough to trigger that and
# needs no X display, so the race configuration can be baked in too.
RUN /usr/local/torcs/bin/torcs -h >/dev/null 2>&1 || true; \
    python3 /opt/torcs-repro/configure_race.py

ENV GYM_TORCS_DIR=/home/student/gym_torcs \
    TORCS_BRIDGE_PREBUILT=1

# The base image's own entrypoint (the desktop/noVNC startup) is inherited.
