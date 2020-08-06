FROM buildpack-deps:bionic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update &&     apt-get -qq install --yes --no-install-recommends locales > /dev/null &&     apt-get -qq purge &&     apt-get -qq clean &&     rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen &&     locale-gen

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV SHELL /bin/bash

ARG NB_USER
ARG NB_UID


ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}
RUN groupadd         --gid ${NB_UID}         ${NB_USER} &&     useradd         --comment "Default user"         --create-home         --gid ${NB_UID}         --no-log-init         --shell /bin/bash         --uid ${NB_UID}         ${NB_USER}

RUN wget --quiet -O - https://deb.nodesource.com/gpgkey/nodesource.gpg.key |  apt-key add - &&     DISTRO="bionic" &&     echo "deb https://deb.nodesource.com/node_10.x $DISTRO main" >> /etc/apt/sources.list.d/nodesource.list &&     echo "deb-src https://deb.nodesource.com/node_10.x $DISTRO main" >> /etc/apt/sources.list.d/nodesource.list

RUN apt-get -qq update &&   apt-get -qq install --yes --no-install-recommends        less        nodejs        unzip        > /dev/null &&     apt-get -qq purge &&     apt-get -qq clean &&     rm -rf /var/lib/apt/lists/*

EXPOSE 8888 5064 5065

ENV APP_BASE /srv
ENV NPM_DIR ${APP_BASE}/npm
ENV NPM_CONFIG_GLOBALCONFIG ${NPM_DIR}/npmrc
ENV CONDA_DIR ${APP_BASE}/conda

ENV NB_PYTHON_PREFIX ${CONDA_DIR}/envs/notebook
ENV KERNEL_PYTHON_PREFIX ${NB_PYTHON_PREFIX}
ENV PATH ${NB_PYTHON_PREFIX}/bin:${CONDA_DIR}/bin:${NPM_DIR}/bin:${PATH}

RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml


RUN git clone https://github.com/jupyter/repo2docker.git /tmp/repo2docker && cd /tmp/repo2docker && git checkout 0.10.0

RUN cp -r /tmp/repo2docker/repo2docker/buildpacks/conda /conda
RUN cp conda/activate-conda.sh /etc/profile.d/activate-conda.sh
RUN cp conda/environment.py-3.6.frozen.yml /tmp/environment.yml
RUN cp conda/install-miniconda.bash /tmp/install-miniconda.bash

RUN mkdir -p ${NPM_DIR} && chown -R ${NB_USER}:${NB_USER} ${NPM_DIR}

USER ${NB_USER}

RUN npm config --global set prefix ${NPM_DIR}

USER root
RUN bash /tmp/install-miniconda.bash && \
rm /tmp/install-miniconda.bash /tmp/environment.yml

ARG REPO_DIR=${HOME}

COPY environment.yml ${REPO_DIR}/environment.yml

ENV REPO_DIR ${REPO_DIR}

WORKDIR ${REPO_DIR}

ENV PATH ${HOME}/.local/bin:${REPO_DIR}/.local/bin:${PATH}
ENV PATH=/root/anaconda/bin:$PATH
ENV CONDA_DEFAULT_ENV ${KERNEL_PYTHON_PREFIX}

USER root

RUN chown -R ${NB_USER}:${NB_USER} ${REPO_DIR}
RUN apt-get -qq update && apt-get install --yes --no-install-recommends net-tools && apt-get -qq purge && apt-get -qq clean && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

RUN conda env update -p ${NB_PYTHON_PREFIX} -f "environment.yml" && conda clean --all -f -y && conda list -p ${NB_PYTHON_PREFIX}

USER root

COPY . ${REPO_DIR}

RUN chown -R ${NB_USER}:${NB_USER} ${REPO_DIR}

LABEL repo2docker.ref="None"
LABEL repo2docker.repo="local"
LABEL repo2docker.version="0.11.0"

RUN cp /tmp/repo2docker/repo2docker/buildpacks/repo2docker-entrypoint /usr/local/bin/repo2docker-entrypoint

USER ${NB_USER}

RUN chmod +x postBuild
RUN ./postBuild
RUN chmod +x "${REPO_DIR}/start"

ENV R2D_ENTRYPOINT "${REPO_DIR}/start"

ENTRYPOINT ["/usr/local/bin/repo2docker-entrypoint"]

CMD ["jupyter", "notebook", "--ip", "127.0.0.1"]
